"""Model wrapper for activation vector injection."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from verl.nla.utils.injection_manager import InjectionTokenManager

# Try to import GenerationMixin for proper generation support
try:
    from transformers.generation import GenerationMixin

    class BaseWrapper(GenerationMixin, nn.Module):
        """Base class with generation support."""
        pass
except ImportError:
    # Fallback if transformers not available
    BaseWrapper = nn.Module


@dataclass
class InjectionConfig:
    """Configuration for activation injection."""

    mode: str = "replace"  # "replace", "add", or "project"
    layer_indices: List[int] = None  # Which layers to inject at
    projection_dim: Optional[int] = None  # For learnable projection
    injection_token: Optional[str] = None  # Optional specific token (auto-selected if None)
    # These will be set by InjectionTokenManager:
    injection_token_id: Optional[int] = None
    injection_character: Optional[str] = None

    def __post_init__(self):
        if self.layer_indices is None:
            self.layer_indices = [0]  # Default to embedding layer


class NLAModelWrapper(BaseWrapper):
    """Wrapper for models to support activation vector injection.

    Inherits from GenerationMixin when available for seamless generation support.
    Acts as a transparent proxy to the base model, intercepting only the methods
    needed for activation injection.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any = None,
        injection_config: Optional[InjectionConfig] = None,
        hidden_dim: int = None,
        activation_dim: int = None,
    ):
        super().__init__()
        # Register base_model as a proper submodule so its parameters are accessible
        self.base_model = base_model
        object.__setattr__(self, "injection_config", injection_config or InjectionConfig())
        object.__setattr__(self, "hidden_dim", hidden_dim or self._infer_hidden_dim())
        object.__setattr__(self, "activation_dim", activation_dim or self.hidden_dim)

        # Also set the model config for transformers compatibility
        if hasattr(base_model, "config"):
            object.__setattr__(self, "config", base_model.config)

        # Set up injection token management
        if tokenizer is not None:
            # Use InjectionTokenManager for consistent token selection
            injection_manager = InjectionTokenManager(tokenizer, self.injection_config.injection_token)
            # Update injection config with auto-selected token info
            self.injection_config.injection_token_id = injection_manager.token_id
            self.injection_config.injection_character = injection_manager.character
            object.__setattr__(self, "injection_manager", injection_manager)
        elif self.injection_config.injection_token_id is not None and self.injection_config.injection_token_id >= 0:
            # Token ID was manually specified - use it directly
            object.__setattr__(self, "injection_manager", None)
        else:
            raise ValueError(
                "Either provide a tokenizer for auto-selection or specify injection_token_id in config. "
                "The tokenizer ensures consistency with the dataset's injection token."
            )

        # Initialize projection layer if needed
        if self.injection_config.mode == "project" and self.injection_config.projection_dim:
            projection_layer = nn.Linear(
                self.activation_dim,
                self.injection_config.projection_dim or self.hidden_dim
            )
            self.add_module('activation_proj', projection_layer)
        else:
            self.activation_proj = None

        # Register hooks for multi-layer injection
        self._register_injection_hooks()

        # Temporary storage for activation vectors during injection
        # Only used within methods, not for state management
        self._current_activations = None
        self._injection_positions = None

    def _infer_hidden_dim(self) -> int:
        """Infer hidden dimension from base model."""
        # Try common attribute names
        if hasattr(self.base_model, "config"):
            for attr in ["hidden_size", "d_model", "embed_dim", "hidden_dim"]:
                if hasattr(self.base_model.config, attr):
                    return getattr(self.base_model.config, attr)

        # Try to get from embeddings
        if hasattr(self.base_model, "get_input_embeddings"):
            embed_layer = self.base_model.get_input_embeddings()
            if hasattr(embed_layer, "embedding_dim"):
                return embed_layer.embedding_dim
            elif hasattr(embed_layer, "weight"):
                return embed_layer.weight.shape[1]

        raise ValueError("Could not infer hidden dimension from base model")

    def _register_injection_hooks(self):
        """Register forward hooks for multi-layer injection."""
        self._hooks = []

        if len(self.injection_config.layer_indices) > 1 or 0 not in self.injection_config.layer_indices:
            # Need hooks for layers beyond embedding
            for layer_idx in self.injection_config.layer_indices:
                if layer_idx > 0:
                    hook = self._create_injection_hook(layer_idx)
                    # Register hook on appropriate layer
                    # This will vary by model architecture
                    self._hooks.append(hook)

    def _create_injection_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if self._current_activations is not None and self._injection_positions is not None:
                # Perform injection at this layer
                output = self._inject_at_layer(output, layer_idx)
            return output
        return hook

    def _inject_at_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Inject activation vectors at a specific layer."""
        batch_size = hidden_states.shape[0]

        for batch_idx in range(batch_size):
            if batch_idx >= len(self._injection_positions):
                continue

            positions = self._injection_positions[batch_idx]
            if positions is None or positions.numel() == 0:
                continue

            # positions is now a tensor
            for pos in positions.tolist():  # Convert to list for iteration
                if batch_idx < self._current_activations.shape[0]:
                    activation = self._current_activations[batch_idx]

                    # Apply projection if configured
                    if self.activation_proj is not None:
                        activation = self.activation_proj(activation)

                    # Perform injection based on mode
                    if self.injection_config.mode == "replace":
                        hidden_states[batch_idx, pos] = activation
                    elif self.injection_config.mode == "add":
                        hidden_states[batch_idx, pos] = hidden_states[batch_idx, pos] + activation

        return hidden_states

    def _find_injection_positions(
        self,
        input_ids: torch.Tensor
    ) -> List[Optional[torch.Tensor]]:
        """Find positions of injection tokens in input - vectorized implementation."""
        positions = []

        for i in range(input_ids.shape[0]):
            # Find all occurrences of the injection token for the current batch item
            batch_positions = (input_ids[i] == self.injection_config.injection_token_id).nonzero(as_tuple=False).squeeze(-1)
            # Store as tensor (empty tensor if no matches)
            positions.append(batch_positions if batch_positions.numel() > 0 else None)

        return positions


    def __getattr__(self, name):
        """Delegate unknown attributes to base model."""
        # Check if base_model exists in instance dictionary
        if "base_model" not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Check if it's in our modules first (for dynamically added modules)
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]

        # Delegate to base model
        return getattr(self.base_model, name)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        activation_vectors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Forward pass with optional activation injection.

        Stateless implementation: injection happens when activation_vectors is provided
        and past_key_values is None (first forward pass in generation).
        """
        # Injection happens only on the first forward pass of generation
        # (when past_key_values is None) or during a normal forward pass
        should_inject = activation_vectors is not None and past_key_values is None

        # Find injection positions if needed
        injection_positions = None
        if should_inject and input_ids is not None:
            injection_positions = self._find_injection_positions(input_ids)

        # Handle injection at embedding layer
        if should_inject and 0 in self.injection_config.layer_indices and input_ids is not None and injection_positions:
            # Get input embeddings
            if hasattr(self.base_model, "get_input_embeddings"):
                embed_layer = self.base_model.get_input_embeddings()
                inputs_embeds = embed_layer(input_ids)

                # Temporarily store state for _inject_at_layer
                self._current_activations = activation_vectors
                self._injection_positions = injection_positions

                # Inject at embedding layer
                inputs_embeds = self._inject_at_layer(inputs_embeds, 0)

                # Clear temporary state
                self._current_activations = None
                self._injection_positions = None

                # Forward with embeddings instead of input_ids
                return self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs
                )

        # Standard forward pass
        if inputs_embeds is not None:
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs
            )
        else:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs
            )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation - delegate to base model."""
        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        # Minimal default implementation
        return {"input_ids": kwargs.get("input_ids", args[0] if args else None)}

    def can_generate(self) -> bool:
        """Check if model can generate."""
        return hasattr(self.base_model, "generate")

    def _reorder_cache(self, *args, **kwargs):
        """Reorder cache for beam search - delegate to base model."""
        if hasattr(self.base_model, "_reorder_cache"):
            return self.base_model._reorder_cache(*args, **kwargs)
        return None

    def generate(
        self,
        input_ids: torch.Tensor = None,
        activation_vectors: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,  # Support both input formats
        **kwargs
    ) -> torch.Tensor:
        """Generate with activation injection - stateless implementation.

        Passes activation_vectors through kwargs so it's available in forward.
        """
        # Handle input format variations
        if input_ids is None and inputs is not None:
            input_ids = inputs

        # Check if we inherited from GenerationMixin
        if not hasattr(super(), "generate"):
            raise RuntimeError(
                "Generation with activation injection requires transformers to be installed. "
                "Install it with: pip install transformers"
            )

        # Pass activation_vectors through kwargs - it will be available in forward
        return super().generate(
            inputs=input_ids,
            activation_vectors=activation_vectors,
            **kwargs
        )