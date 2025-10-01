"""Model wrapper for activation vector injection."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from torch.utils.hooks import RemovableHandle
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

    # Required by HuggingFace transformers generation utils
    _is_stateful = False

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

        # Register hooks for injection
        self._register_injection_hooks()

        # Temporary storage for activation vectors/positions during a forward
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

    def _register_injection_hooks(self) -> None:
        """Register forward hooks for injection layers."""
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        extra_layers = [idx for idx in self.injection_config.layer_indices if idx != 0]
        if extra_layers:
            raise NotImplementedError(
                "Injection for layers beyond embeddings is not yet implemented under FSDP."
            )

        if 0 in self.injection_config.layer_indices:
            embedding_layer = self.base_model.get_input_embeddings()
            if embedding_layer is None:
                raise ValueError("Base model does not expose an input embedding layer.")
            handle = embedding_layer.register_forward_hook(self._embedding_forward_hook)
            self._hook_handles.append(handle)

    def _embedding_forward_hook(self, module, inputs, output):
        if self._current_activations is None or self._injection_positions is None:
            return output
        return self._inject_hidden_states(output)

    def _inject_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply activation injections in-place on hidden states."""
        batch_size = hidden_states.shape[0]

        for batch_idx in range(batch_size):
            if batch_idx >= len(self._injection_positions):
                continue

            positions = self._injection_positions[batch_idx]
            if positions is None or positions.numel() == 0:
                continue

            for pos in positions.tolist():
                if batch_idx >= self._current_activations.shape[0]:
                    continue

                activation = self._current_activations[batch_idx]
                if self.activation_proj is not None:
                    activation = self.activation_proj(activation)
                activation = activation.to(hidden_states.dtype)

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
        # First check if it's in our modules (including base_model)
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]

        # Check if base_model exists as a module
        if "_modules" in self.__dict__ and "base_model" in self._modules:
            # Delegate to base model
            return getattr(self._modules["base_model"], name)

        # Fallback error
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
        # Check if past_key_values is None OR empty (newer transformers uses DynamicCache with len=0)
        is_first_forward = past_key_values is None or (
            hasattr(past_key_values, '__len__') and len(past_key_values) == 0
        )
        should_inject = activation_vectors is not None and is_first_forward

        if should_inject:
            if input_ids is None:
                raise ValueError("input_ids must be provided for activation injection.")
            injection_positions = self._find_injection_positions(input_ids)
            self._set_injection_state(activation_vectors, injection_positions, input_ids.device)

        try:
            if inputs_embeds is not None:
                result = self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs
                )
            else:
                result = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs
                )
        finally:
            if should_inject:
                self._clear_injection_state()

        return result

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation - delegate to base model and preserve activation_vectors."""
        # Extract activation_vectors before delegating (it's not a standard arg)
        activation_vectors = kwargs.pop("activation_vectors", None)

        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            model_inputs = self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        else:
            # Minimal default implementation
            model_inputs = {"input_ids": kwargs.get("input_ids", args[0] if args else None)}

        # Add activation_vectors back to the inputs dict so it gets passed to forward()
        if activation_vectors is not None:
            model_inputs["activation_vectors"] = activation_vectors

        return model_inputs

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

    def _set_injection_state(
        self,
        activation_vectors: torch.Tensor,
        positions: List[Optional[torch.Tensor]],
        device: torch.device,
    ) -> None:
        if activation_vectors.dim() == 1:
            activation_vectors = activation_vectors.unsqueeze(0)
        self._current_activations = activation_vectors.to(device)

        processed_positions: List[Optional[torch.Tensor]] = []
        for pos in positions:
            if pos is None:
                processed_positions.append(None)
            else:
                processed_positions.append(pos.to(device))
        self._injection_positions = processed_positions

    def _clear_injection_state(self) -> None:
        self._current_activations = None
        self._injection_positions = None
