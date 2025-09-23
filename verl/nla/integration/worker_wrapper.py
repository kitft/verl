"""Worker wrapper to add NLA capability to existing verl workers."""

import torch
import torch.nn as nn
from typing import Optional, Any, Dict
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from ..models import NLAModelWrapper, InjectionConfig
from .dataproto_adapter import NLADataProtoAdapter


class NLAWorkerWrapper(Worker):
    """Wrapper that adds NLA capability to existing verl workers."""

    def __init__(
        self,
        base_worker_class: type,
        config: DictConfig,
        role: str,
        nla_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Initialize the base worker
        self.base_worker = base_worker_class(config=config, role=role)

        # NLA configuration
        self.nla_config = nla_config or {}
        self.injection_enabled = self.nla_config.get("enabled", False)

        if self.injection_enabled:
            # Initialize NLA components
            self.injection_config = InjectionConfig(
                mode=self.nla_config.get("mode", "replace"),
                layer_indices=self.nla_config.get("layer_indices", [0]),
                projection_dim=self.nla_config.get("projection_dim", None),
                injection_token_id=self.nla_config.get("injection_token_id", -1),
            )

            self.adapter = NLADataProtoAdapter(
                injection_token=self.nla_config.get("injection_token", "<INJECT>")
            )

            self._wrap_model = False
            self._nla_wrapper = None

    def _wrap_model_with_nla(self):
        """Wrap the base model with NLA wrapper."""
        if not self.injection_enabled or self._wrap_model:
            return

        # Get the model from base worker
        if hasattr(self.base_worker, "actor_model"):
            base_model = self.base_worker.actor_model
            model_attr = "actor_model"
        elif hasattr(self.base_worker, "model"):
            base_model = self.base_worker.model
            model_attr = "model"
        else:
            return

        # Wrap with NLA
        self._nla_wrapper = NLAModelWrapper(
            base_model=base_model,
            injection_config=self.injection_config,
            hidden_dim=self.nla_config.get("hidden_dim", None),
            activation_dim=self.nla_config.get("activation_dim", None),
        )

        # Replace the model in base worker
        setattr(self.base_worker, model_attr, self._nla_wrapper)
        self._wrap_model = True

    def _extract_activation_vectors(self, prompts: DataProto) -> Optional[torch.Tensor]:
        """Extract activation vectors from prompts DataProto."""
        if not self.injection_enabled:
            return None
        return self.adapter.extract_activation_vectors_from_dataproto(prompts)

    def _inject_activation_vectors(
        self, prompts: DataProto, activation_vectors: torch.Tensor
    ) -> DataProto:
        """Inject activation vectors into prompts DataProto."""
        if not self.injection_enabled:
            return prompts

        # Find injection positions if not already provided
        injection_positions = self.adapter.extract_injection_positions_from_dataproto(prompts)
        if injection_positions is None:
            injection_positions = self.adapter.find_injection_positions_in_prompts(
                prompts, self.base_worker.tokenizer
            )

        return self.adapter.add_activation_vectors_to_dataproto(
            prompts, activation_vectors, injection_positions
        )

    def init_model(self):
        """Initialize model and optionally wrap with NLA."""
        # Initialize base model
        self.base_worker.init_model()

        # Wrap with NLA if enabled
        if self.injection_enabled:
            self._wrap_model_with_nla()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences with optional activation injection."""
        # Extract activation vectors if present
        activation_vectors = self._extract_activation_vectors(prompts)

        if activation_vectors is not None and self._nla_wrapper is not None:
            # Store activation vectors in the wrapper for use during generation
            self._nla_wrapper._current_activations = activation_vectors
            self._nla_wrapper._injection_positions = self.adapter.extract_injection_positions_from_dataproto(prompts)

        # Call base worker's generate method
        return self.base_worker.generate_sequences(prompts)

    def compute_ref_log_prob(self, prompts: DataProto) -> DataProto:
        """Compute reference log probabilities with optional activation injection."""
        # Extract and set activation vectors if present
        activation_vectors = self._extract_activation_vectors(prompts)

        if activation_vectors is not None and self._nla_wrapper is not None:
            self._nla_wrapper._current_activations = activation_vectors
            self._nla_wrapper._injection_positions = self.adapter.extract_injection_positions_from_dataproto(prompts)

        return self.base_worker.compute_ref_log_prob(prompts)

    def update_actor(self, data: DataProto) -> Dict[str, Any]:
        """Update actor with optional activation injection."""
        # Extract and set activation vectors if present
        activation_vectors = self._extract_activation_vectors(data)

        if activation_vectors is not None and self._nla_wrapper is not None:
            self._nla_wrapper._current_activations = activation_vectors
            self._nla_wrapper._injection_positions = self.adapter.extract_injection_positions_from_dataproto(data)

        return self.base_worker.update_actor(data)

    def __getattr__(self, name):
        """Forward any other method calls to the base worker."""
        return getattr(self.base_worker, name)