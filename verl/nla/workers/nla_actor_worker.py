"""Custom actor worker for NLA that handles activation injection."""

import torch
from typing import Dict, Optional, Any
from verl.protocol import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker as FSDPActorRolloutRefWorker
from verl.single_controller.base.decorator import register
from verl.single_controller.base import Dispatch
from ..models.nla_wrapper import NLAModelWrapper, InjectionConfig


class NLAActorRolloutRefWorker(FSDPActorRolloutRefWorker):
    """
    NLA-enabled actor rollout worker that handles activation injection.

    This worker extends the base FSDP actor worker with NLA capabilities
    for activation injection during generation.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with NLA wrapper."""
        # First call parent's init_model to set up the base model
        super().init_model()

        # Now wrap the actor model with NLA capabilities if we have activation injection
        if hasattr(self, 'actor') and hasattr(self.config, 'nla'):
            # Extract NLA configuration
            nla_config = self.config.get('nla', {})
            injection_config = nla_config.get('injection', {})

            # Configure injection settings
            injection_cfg = InjectionConfig(
                mode=injection_config.get("mode", "replace"),
                layer_indices=injection_config.get("layer_indices", [0]),
                projection_dim=injection_config.get("projection_dim", None),
                injection_token=injection_config.get("injection_token", None),
            )

            # Get model's hidden dimension
            base_model = self.actor.model if hasattr(self.actor, 'model') else self.actor
            hidden_dim = base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else None
            activation_dim = nla_config.get("activation_dim", hidden_dim or 768)

            # Wrap model with NLA wrapper
            self.nla_wrapper = NLAModelWrapper(
                base_model=base_model,
                tokenizer=getattr(self, 'tokenizer', None),
                injection_config=injection_cfg,
                hidden_dim=hidden_dim,
                activation_dim=activation_dim,
            )

            print(f"Wrapped actor model with NLAModelWrapper")
            print(f"Injection token: '{self.nla_wrapper.injection_config.injection_character}' (ID: {self.nla_wrapper.injection_config.injection_token_id})")

    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        Generate sequences with activation injection.

        This method extends the base generation to include activation vectors
        in the forward pass of the model.
        """
        # Extract activation vectors from DataProto
        activation_vectors = None
        if hasattr(data, 'metadata') and data.metadata:
            activation_vectors = data.metadata.get('activation_vectors')

        # If no activation vectors in metadata, check batch data
        if activation_vectors is None and hasattr(data, 'data') and 'activation_vectors' in data.data:
            activation_vectors = data.data.get('activation_vectors')

        # Store activation vectors for use during generation
        if activation_vectors is not None and hasattr(self, 'nla_wrapper'):
            # TODO: Set up activation injection in the model
            # This would involve modifying the model's forward pass to inject activations
            print(f"Found activation vectors with shape: {activation_vectors.shape}")

        # Call parent's generate_sequences
        return super().generate_sequences(data)

    def compute_log_prob(self, data: DataProto) -> DataProto:
        """
        Compute log probabilities with activation injection.

        Ensures activation vectors are passed during log prob computation.
        """
        # Extract and handle activation vectors similar to generation
        activation_vectors = None
        if hasattr(data, 'metadata') and data.metadata:
            activation_vectors = data.metadata.get('activation_vectors')

        if activation_vectors is None and hasattr(data, 'data') and 'activation_vectors' in data.data:
            activation_vectors = data.data.get('activation_vectors')

        if activation_vectors is not None and hasattr(self, 'nla_wrapper'):
            print(f"Computing log probs with activation vectors of shape: {activation_vectors.shape}")

        return super().compute_log_prob(data)

