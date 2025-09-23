"""Custom actor worker for NLA that handles activation injection."""

import torch
from typing import Dict, Optional, Any
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from ..models.nla_wrapper import NLAModelWrapper, InjectionConfig


class NLAActorWorker:
    """
    Custom actor worker that wraps the base model with NLAModelWrapper
    for activation injection during generation.

    This worker is designed to be used with GRPO/PPO trainers where
    the actor needs to generate text conditioned on activation vectors.
    """

    def __init__(self, base_worker_cls: type):
        """
        Initialize NLA actor worker by wrapping a base worker class.

        Args:
            base_worker_cls: The base worker class to extend
        """
        self.base_worker_cls = base_worker_cls
        self._create_wrapped_class()

    def _create_wrapped_class(self):
        """Create a wrapped worker class with NLA functionality."""

        class NLAActorWorkerWrapped(self.base_worker_cls):
            """Inner class that extends the base worker with NLA capabilities."""

            def init_model(self):
                """Initialize model with NLA wrapper."""
                # First initialize the base model
                super().init_model()

                # Extract model and config
                base_model = self.model
                config = getattr(self, 'config', None)

                # Configure injection settings
                injection_config = InjectionConfig(
                    mode=config.model.injection.get("mode", "replace") if config else "replace",
                    layer_indices=config.model.injection.get("layer_indices", [0]) if config else [0],
                    projection_dim=config.model.injection.get("projection_dim", None) if config else None,
                    injection_token_id=config.model.injection.get("injection_token_id", -1) if config else -1,
                )

                # Wrap model with NLA wrapper
                self.model = NLAModelWrapper(
                    base_model=base_model,
                    injection_config=injection_config,
                    hidden_dim=base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else None,
                    activation_dim=config.model.get("activation_dim", 768) if config else 768,
                )

                print(f"Wrapped actor model with NLAModelWrapper")
                print(f"Injection config: {injection_config}")

            def generate_sequences(self, data: DataProto) -> DataProto:
                """
                Generate sequences with activation injection.

                This method extends the base generation to include activation vectors
                in the forward pass of the model.
                """
                # Extract activation vectors from DataProto metadata
                activation_vectors = None
                if hasattr(data, 'metadata') and data.metadata:
                    activation_vectors = data.metadata.get('activation_vectors')

                # If no activation vectors in metadata, check batch data
                if activation_vectors is None and hasattr(data, 'batch'):
                    activation_vectors = data.batch.get('activation_vectors')

                # Store activation vectors in a place accessible during generation
                if activation_vectors is not None:
                    # Add to model kwargs that will be passed to forward
                    if not hasattr(data, 'meta_info'):
                        data.meta_info = {}
                    data.meta_info['activation_vectors'] = activation_vectors
                    print(f"Found activation vectors with shape: {activation_vectors.shape}")

                # Call base generation
                result = super().generate_sequences(data)

                return result

            def compute_log_prob(self, data: DataProto) -> DataProto:
                """
                Compute log probabilities with activation injection.

                Ensures activation vectors are passed during log prob computation.
                """
                # Extract and pass activation vectors similar to generation
                activation_vectors = None
                if hasattr(data, 'metadata') and data.metadata:
                    activation_vectors = data.metadata.get('activation_vectors')

                if activation_vectors is None and hasattr(data, 'batch'):
                    activation_vectors = data.batch.get('activation_vectors')

                if activation_vectors is not None:
                    if not hasattr(data, 'meta_info'):
                        data.meta_info = {}
                    data.meta_info['activation_vectors'] = activation_vectors

                return super().compute_log_prob(data)

        self.wrapped_class = NLAActorWorkerWrapped

    def get_worker_class(self):
        """Return the wrapped worker class."""
        return self.wrapped_class


def create_nla_actor_worker(base_worker_cls: type, config: Optional[Dict] = None):
    """
    Factory function to create an NLA actor worker from a base worker class.

    Args:
        base_worker_cls: The base worker class to wrap
        config: Optional configuration for NLA settings

    Returns:
        The wrapped worker class with NLA capabilities
    """
    wrapper = NLAActorWorker(base_worker_cls)
    return wrapper.get_worker_class()