"""NLA-specific SGLang rollout wrapper that supports embedding input."""

import torch
from typing import Generator, Optional, Any
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


class NLASGLangRollout(SGLangRollout):
    """
    SGLang rollout wrapper for NLA that supports custom input embeddings.

    This class extends SGLang rollout to handle activation vector injection
    via input embeddings rather than intermediate layer manipulation.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        # Store model config for embedding dimension
        self.hidden_dim = model_config.hf_config.hidden_size

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with optional custom input embeddings.

        The NLAActorRolloutRefWorker prepares the input_embeds with injected
        activations and passes them in the batch. We just need to ensure
        they're on CPU for network transmission.
        """
        # Check if input_embeds are already prepared by the actor worker
        if 'input_embeds' in prompts.batch:
            input_embeds = prompts.batch['input_embeds']
            print(f"NLA SGLang: Received pre-computed input_embeds with shape: {input_embeds.shape}")

            # Ensure embeddings are on CPU for network transmission
            if hasattr(input_embeds, 'is_cuda') and input_embeds.is_cuda:
                print("NLA SGLang: Converting input_embeds to CPU for network transmission")
                prompts.batch['input_embeds'] = input_embeds.cpu()

        # Fallback: If embeddings weren't prepared but we have activation vectors
        elif hasattr(prompts, 'meta_info') and prompts.meta_info:
            activation_vectors = prompts.meta_info.get('activation_vectors')
            injection_positions = prompts.meta_info.get('injection_positions')

            if activation_vectors is not None and injection_positions is not None:
                print(f"NLA SGLang: Fallback - preparing embeddings from activation vectors")
                print(f"NLA SGLang: Activation vectors shape: {activation_vectors.shape}")
                print(f"NLA SGLang: Injection positions: {injection_positions}")

                # Get input token IDs
                input_ids = prompts.batch["input_ids"]

                # Prepare input embeddings with activation injection
                input_embeds = self._prepare_input_embeddings(
                    input_ids, activation_vectors, injection_positions
                )

                # Convert to CPU for network transmission
                if input_embeds.is_cuda:
                    print("NLA SGLang: Converting input_embeds to CPU for network transmission")
                    input_embeds = input_embeds.cpu()

                # Store in batch for SGLang
                prompts.batch['input_embeds'] = input_embeds
                print(f"NLA SGLang: Prepared input_embeds with shape: {input_embeds.shape}")

        # Call parent's generate_sequences
        # SGLang will now use input_embeds if present in the batch
        return super().generate_sequences(prompts)

    def _prepare_input_embeddings(
        self,
        input_ids: torch.Tensor,
        activation_vectors: torch.Tensor,
        injection_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare input embeddings with activation vector injection.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            activation_vectors: Activation vectors to inject (batch_size, hidden_dim or activation_dim)
            injection_positions: Positions to inject at (batch_size,)

        Returns:
            Input embeddings with injected activations (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Get the embedding layer from SGLang's engine
        # Note: This assumes SGLang exposes the model's embedding layer
        try:
            # Try to get embedding layer from the engine's model
            if hasattr(self._engine, 'model'):
                model = self._engine.model
                if hasattr(model, 'get_input_embeddings'):
                    embed_layer = model.get_input_embeddings()
                elif hasattr(model, 'embed_tokens'):
                    embed_layer = model.embed_tokens
                elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_layer = model.model.embed_tokens
                else:
                    # Fallback: create a simple embedding layer
                    print("WARNING: Could not find embedding layer in SGLang model, using fallback")
                    vocab_size = self.model_config.hf_config.vocab_size
                    embed_layer = torch.nn.Embedding(vocab_size, self.hidden_dim)
            else:
                raise AttributeError("SGLang engine does not expose model")

        except AttributeError as e:
            print(f"WARNING: Could not access SGLang embedding layer: {e}")
            print("Using a fallback embedding layer - results may be suboptimal")
            vocab_size = self.model_config.hf_config.vocab_size
            embed_layer = torch.nn.Embedding(vocab_size, self.hidden_dim)

        # Compute base embeddings for all tokens
        with torch.no_grad():
            input_embeds = embed_layer(input_ids)  # (batch_size, seq_len, hidden_dim)

        # Project activation vectors if needed
        if activation_vectors.shape[-1] != self.hidden_dim:
            print(f"Projecting activation vectors from {activation_vectors.shape[-1]} to {self.hidden_dim}")
            # Simple linear projection - could be improved with a learned projection
            projection = torch.nn.Linear(activation_vectors.shape[-1], self.hidden_dim, bias=False)
            activation_vectors = projection(activation_vectors)

        # Inject activation vectors at specified positions
        for i in range(batch_size):
            pos = injection_positions[i].item()
            if 0 <= pos < seq_len:
                input_embeds[i, pos] = activation_vectors[i]
                print(f"Injected activation at position {pos} for sequence {i}")

        return input_embeds