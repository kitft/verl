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

        # Patch engine's async_generate ONCE at initialization (not per-call)
        # This is faster than patching per batch
        self._patch_engine_async_generate()

    def _patch_engine_async_generate(self):
        """
        Patch SGLang engine's async_generate method ONCE at initialization.

        This intercepts calls and removes input_ids when input_embeds is provided,
        forcing SGLang to use embeddings instead of token IDs.
        """
        if self._engine is None:
            print("NLA SGLang: Engine not initialized yet, patch will be applied later")
            return

        from functools import wraps

        original_async_generate = self._engine.async_generate

        @wraps(original_async_generate)
        async def nla_async_generate(*args, **gen_kwargs):
            """Patched version that nullifies input_ids when input_embeds is provided."""
            print("=" * 80)
            print("NLA PATCH: nla_async_generate called!")
            print(f"  args: {len(args)}")
            print(f"  gen_kwargs keys: {list(gen_kwargs.keys())}")
            print(f"  has input_embeds: {'input_embeds' in gen_kwargs}")
            if 'input_embeds' in gen_kwargs:
                print(f"  input_embeds is not None: {gen_kwargs['input_embeds'] is not None}")
            print(f"  has input_ids: {'input_ids' in gen_kwargs}")
            if 'input_ids' in gen_kwargs:
                print(f"  input_ids type before: {type(gen_kwargs['input_ids'])}")

            if 'input_embeds' in gen_kwargs and gen_kwargs['input_embeds'] is not None:
                print(f"NLA PATCH: Nullifying input_ids! (was {type(gen_kwargs.get('input_ids'))})")
                gen_kwargs['input_ids'] = None
                print(f"NLA PATCH: After nullification, input_ids = {gen_kwargs['input_ids']}")
            else:
                print("NLA PATCH: NOT nullifying input_ids (no input_embeds or embeds is None)")

            # Final verification before calling original
            print(f"NLA PATCH: Calling original async_generate with:")
            print(f"  prompt = {gen_kwargs.get('prompt')} (is None: {gen_kwargs.get('prompt') is None})")
            print(f"  input_ids = {type(gen_kwargs.get('input_ids'))} (is None: {gen_kwargs.get('input_ids') is None})")
            print(f"  input_embeds = {type(gen_kwargs.get('input_embeds'))}")
            if gen_kwargs.get('input_embeds'):
                print(f"    input_embeds length: {len(gen_kwargs['input_embeds'])}")
            print("=" * 80)

            result = await original_async_generate(*args, **gen_kwargs)
            print("NLA PATCH: original async_generate returned successfully")
            return result

        self._engine.async_generate = nla_async_generate
        print("NLA SGLang: Patched engine.async_generate (permanent, zero overhead per call)")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences with optional custom input embeddings.

        The NLAActorRolloutRefWorker prepares the input_embeds with injected
        activations and passes them in the batch. We just need to ensure
        they're on CPU for network transmission.
        """
        print("=" * 80)
        print("NLA SGLang: generate_sequences() ENTRY POINT")
        print(f"NLA SGLang: Batch keys: {list(prompts.batch.keys())}")
        print("=" * 80)

        # Check if input_embeds are already prepared by the actor worker
        if 'input_embeds' in prompts.non_tensor_batch:
            input_embeds = prompts.non_tensor_batch['input_embeds']
            print(f"NLA SGLang: Received pre-computed input_embeds with first item shape: {input_embeds[0].shape}", f"second item shape: {input_embeds[1].shape if len(input_embeds) > 1 else 'None'}")

            # # Ensure embeddings are on CPU for network transmission
            # if hasattr(input_embeds, 'is_cuda') and input_embeds.is_cuda:
            #     print("NLA SGLang: Converting input_embeds to CPU for network transmission")
            #     prompts.batch['input_embeds'] = input_embeds.cpu()

            # CRITICAL: SGLang's priority is text > input_ids > input_embeds
            # (See sglang/srt/managers/io_struct.py:170-196)
            # If input_ids exists, SGLang will DISCARD input_embeds and use input_ids instead!
            # We MUST set input_ids=None to force SGLang to use input_embeds.
            #
            # However, parent's _batch_level_generate_sequences needs input_ids for:
            # 1. batch_size = idx.size(0) at line 850
            # 2. Creating raw_prompt_ids at lines 854-858
            #
            # Solution: Pre-create raw_prompt_ids, then nullify input_ids
            if 'input_ids' in prompts.batch and prompts.batch['input_ids'] is not None:
                # Pre-create raw_prompt_ids before nullifying input_ids
                # (Replicates logic from sglang_rollout.py:854-858)
                input_ids = prompts.batch['input_ids']
                batch_size = input_ids.size(0)

                # Import _pre_process_inputs from parent module
                from verl.workers.rollout.sglang_rollout.sglang_rollout import _pre_process_inputs
                import numpy as np

                # Get pad_token_id from parent's config
                pad_token_id = self.pad_token_id if hasattr(self, 'pad_token_id') else 0

                prompts.non_tensor_batch["raw_prompt_ids"] = np.array(
                    [_pre_process_inputs(pad_token_id, input_ids[i]).tolist() for i in range(batch_size)],
                    dtype=object,
                )
                print(f"NLA SGLang: Pre-created raw_prompt_ids from input_ids (batch_size={batch_size})")

                # DON'T nullify input_ids - keep it so parent can use it for batch_size
                # SGLang will use input_embeds anyway because we pass it explicitly
                #print("NLA SGLang: Kept input_ids for batch_size calculation")

            #print("NLA SGLang: input_embeds will be used by SGLang")

        # # Prepare embeddings from activation vectors attached to the batch
        # elif 'activation_vectors' in prompts.batch:
        #     # they are directly passed the input embeds, it seems
        #     activation_vectors = prompts.batch['activation_vectors']
        #     injection_positions = None
        #     if hasattr(prompts, 'meta_info') and prompts.meta_info:
        #         injection_positions = prompts.meta_info.get('injection_positions')

        #     if injection_positions is None:
        #         raise ValueError("Injection positions missing while activation vectors are provided")

        #     print("NLA SGLang: Preparing embeddings from batch activation vectors")
        #     print(f"NLA SGLang: Activation vectors shape: {activation_vectors.shape}")
        #     print(f"NLA SGLang: Injection positions: {injection_positions}")

        #     input_ids = prompts.batch["input_ids"]
        #     input_embeds = self._prepare_input_embeddings(
        #         input_ids, activation_vectors, injection_positions
        #     )

        #     if input_embeds.is_cuda:
        #         print("NLA SGLang: Converting input_embeds to CPU for network transmission")
        #         input_embeds = input_embeds.cpu()
        #         print(f"NLA SGLang: input_embeds is now on CPU")

        #     prompts.batch['input_embeds'] = input_embeds
        #     print(f"NLA SGLang: Prepared input_embeds with shape: {input_embeds.shape}")

        #     # Pre-create raw_prompt_ids before nullifying input_ids (same logic as above)
        #     input_ids = prompts.batch["input_ids"]
        #     batch_size = input_ids.size(0)

        #     from verl.workers.rollout.sglang_rollout.sglang_rollout import _pre_process_inputs
        #     import numpy as np

        #     pad_token_id = self.pad_token_id if hasattr(self, 'pad_token_id') else 0

        #     prompts.non_tensor_batch["raw_prompt_ids"] = np.array(
        #         [_pre_process_inputs(pad_token_id, input_ids[i]).tolist() for i in range(batch_size)],
        #         dtype=object,
        #     )
        #     print(f"NLA SGLang: Pre-created raw_prompt_ids from input_ids (batch_size={batch_size})")

        #     # DON'T nullify input_ids - keep it so parent can use it for batch_size
        #     # SGLang will use input_embeds anyway because we pass it explicitly
        #     print("NLA SGLang: Kept input_ids for batch_size calculation")

        elif hasattr(prompts, 'meta_info') and prompts.meta_info and prompts.meta_info.get('activation_vectors') is not None:
            raise ValueError(
                "Activation vectors present in meta_info; expected them under batch['activation_vectors']"
            )

        else:
            # FAIL FAST: Neither input_embeds nor activation_vectors present
            raise ValueError(
                "NLA SGLang: Neither 'input_embeds' nor 'activation_vectors' found in batch! "
                "NLA rollout requires one of these for activation injection. "
                "Available batch keys: " + str(list(prompts.batch.keys()))
            )

        # print(prompts.batch.keys())
        # print(prompts.non_tensor_batch.keys())
        # print(prompts.meta_info.keys())

        # Note: We no longer nullify input_ids, so no dummy creation needed
        # input_embeds will be used by SGLang because we provide raw_prompt_ids and input_embeds
        #print(f"NLA SGLang: input_ids kept in batch, input_embeds will take priority in SGLang")

        # Call parent's generate_sequences
        # SGLang will now use input_embeds if present in the batch
        print("NLA SGLang: Calling parent generate_sequences...")
        result = super().generate_sequences(prompts)
        print("NLA SGLang: Parent generate_sequences returned!")
        return result

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
                    raise AttributeError("Could not find embedding layer in SGLang model")
            else:
                raise AttributeError("SGLang engine does not expose model")

        except AttributeError as e:
            print(f"WARNING: Could not access SGLang embedding layer: {e}")
            print("Using a fallback embedding layer - results may be suboptimal")
            print(f"Could not access SGLang embedding layer: {e}")
            print(f"FIX THIS LATER")
            vocab_size = self.model_config.hf_config.vocab_size
            embed_layer = torch.nn.Embedding(vocab_size, self.hidden_dim)
            raise e

        # Compute base embeddings for all tokens
        with torch.no_grad():
            input_embeds = embed_layer(input_ids)  # (batch_size, seq_len, hidden_dim)

        # Project activation vectors if needed
        if activation_vectors.shape[-1] != self.hidden_dim:
            raise ValueError(f"Activation vectors dimension {activation_vectors.shape[-1]} does not match hidden dimension {self.hidden_dim}")
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
