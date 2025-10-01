"""Custom actor worker for NLA that handles activation injection."""

import torch
from typing import Dict, Optional, Any
from verl.protocol import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker as FSDPActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch, make_nd_compute_dataproto_dispatch_fn
from ..models.nla_wrapper import NLAModelWrapper, InjectionConfig
from ..utils.injection_manager import InjectionTokenManager


class NLAActorRolloutRefWorker(FSDPActorRolloutRefWorker):
    """
    NLA-enabled actor rollout worker that handles activation injection.

    This worker extends the base FSDP actor worker with NLA capabilities
    for activation injection during generation.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with NLA wrapper."""
        print("=" * 80)
        print("NLA ACTOR WORKER: init_model() called")
        print("=" * 80)

        # First call parent's init_model to set up the base model
        super().init_model()

        print(f"NLA ACTOR: After super().init_model(), has rollout: {hasattr(self, 'rollout')}")
        print(f"NLA ACTOR: Config has nla: {hasattr(self.config, 'nla')}")
        if hasattr(self.config, 'nla'):
            print(f"NLA ACTOR: Config.nla = {self.config.nla}")

        # Store reference to the embedding layer for later use
        self._store_embedding_layer()

        # Now wrap the rollout model with NLA capabilities (if not using SGLang)
        if hasattr(self, 'rollout') and hasattr(self.config, 'nla'):
            print("NLA ACTOR: Setting up NLA injection configuration")
            # Extract NLA configuration
            nla_config = self.config.get('nla', {})
            injection_config = nla_config.get('injection', {})
            print(f"NLA ACTOR: injection_config = {injection_config}")

            # Configure injection settings
            self.injection_cfg = InjectionConfig(
                mode=injection_config.get("mode", "replace"),
                layer_indices=injection_config.get("layer_indices", [0]),
                projection_dim=injection_config.get("projection_dim", None),
                injection_token=injection_config.get("injection_token", None),
            )


            # For SGLang rollout (nla_sglang), we use embedding-based injection
            # The rollout itself doesn't need wrapping - we prepare input_embeds before calling it
            rollout_name = self.config.rollout.name if hasattr(self.config, 'rollout') else 'unknown'
            print(f"NLA ACTOR: Rollout type: {rollout_name}")

            if rollout_name == 'nla_sglang':
                print("NLA ACTOR: Using SGLang with embedding-based injection (no rollout wrapping needed)")
                # Store injection config for use in generate_sequences
                self.nla_injection_mode = 'embedding'
            elif hasattr(self.rollout, 'module'):
                # For other rollouts that have a module attribute (e.g., HF rollout)
                print("NLA ACTOR: Wrapping rollout module with NLA capabilities")
                base_model = self.rollout.module
                hidden_dim = base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else None
                activation_dim = nla_config.get("activation_dim", hidden_dim or 768)

                # Get tokenizer from model config
                tokenizer = self.model_config.tokenizer if hasattr(self, 'model_config') else None

                # Wrap rollout's module with NLA wrapper
                self.nla_wrapper = NLAModelWrapper(
                    base_model=base_model,
                    tokenizer=tokenizer,
                    injection_config=self.injection_cfg,
                    hidden_dim=hidden_dim,
                    activation_dim=activation_dim,
                )

                # Replace the rollout's module with the wrapped version
                self.rollout.module = self.nla_wrapper
                self.nla_injection_mode = 'wrapper'

                print(f"Wrapped rollout model with NLAModelWrapper")
                print(f"Injection token: '{self.nla_wrapper.injection_config.injection_character}' (ID: {self.nla_wrapper.injection_config.injection_token_id})")
            else:
                print(f"WARNING: Rollout type '{rollout_name}' doesn't have a module attribute and isn't nla_sglang")
                self.nla_injection_mode = 'unknown'

            
            tokenizer = self.model_config.tokenizer if hasattr(self, 'model_config') else None
            if tokenizer is not None:
                # Use InjectionTokenManager for consistent token selection
                injection_manager = InjectionTokenManager(tokenizer, self.injection_cfg.injection_token)
                # Update injection config with auto-selected token info
                self.injection_cfg.injection_token_id = injection_manager.token_id
                self.injection_cfg.injection_character = injection_manager.character
                object.__setattr__(self, "injection_manager", injection_manager)
            elif self.injection_cfg.injection_token_id is not None and self.injection_cfg.injection_token_id >= 0:
                # Token ID was manually specified - use it directly
                object.__setattr__(self, "injection_manager", None)
            else:
                raise ValueError(
                    "Either provide a tokenizer for auto-selection or specify injection_token_id in config. "
                    "The tokenizer ensures consistency with the dataset's injection token."
            )

        # Also wrap actor model if it exists (for compute_log_prob)
        if hasattr(self, 'actor') and hasattr(self.config, 'nla'):
            print("NLA ACTOR: Setting up actor model for NLA (for log prob computation)")
            # For now, we don't wrap the actor model - we'll handle activation injection
            # in compute_log_prob by passing activation_vectors through the data
            # This is simpler and doesn't require wrapping the FSDP model
            print("NLA ACTOR: Actor model wrapping skipped (using data-based injection for log probs)")

    def _store_embedding_layer(self):
        """Store reference to the model's embedding layer for input embedding computation."""
        self.embed_layer = None

        # Try to get embedding layer from actor model
        if hasattr(self, 'actor_module_fsdp'):
            model = self.actor_module_fsdp

            # Try different access patterns for different model architectures
            if hasattr(model, 'get_input_embeddings'):
                self.embed_layer = model.get_input_embeddings()
            elif hasattr(model, 'embed_tokens'):
                self.embed_layer = model.embed_tokens
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                self.embed_layer = model.model.embed_tokens
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                self.embed_layer = model.transformer.wte  # GPT-style
            elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                self.embed_layer = model.bert.embeddings.word_embeddings  # BERT-style

            if self.embed_layer is not None:
                print(f"Successfully stored embedding layer from actor model")
            else:
                print("WARNING: Could not find embedding layer in actor model")

    def compute_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute input embeddings using the actor model's embedding layer.

        Args:
            input_ids: Token IDs (batch_size, seq_len)

        Returns:
            Input embeddings (batch_size, seq_len, hidden_dim)
        """
        if self.embed_layer is None:
            raise RuntimeError("Embedding layer not available. Cannot compute input embeddings.")

        with torch.no_grad():
            # Compute embeddings
            input_embeds = self.embed_layer(input_ids)

        return input_embeds

    def _extract_activation_vectors(self, data: DataProto):
        activation_vectors = None
        if data.batch is not None:
            batch_keys = getattr(data.batch, 'keys', lambda: [])()
            if 'activation_vectors' in batch_keys:
                activation_vectors = data.batch['activation_vectors']

        if activation_vectors is None and data.meta_info and data.meta_info.get('activation_vectors') is not None:
            raise ValueError("Activation vectors must be provided via batch['activation_vectors']")

        return activation_vectors

    def _prepare_input_embeddings(self, data: DataProto, input_ids: torch.Tensor, activation_vectors: torch.Tensor) -> torch.Tensor:
        if self.embed_layer is None:
            raise RuntimeError("Embedding layer not available. Cannot compute input embeddings.")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed._tensor import DTensor

        target_device = self.embed_layer.weight.device if hasattr(self.embed_layer, "weight") else None
        if target_device is not None:
            input_ids = input_ids.to(target_device)

        with torch.no_grad():
            # Check if embedding weight is sharded (DTensor from FSDP)
            if isinstance(self.embed_layer.weight, DTensor):
                # All-gather just the embedding weights for full embeddings
                full_weight = self.embed_layer.weight.full_tensor()
                # Manually do embedding lookup with gathered weights
                input_embeds = torch.nn.functional.embedding(
                    input_ids,
                    full_weight,
                    padding_idx=getattr(self.embed_layer, 'padding_idx', None)
                )
            else:
                # Regular embedding layer (not sharded)
                input_embeds = self.embed_layer(input_ids)

        activation_vectors = activation_vectors.to(input_embeds.device)
        batch_size = activation_vectors.shape[0]
        injection_positions = torch.ones(batch_size, dtype=torch.long, device=activation_vectors.device)

        # Check for injection token in the config
        injection_token_id = None
        if hasattr(self, 'injection_cfg') and self.injection_cfg.injection_token_id is not None:
            injection_token_id = self.injection_cfg.injection_token_id
        elif hasattr(self, 'nla_wrapper') and self.nla_wrapper.injection_config.injection_token_id is not None:
            injection_token_id = self.nla_wrapper.injection_config.injection_token_id

        if injection_token_id is not None:
            injection_mask = (input_ids == injection_token_id)
            if injection_mask.any():
                injection_positions = injection_mask.float().argmax(dim=1)
                print(f"NLA Actor: Found injection tokens at positions: {injection_positions}")

        hidden_dim = input_embeds.shape[-1]

        if activation_vectors.shape[-1] != hidden_dim:
            print(f"NLA Actor: Projecting activation vectors from {activation_vectors.shape[-1]} to {hidden_dim}")
            projection = torch.nn.Linear(activation_vectors.shape[-1], hidden_dim, bias=False).to(activation_vectors.device)
            activation_vectors = projection(activation_vectors)

        for i in range(batch_size):
            pos = injection_positions[i].item()
            if 0 <= pos < input_embeds.shape[1]:
                input_embeds[i, pos] = activation_vectors[i]
                print(f"NLA Actor: Injected activation at position {pos} for sequence {i} at token # input_ids[{i}, {pos}] = {input_ids[i, pos]}, counting left-padding.")

        attention_mask = data.batch['attention_mask']

        if attention_mask is None:
            raise ValueError("Attention mask is required to remove padding vectors.")

        # attention_mask: [batch, seq]
        # input_embeds: [batch, seq, hidden]
        input_embeds_list = []
        for i in range(input_embeds.shape[0]):
            mask = attention_mask[i].bool()
            input_embeds_list.append(input_embeds[i][mask].cpu())

        input_embeds = input_embeds_list

        return input_embeds

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        Generate sequences with activation injection.

        For SGLang: Prepares activation vectors and injection positions
        to be used as custom input embeddings.
        """
        print("=" * 80)
        print("NLA ACTOR: generate_sequences() called")
        print("=" * 80)
        print(f"NLA ACTOR: data.batch type: {type(data.batch)}")
        print(f"NLA ACTOR: data.batch keys: {list(data.batch.keys()) if data.batch is not None else 'None'}")
        print(f"NLA ACTOR: data.meta_info keys: {list(data.meta_info.keys()) if data.meta_info is not None else 'None'}")

        activation_vectors = self._extract_activation_vectors(data)
        print(f"NLA ACTOR: Extracted activation_vectors: {activation_vectors.shape if activation_vectors is not None else None}")

        # FAIL FAST: Ensure activation vectors are present
        if activation_vectors is None:
            raise ValueError(
                "NLA Actor Worker: activation_vectors missing from batch! "
                "NLA training requires activation vectors for all samples. "
                "Check that the dataset includes 'activation_vectors' and the collate_fn preserves them."
            )

        # FAIL FAST: Ensure embedding layer is available
        if self.embed_layer is None:
            raise RuntimeError(
                "NLA Actor Worker: embed_layer not initialized! "
                "Cannot inject activations without access to the model's embedding layer. "
                "Check that _store_embedding_layer() successfully found the embedding layer during init_model()."
            )

        # Prepare for SGLang embedding-based injection
        input_ids = data.batch['input_ids']
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        print(f"NLA Actor: Prepared input_embeds for SGLang with shape: {input_embeds[0].shape}", f"second item shape: {input_embeds[1].shape if len(input_embeds) > 1 else 'None'}")

        # elif activation_vectors is not None and self.embed_layer is None:
        #     print("WARNING: Cannot inject activations - embedding layer not available")
        #     # Fall back to passing activation vectors through meta_info
        #     # The SGLang rollout will handle this with its own embedding layer
        #     meta = data.meta_info if data.meta_info is not None else {}
        #     if meta is data.meta_info:
        #         meta = dict(meta)
        #     meta['activation_vectors'] = activation_vectors

        #     # Still need to find injection positions
        #     batch_size = activation_vectors.shape[0]
        #     injection_positions = torch.ones(batch_size, dtype=torch.long)

        #     if hasattr(self, 'nla_wrapper') and self.nla_wrapper.injection_config.injection_token_id is not None:
        #         input_ids = data.batch['input_ids']
        #         injection_token_id = self.nla_wrapper.injection_config.injection_token_id
        #         injection_mask = (input_ids == injection_token_id)
        #         if injection_mask.any():
        #             injection_positions = injection_mask.float().argmax(dim=1)

        #     meta['injection_positions'] = injection_positions
        #     data.meta_info = meta

        # If no activation vectors, use standard generation
        return super().generate_sequences(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_input_embeddings(self, data: DataProto) -> DataProto:
        """Compute and attach input embeddings without triggering generation."""
        activation_vectors = self._extract_activation_vectors(data)
        if activation_vectors is None:
            return data

        input_ids = data.batch['input_ids']
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """
        Compute log probabilities with activation injection.

        Ensures activation vectors are passed during log prob computation.
        """
        # Extract activation vectors
        activation_vectors = None
        if data.meta_info:
            activation_vectors = data.meta_info.get('activation_vectors')

        if activation_vectors is None and "activation_vectors" in data.batch.keys():
            activation_vectors = data.batch["activation_vectors"]

        if activation_vectors is not None and hasattr(self, 'actor_nla_wrapper'):
            print(f"Computing log probs with activation vectors of shape: {activation_vectors.shape}")

            # Store activation vectors for the forward pass
            # The actor's forward method needs to receive the activation vectors
            # We'll modify the data to include them for the forward pass
            data.batch.update({"activation_vectors": activation_vectors})

            # Create a custom forward function that includes activation_vectors
            original_forward = self.actor.model.forward if hasattr(self.actor, 'model') else self.actor.forward

            def forward_with_activations(input_ids, attention_mask=None, **kwargs):
                # Add activation vectors to kwargs
                kwargs['activation_vectors'] = activation_vectors
                return original_forward(input_ids, attention_mask, **kwargs)

            # Temporarily replace the forward method
            if hasattr(self.actor, 'model'):
                self.actor.model.forward = forward_with_activations
            else:
                self.actor.forward = forward_with_activations

            # Call parent's compute_log_prob
            result = super().compute_log_prob(data)

            # Restore original forward method
            if hasattr(self.actor, 'model'):
                self.actor.model.forward = original_forward
            else:
                self.actor.forward = original_forward

            return result

        return super().compute_log_prob(data)
