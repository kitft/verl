"""Megatron-specific NLA workers for activation injection with tensor/pipeline parallelism support.

ARCHITECTURE OVERVIEW - Megatron Pipeline Parallelism (PP) Integration
========================================================================

This module implements activation injection for Megatron models with full support for:
- Tensor Parallelism (TP): Model layers sharded across GPUs
- Pipeline Parallelism (PP): Model stages distributed across GPUs
- Data Parallelism (DP): Batches distributed across GPU groups

KEY DESIGN DECISIONS FOR PP SUPPORT
------------------------------------

1. EMBEDDING-LAYER HOOK APPROACH (Not Pre-Modified Embeddings)
   - We use a forward hook on the embedding layer (PP stage 0 only)
   - The hook modifies embeddings during the forward pass
   - This integrates cleanly with Megatron's forward_backward_func scheduler

2. WHY HOOKS WORK WITH MEGATRON PP
   - PyTorch hooks execute synchronously during layer forward passes
   - Megatron's pipeline scheduler calls model stages in sequence
   - Stage 0 embedding hook fires → modifies embeddings → passes to Stage 1
   - Downstream stages receive already-modified embeddings
   - No special pipeline coordination needed

3. MICRO-BATCH TRACKING (CRITICAL FIX)
   - Megatron splits global batches into micro-batches for memory efficiency
   - The hook fires ONCE PER MICRO-BATCH, not once per global batch
   - We track which micro-batch is being processed via _current_micro_batch_offset
   - Each hook invocation uses the correct slice of activation vectors

4. STATE MANAGEMENT FLOW
   - compute_log_prob/update_policy wrapper:
     * Called with global batch (e.g., 32 samples)
     * Sets injection state: all 32 activation vectors + positions
     * Calls original method (triggers pipeline scheduler)
     * Clears injection state after completion
   - Hook execution (fires per micro-batch):
     * Micro-batch 1 (samples 0-7): uses activations[0:8]
     * Micro-batch 2 (samples 8-15): uses activations[8:16]
     * Micro-batch 3 (samples 16-23): uses activations[16:24]
     * Micro-batch 4 (samples 24-31): uses activations[24:32]

5. SGLANG ROLLOUT (DIFFERENT PATH)
   - generate_sequences uses _prepare_input_embeddings (not hooks)
   - Computes embeddings with injection before entering SGLang
   - Returns pre-modified embeddings via data.non_tensor_batch["input_embeds"]

PARALLELISM SUPPORT MATRIX
---------------------------
✅ TP (Tensor Parallelism): Any size - embedding layer handles TP internally
✅ PP (Pipeline Parallelism): Any size - hook only on stage 0, clean handoff
✅ DP (Data Parallelism): Any size - each DP rank processes independent batches
✅ SGLang rollout: Works with all parallelism configurations
✅ Training (compute_log_prob, update_policy): Works with all configurations

VALIDATION CHECKLIST
--------------------
Before deploying with PP>1, verify:
- Hook only registers on mpu.is_pipeline_first_stage()
- Hook uses _current_micro_batch_offset to index activation vectors
- Offset resets to 0 in _set_injection_state
- Offset increments by micro_batch_size after each hook invocation
- State clears in finally block after method completion
"""

import logging

import torch

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from verl.workers.megatron_workers import ActorRolloutRefWorker as MegatronActorRolloutRefWorker
from verl.workers.megatron_workers import CriticWorker as MegatronCriticWorker

# Megatron parallel state utils
try:
    from megatron.core import parallel_state as mpu
except ImportError:
    # Fallback for testing without Megatron
    class _MockMPU:
        @staticmethod
        def is_pipeline_first_stage():
            return True

        @staticmethod
        def is_pipeline_last_stage():
            return True

    mpu = _MockMPU()

from ..models.nla_wrapper import InjectionConfig
from ..utils.injection_manager import InjectionTokenManager

log = logging.getLogger(__name__)


class NLAMegatronActorRolloutRefWorker(MegatronActorRolloutRefWorker):
    """
    NLA-enabled Megatron actor rollout worker with activation injection support.

    Key differences from FSDP version:
    - Embedding layer only exists on first pipeline stage
    - Model accessed via self.actor_module[0] (list of modules)
    - Wraps higher-level methods (compute_log_prob, update_policy) instead of _forward_micro_batch
    - All embedding operations guarded by mpu.is_pipeline_first_stage()
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with NLA wrapper for Megatron."""
        log.debug("=" * 80)
        log.debug("NLA MEGATRON ACTOR WORKER: init_model() called")
        log.debug("=" * 80)

        # First call parent's init_model to set up the base Megatron model
        super().init_model()

        log.debug(f"NLA MEGATRON ACTOR: After super().init_model(), has rollout: {hasattr(self, 'rollout')}")
        log.debug(f"NLA MEGATRON ACTOR: Config has nla: {hasattr(self.config, 'nla')}")

        # Store reference to the embedding layer (only on first pipeline stage)
        self._store_embedding_layer()

        # Setup NLA injection if configured
        if hasattr(self.config, "nla"):
            log.debug("NLA MEGATRON ACTOR: Setting up NLA injection configuration")
            nla_config = self.config.get("nla", {})
            injection_config = nla_config.get("injection", {})

            # Configure injection settings
            self.injection_cfg = InjectionConfig(
                mode=injection_config.get("mode", "replace"),
                layer_indices=injection_config.get("layer_indices", [0]),
                projection_dim=injection_config.get("projection_dim", None),
                injection_token=injection_config.get("injection_token", None),
            )

            # Setup tokenizer and injection token manager
            tokenizer = self.tokenizer if hasattr(self, "tokenizer") else None
            if tokenizer is not None:
                injection_manager = InjectionTokenManager(tokenizer, self.injection_cfg.injection_token)
                self.injection_cfg.injection_token_id = injection_manager.token_id
                self.injection_cfg.injection_character = injection_manager.character
                object.__setattr__(self, "injection_manager", injection_manager)
                log.debug(
                    f"NLA MEGATRON ACTOR: Injection token: '{injection_manager.character}' (ID: {injection_manager.token_id})"
                )
            elif self.injection_cfg.injection_token_id is not None and self.injection_cfg.injection_token_id >= 0:
                object.__setattr__(self, "injection_manager", None)
            else:
                raise ValueError(
                    "Either provide a tokenizer for auto-selection or specify injection_token_id in config."
                )

            # Setup embedding injection (only on first pipeline stage)
            if mpu.is_pipeline_first_stage() and hasattr(self, "actor_module") and self.embed_layer is not None:
                log.debug("NLA MEGATRON ACTOR: Setting up activation injection on first pipeline stage")

                # Determine dimensions for projection layer
                hidden_dim = None
                activation_dim = nla_config.get("activation_dim", None)

                # Try to get hidden_dim from Megatron model
                if hasattr(self, "actor_module") and len(self.actor_module) > 0:
                    model = self.actor_module[0]
                    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                        hidden_dim = model.config.hidden_size

                if activation_dim is None:
                    activation_dim = hidden_dim

                # Create projection layer if dimensions mismatch
                if activation_dim is not None and hidden_dim is not None and activation_dim != hidden_dim:
                    log.debug(f"NLA MEGATRON ACTOR: Creating projection layer: {activation_dim} -> {hidden_dim}")
                    self._activation_projection = torch.nn.Linear(activation_dim, hidden_dim, bias=False)
                    # Move to same device as embedding layer
                    device = self.embed_layer.weight.device if hasattr(self.embed_layer, "weight") else None
                    if device is not None:
                        self._activation_projection = self._activation_projection.to(device)
                else:
                    self._activation_projection = None

                log.debug("NLA MEGATRON ACTOR: Registering embedding injection hook")
                # Register forward hook on embedding layer
                self._injection_hook_handle = self.embed_layer.register_forward_hook(self._embedding_injection_hook)

                # Wrap actor methods for state management
                if hasattr(self, "actor"):
                    log.debug("NLA MEGATRON ACTOR: Wrapping actor methods for injection state management")
                    self._wrap_actor_methods()

                    # Initialize telemetry
                    self._injection_telemetry = {
                        "method_calls": 0,
                        "calls_with_activations": 0,
                        "hook_calls": 0,
                        "actual_injections": 0,
                    }

    def _store_embedding_layer(self):
        """Store reference to Megatron model's embedding layer (first pipeline stage only)."""
        self.embed_layer = None

        # Only first pipeline stage has the embedding layer
        if not mpu.is_pipeline_first_stage():
            log.info("NLA MEGATRON ACTOR: Not on first pipeline stage, skipping embedding layer setup")
            return

        # Megatron models are stored in a list
        if hasattr(self, "actor_module") and len(self.actor_module) > 0:
            model = self.actor_module[0]

            # Debug: Print model structure
            log.info(f"NLA MEGATRON ACTOR: Model type: {type(model)}")
            log.info(f"NLA MEGATRON ACTOR: Model attributes: {dir(model)[:10]}...")  # First 10 attrs
            log.info(f"NLA MEGATRON ACTOR: Has 'embedding': {hasattr(model, 'embedding')}")
            log.info(f"NLA MEGATRON ACTOR: Has 'module': {hasattr(model, 'module')}")
            log.info(f"NLA MEGATRON ACTOR: Has 'model': {hasattr(model, 'model')}")

            # Try Megatron-specific patterns first
            if hasattr(model, "embedding") and hasattr(model.embedding, "word_embeddings"):
                self.embed_layer = model.embedding.word_embeddings
                log.info(f"✓ Successfully stored Megatron embedding layer: {type(self.embed_layer)}")
            # Try module wrapper (DistributedDataParallel)
            elif hasattr(model, "module"):
                inner_model = model.module
                log.info(f"NLA MEGATRON ACTOR: Found module wrapper, inner type: {type(inner_model)}")

                # Float16Module might wrap the actual model - unwrap again
                if hasattr(inner_model, "module"):
                    actual_model = inner_model.module
                    log.info(f"NLA MEGATRON ACTOR: Found nested module, actual type: {type(actual_model)}")
                else:
                    actual_model = inner_model

                # Now try to find embeddings
                if hasattr(actual_model, "embedding") and hasattr(actual_model.embedding, "word_embeddings"):
                    self.embed_layer = actual_model.embedding.word_embeddings
                    log.info(f"✓ Successfully stored embedding via module.module.embedding.word_embeddings")
                elif hasattr(actual_model, "model") and hasattr(actual_model.model, "embed_tokens"):
                    self.embed_layer = actual_model.model.embed_tokens
                    log.info(f"✓ Successfully stored embedding via module.module.model.embed_tokens")
                elif hasattr(inner_model, "embedding") and hasattr(inner_model.embedding, "word_embeddings"):
                    self.embed_layer = inner_model.embedding.word_embeddings
                    log.info(f"✓ Successfully stored embedding via module.embedding.word_embeddings")
                elif hasattr(inner_model, "model") and hasattr(inner_model.model, "embed_tokens"):
                    self.embed_layer = inner_model.model.embed_tokens
                    log.info(f"✓ Successfully stored embedding via module.model.embed_tokens")
            # Fallback to HF patterns
            elif hasattr(model, "get_input_embeddings"):
                self.embed_layer = model.get_input_embeddings()
                log.info("✓ Successfully stored embedding layer via get_input_embeddings")
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_layer = model.model.embed_tokens
                log.info("✓ Successfully stored embedding layer via model.embed_tokens")
            else:
                log.error("ERROR: Could not find embedding layer in Megatron actor model!")
                log.error(f"Model type: {type(model)}, Has module: {hasattr(model, 'module')}")
        else:
            log.error("ERROR: actor_module not found or empty")

    def _embedding_injection_hook(self, module, input, output):
        """Forward hook that injects activation vectors into embeddings (first pipeline stage only).

        CRITICAL: This hook fires once per micro-batch during Megatron's pipeline execution.
        We must track which micro-batch we're processing to use the correct activation vectors.
        """
        # This should only fire on first pipeline stage, but double-check
        if not mpu.is_pipeline_first_stage():
            return output

        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["hook_calls"] += 1

        if not hasattr(self, "_current_activation_vectors") or self._current_activation_vectors is None:
            return output

        activation_vectors = self._current_activation_vectors
        injection_positions = self._current_injection_positions

        # Get current micro-batch offset to index into the global activation vectors
        micro_batch_offset = getattr(self, "_current_micro_batch_offset", 0)
        micro_batch_size = output.shape[0]

        log.debug(
            f"🔥 NLA MEGATRON HOOK FIRING: micro_batch_offset={micro_batch_offset}, micro_batch_size={micro_batch_size}, total_activations={activation_vectors.shape[0]}"
        )

        # Clone output to avoid in-place modification issues
        output = output.clone()

        injections_performed = 0

        for local_idx in range(micro_batch_size):
            # Map from local micro-batch index to global batch index
            global_idx = micro_batch_offset + local_idx

            if global_idx >= len(injection_positions):
                log.debug(
                    f"WARNING: global_idx {global_idx} exceeds injection_positions length {len(injection_positions)}"
                )
                continue

            pos = injection_positions[global_idx]
            if pos is None:
                continue

            # Handle tensor positions
            if isinstance(pos, torch.Tensor):
                if pos.numel() == 0:
                    continue
                if pos.numel() > 1:
                    log.debug(f"WARNING: Multiple injection positions for global_idx {global_idx}, using first")
                pos = pos[0].item()

            # Inject activation at position
            if 0 <= pos < output.shape[1]:
                activation = activation_vectors[global_idx]

                # Apply projection if needed
                if activation.shape[-1] != output.shape[-1]:
                    if hasattr(self, "_activation_projection") and self._activation_projection is not None:
                        activation = self._activation_projection(activation)
                    else:
                        raise RuntimeError(
                            f"Activation dimension mismatch: got {activation.shape[-1]}, expected {output.shape[-1]}"
                        )

                # Inject with correct dtype (using local_idx for output indexing)
                output[local_idx, pos] = activation.to(device=output.device, dtype=output.dtype)
                injections_performed += 1

        log.debug(
            f"✓ NLA MEGATRON HOOK: Performed {injections_performed}/{micro_batch_size} injections for global indices [{micro_batch_offset}:{micro_batch_offset + micro_batch_size}]"
        )

        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["actual_injections"] += injections_performed

        # Increment offset for next micro-batch
        self._current_micro_batch_offset = micro_batch_offset + micro_batch_size

        return output

    def _set_injection_state(self, activation_vectors: torch.Tensor, injection_positions: list):
        """Set activation vectors and positions for the next forward pass.

        This is called once per global batch (e.g., in compute_log_prob or update_policy).
        The hook will fire multiple times (once per micro-batch) and must track its progress.
        """
        self._current_activation_vectors = activation_vectors
        self._current_injection_positions = injection_positions
        self._current_micro_batch_offset = 0  # Reset offset for new global batch

    def _clear_injection_state(self):
        """Clear activation injection state after forward pass."""
        self._current_activation_vectors = None
        self._current_injection_positions = None
        self._current_micro_batch_offset = 0

    def _find_injection_positions_from_ids(self, input_ids: torch.Tensor) -> list:
        """Find positions of injection tokens in input_ids."""
        positions = []
        injection_token_id = self.injection_cfg.injection_token_id

        for i in range(input_ids.shape[0]):
            batch_positions = (input_ids[i] == injection_token_id).nonzero(as_tuple=False).squeeze(-1)
            positions.append(batch_positions if batch_positions.numel() > 0 else None)

        return positions

    def _wrap_actor_methods(self):
        """Wrap actor's methods to manage injection state around forward passes."""
        # Wrap compute_log_prob
        if hasattr(self.actor, "compute_log_prob"):
            log.debug("NLA MEGATRON ACTOR: Wrapping compute_log_prob")
            self._wrap_method("compute_log_prob")

        # Note: We don't wrap update_policy because it takes a dataloader, not a single DataProto
        # The injection state is managed at the worker level via the update_actor() override

    def _wrap_method(self, method_name: str):
        """Generic wrapper to manage injection state around an actor method."""
        original_method = getattr(self.actor, method_name)

        def wrapped_method(data: DataProto, *args, **kwargs):
            # Track calls
            if hasattr(self, "_injection_telemetry"):
                self._injection_telemetry["method_calls"] += 1

            # Only manage injection state on first pipeline stage
            has_activations = "activation_vectors" in data.batch
            should_inject = has_activations and mpu.is_pipeline_first_stage()

            if should_inject:
                if hasattr(self, "_injection_telemetry"):
                    self._injection_telemetry["calls_with_activations"] += 1

                activation_vectors = data.batch["activation_vectors"]
                input_ids = data.batch["input_ids"]
                injection_positions = self._find_injection_positions_from_ids(input_ids)

                log.debug(f"🎯 NLA MEGATRON: Setting injection state for {method_name}")
                log.debug(f"   Activations shape: {activation_vectors.shape}")
                log.debug(
                    f"   Injection positions: {[(p.item() if isinstance(p, torch.Tensor) and p.numel() > 0 else p) for p in injection_positions]}"
                )

                self._set_injection_state(activation_vectors, injection_positions)

            try:
                return original_method(data, *args, **kwargs)
            finally:
                if should_inject:
                    self._clear_injection_state()
                    log.debug(f"✓ NLA MEGATRON: Cleared injection state after {method_name}")

        setattr(self.actor, method_name, wrapped_method)

    def _extract_activation_vectors(self, data: DataProto):
        """Extract activation vectors from DataProto."""
        activation_vectors = None
        if data.batch is not None:
            batch_keys = getattr(data.batch, "keys", lambda: [])()
            if "activation_vectors" in batch_keys:
                activation_vectors = data.batch["activation_vectors"]
        return activation_vectors

    def _prepare_input_embeddings(
        self, data: DataProto, input_ids: torch.Tensor, activation_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare input embeddings with activation injection for SGLang rollout.

        This method handles both FSDP and Megatron backends:
        - FSDP: DTensor sharding requires full_tensor() gather
        - Megatron: Tensor-parallel embeddings handle all-reduce internally

        Returns:
            List of embedding tensors (one per sequence, padding removed)
        """
        if self.embed_layer is None:
            raise RuntimeError("Embedding layer not available. Cannot compute input embeddings.")

        # Move input_ids to embedding layer device
        target_device = self.embed_layer.weight.device if hasattr(self.embed_layer, "weight") else None
        if target_device is not None:
            input_ids = input_ids.to(target_device)

        with torch.no_grad():
            # Check if this is FSDP DTensor (sharded)
            try:
                from torch.distributed._tensor import DTensor

                if isinstance(self.embed_layer.weight, DTensor):
                    # FSDP: All-gather the sharded embedding weights
                    full_weight = self.embed_layer.weight.full_tensor()
                    input_embeds = torch.nn.functional.embedding(
                        input_ids, full_weight, padding_idx=getattr(self.embed_layer, "padding_idx", None)
                    )
                else:
                    # Megatron (or non-sharded): Call embedding layer directly
                    # For Megatron, the embedding layer's forward() handles TP all-reduce internally
                    input_embeds = self.embed_layer(input_ids)
            except ImportError:
                # DTensor not available (e.g., older PyTorch), fall back to direct call
                input_embeds = self.embed_layer(input_ids)

        # Inject activation vectors
        activation_vectors = activation_vectors.to(input_embeds.device)
        batch_size = activation_vectors.shape[0]
        injection_positions = torch.ones(batch_size, dtype=torch.long, device=activation_vectors.device)

        # Find injection token positions
        injection_token_id = None
        if hasattr(self, "injection_cfg") and self.injection_cfg.injection_token_id is not None:
            injection_token_id = self.injection_cfg.injection_token_id

        if injection_token_id is not None:
            injection_mask = input_ids == injection_token_id
            if injection_mask.any():
                injection_positions = injection_mask.float().argmax(dim=1)
                log.debug(f"NLA Megatron Actor: Found injection tokens at positions: {injection_positions}")

        hidden_dim = input_embeds.shape[-1]

        # Validate activation vector dimensions match model hidden size
        if activation_vectors.shape[-1] != hidden_dim:
            raise ValueError(
                f"Activation vector dimension mismatch: activation_vectors have shape {activation_vectors.shape} "
                f"(dimension {activation_vectors.shape[-1]}), but model hidden_dim is {hidden_dim}. "
                f"activation_dim must equal model hidden_size. Check your config and dataset."
            )

        # Inject activations at specified positions
        from collections import defaultdict

        pos_token_counts = defaultdict(list)

        for i in range(batch_size):
            pos = injection_positions[i].item()
            if 0 <= pos < input_embeds.shape[1]:
                input_embeds[i, pos] = activation_vectors[i]
                token = input_ids[i, pos].item()
                pos_token_counts[(pos, token)].append(i)

        if len(pos_token_counts) > 1:
            for (pos, token), idxs in pos_token_counts.items():
                log.debug(f"NLA Megatron Actor: Injected activation at position {pos} for {len(idxs)} sequence(s)")

        # Remove padding using attention mask
        attention_mask = data.batch["attention_mask"]
        if attention_mask is None:
            raise ValueError("Attention mask is required to remove padding vectors.")

        # Return list of embeddings (one per sequence, padding removed)
        input_embeds_list = []
        for i in range(input_embeds.shape[0]):
            mask = attention_mask[i].bool()
            input_embeds_list.append(input_embeds[i][mask].cpu())

        return input_embeds_list

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        Generate sequences with activation injection for Megatron backend.

        For SGLang rollout: Prepares activation vectors and injection positions
        to be used as custom input embeddings.
        """
        log.debug("=" * 80)
        log.debug("NLA MEGATRON ACTOR: generate_sequences() called")
        log.debug("=" * 80)

        # Extract activation vectors
        activation_vectors = self._extract_activation_vectors(data)
        log.debug(
            f"NLA MEGATRON ACTOR: Extracted activation_vectors: {activation_vectors.shape if activation_vectors is not None else None}"
        )

        # FAIL FAST: Ensure activation vectors are present
        if activation_vectors is None:
            raise ValueError(
                "NLA Megatron Actor Worker: activation_vectors missing from batch! "
                "NLA training requires activation vectors for all samples."
            )

        # FAIL FAST: Ensure embedding layer is available
        # Note: For Megatron with pipeline parallelism, embedding layer only exists on first stage
        if not mpu.is_pipeline_first_stage():
            log.debug("NLA MEGATRON ACTOR: Not on first pipeline stage, skipping embedding preparation")
            # Non-first stages don't have embeddings, just pass through
            return super().generate_sequences(data)

        if self.embed_layer is None:
            raise RuntimeError(
                "NLA Megatron Actor Worker: embed_layer not initialized on first pipeline stage! "
                "Cannot inject activations without access to the model's embedding layer."
            )

        # Prepare input embeddings with activation injection for SGLang
        input_ids = data.batch["input_ids"]
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        log.debug(f"NLA Megatron Actor: Prepared input_embeds for SGLang, first item shape: {input_embeds[0].shape}")

        return super().generate_sequences(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_input_embeddings(self, data: DataProto) -> DataProto:
        """Compute and attach input embeddings without triggering generation."""
        activation_vectors = self._extract_activation_vectors(data)
        if activation_vectors is None:
            return data

        # Only prepare embeddings on first pipeline stage
        if not mpu.is_pipeline_first_stage():
            return data

        if self.embed_layer is None:
            log.debug("WARNING: embed_layer not available, skipping embedding preparation")
            return data

        input_ids = data.batch["input_ids"]
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        return data

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def compute_log_prob(self, data: DataProto):
        """Compute log probabilities with activation injection."""
        log.debug("🎯 NLA MEGATRON ACTOR WORKER: compute_log_prob() CALLED")

        # Ensure activation vectors are in batch
        activation_vectors = self._extract_activation_vectors(data)
        if activation_vectors is not None:
            if "activation_vectors" not in data.batch.keys():
                data.batch["activation_vectors"] = activation_vectors
            log.debug(f"✓ compute_log_prob with activation vectors: {activation_vectors.shape}")

        # Call parent - wrapped actor method will handle injection state
        result = super().compute_log_prob(data)
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def update_actor(self, data: DataProto):
        """Update actor with activation injection."""
        log.debug("🎯 NLA MEGATRON ACTOR WORKER: update_actor() CALLED")

        # Ensure activation vectors are in batch
        activation_vectors = self._extract_activation_vectors(data)
        if activation_vectors is not None:
            if "activation_vectors" not in data.batch.keys():
                data.batch["activation_vectors"] = activation_vectors
            log.debug(f"✓ update_actor with activation vectors: {activation_vectors.shape}")

        # Set injection state for the entire training batch
        # The forward hooks will fire for each micro-batch during training
        if activation_vectors is not None and mpu.is_pipeline_first_stage():
            input_ids = data.batch["input_ids"]
            injection_positions = self._find_injection_positions_from_ids(input_ids)
            log.debug("Setting injection state for update_actor")
            self._set_injection_state(activation_vectors, injection_positions)

        try:
            # Call parent's update_actor which will process the dataloader
            result = super().update_actor(data)
        finally:
            # Clear injection state after training
            if activation_vectors is not None and mpu.is_pipeline_first_stage():
                log.debug("Clearing injection state after update_actor")
                self._clear_injection_state()

        return result


class NLAMegatronCriticWorker(MegatronCriticWorker):
    """NLA Critic worker for Megatron that outputs activation vectors instead of scalars."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize base Megatron critic then wrap with NLA critic behavior."""
        super().init_model()

        # Import NLA Megatron critic wrapper
        from verl.nla.workers.nla_megatron_critic import NLAMegatronCritic
        import logging
        log = logging.getLogger(__name__)

        # Extract critic_optimizer_config from the parent's critic before we replace it
        critic_optimizer_config = self.critic.critic_optimizer_config

        # Debug logging
        log.info(f"NLA Megatron Critic Worker init_model:")
        log.info(f"  hf_config: {self.hf_config is not None}")
        log.info(f"  tf_config: {self.tf_config is not None}")
        log.info(f"  critic_optimizer_config: {critic_optimizer_config is not None}")

        # Wrap the Megatron critic with NLA logic
        # Pass hf_config, tf_config, and critic_optimizer_config as kwargs
        self.critic = NLAMegatronCritic(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            tokenizer=self.tokenizer,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_optimizer_config=critic_optimizer_config,
        )

    def compute_activation_predictions(self, response_ids, attention_mask):
        """Predict activation vectors for the final response token using Megatron critic.

        Delegates to the critic's forward logic to ensure consistency with training,
        including critic prompt prepending if configured and respecting output_layer_index.
        """
        device = get_device_id()
        response_ids = response_ids.to(device)
        if attention_mask is None:
            raise ValueError("Attention mask is required for NLA Megatron critic worker")
        attention_mask = attention_mask.to(device)

        # Handle model offloading if enabled
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)

        # Megatron critic_module is a list
        critic_model = self.critic_module[0]

        was_training = critic_model.training
        try:
            critic_model.eval()
            with torch.no_grad():
                # Build minimal micro-batch for the critic's forward path
                micro_batch = {
                    "responses": response_ids,
                    "attention_mask": attention_mask,
                    "position_ids": torch.zeros_like(attention_mask, dtype=torch.long),
                }

                # Use critic's forward logic (handles prompt prepending, layer selection, etc.)
                full_activations = self.critic._forward_micro_batch(micro_batch)

                # Extract last token using critic's pooling logic
                activations = self.critic.extract_predicted_activations(
                    full_activations, attention_mask, pooling="last"
                )
        finally:
            if was_training:
                critic_model.train(True)

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)

        return activations
