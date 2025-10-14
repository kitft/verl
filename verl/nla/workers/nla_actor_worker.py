"""Custom actor worker for NLA that handles activation injection."""

import logging
import os
from typing import Optional

import torch

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker as FSDPActorRolloutRefWorker

from ..models.nla_wrapper import InjectionConfig, NLAModelWrapper
from ..utils.injection_manager import InjectionTokenManager

log = logging.getLogger(__name__)


class NLAActorRolloutRefWorker(FSDPActorRolloutRefWorker):
    """
    NLA-enabled actor rollout worker that handles activation injection.

    This worker extends the base FSDP actor worker with NLA capabilities
    for activation injection during generation.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with NLA wrapper."""
        log.debug("=" * 80)
        log.debug("NLA ACTOR WORKER: init_model() called")
        log.debug("=" * 80)

        # First call parent's init_model to set up the base model
        super().init_model()

        log.debug(f"NLA ACTOR: After super().init_model(), has rollout: {hasattr(self, 'rollout')}")
        log.debug(f"NLA ACTOR: Config has nla: {hasattr(self.config, 'nla')}")
        if hasattr(self.config, "nla"):
            log.debug(f"NLA ACTOR: Config.nla = {self.config.nla}")

        # Store reference to the embedding layer for later use
        self._store_embedding_layer()

        # Now wrap the rollout model with NLA capabilities (if not using SGLang)
        if hasattr(self, "rollout") and hasattr(self.config, "nla"):
            log.debug("NLA ACTOR: Setting up NLA injection configuration")
            # Extract NLA configuration
            nla_config = self.config.get("nla", {})
            injection_config = nla_config.get("injection", {})
            log.debug(f"NLA ACTOR: injection_config = {injection_config}")

            # Configure injection settings
            self.injection_cfg = InjectionConfig(
                mode=injection_config.get("mode", "replace"),
                layer_indices=injection_config.get("layer_indices", [0]),
                projection_dim=injection_config.get("projection_dim", None),
                injection_token=injection_config.get("injection_token", None),
            )

            # For SGLang rollout (nla_sglang), we use embedding-based injection
            # The rollout itself doesn't need wrapping - we prepare input_embeds before calling it
            rollout_name = self.config.rollout.name if hasattr(self.config, "rollout") else "unknown"
            log.debug(f"NLA ACTOR: Rollout type: {rollout_name}")

            if rollout_name == "nla_sglang":
                log.debug("NLA ACTOR: Using SGLang with embedding-based injection (no rollout wrapping needed)")
                # Store injection config for use in generate_sequences
                self.nla_injection_mode = "embedding"
            elif hasattr(self.rollout, "module"):
                # For other rollouts that have a module attribute (e.g., HF rollout)
                log.debug("NLA ACTOR: Wrapping rollout module with NLA capabilities")
                base_model = self.rollout.module
                hidden_dim = base_model.config.hidden_size if hasattr(base_model.config, "hidden_size") else None
                activation_dim = nla_config.get("activation_dim", hidden_dim or 768)

                # Get tokenizer from model config
                tokenizer = self.model_config.tokenizer if hasattr(self, "model_config") else None

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
                self.nla_injection_mode = "wrapper"

                log.debug("Wrapped rollout model with NLAModelWrapper")
                log.debug(
                    f"Injection token: '{self.nla_wrapper.injection_config.injection_character}' (ID: {self.nla_wrapper.injection_config.injection_token_id})"
                )
            else:
                log.debug(
                    f"WARNING: Rollout type '{rollout_name}' doesn't have a module attribute and isn't nla_sglang"
                )
                self.nla_injection_mode = "unknown"

            tokenizer = self.model_config.tokenizer if hasattr(self, "model_config") else None
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

            # Set up actor model for activation injection at embedding layer
            if hasattr(self, "actor_module_fsdp") and self.embed_layer is not None:
                log.debug("NLA ACTOR: Setting up activation injection for actor model")

                # Determine dimensions for projection layer if needed
                hidden_dim = (
                    self.base_model.config.hidden_size
                    if hasattr(self, "base_model") and hasattr(self.base_model, "config")
                    else None
                )
                if hidden_dim is None and hasattr(self, "actor_module_fsdp"):
                    if hasattr(self.actor_module_fsdp.config, "hidden_size"):
                        hidden_dim = self.actor_module_fsdp.config.hidden_size

                activation_dim = nla_config.get("activation_dim", hidden_dim)

                # Pre-create projection layer if dimensions mismatch
                if activation_dim is not None and hidden_dim is not None and activation_dim != hidden_dim:
                    log.debug(f"NLA ACTOR: Creating projection layer: {activation_dim} -> {hidden_dim}")
                    self._activation_projection = torch.nn.Linear(activation_dim, hidden_dim, bias=False)
                    # Move to same device as embedding layer
                    device = self.embed_layer.weight.device if hasattr(self.embed_layer, "weight") else None
                    if device is not None:
                        self._activation_projection = self._activation_projection.to(device)
                    log.debug(f"NLA ACTOR: Projection layer created and moved to device: {device}")
                else:
                    self._activation_projection = None
                    if activation_dim != hidden_dim:
                        log.debug("WARNING: Could not determine dimensions for projection layer")

                log.debug("NLA ACTOR: Registering embedding injection hook on actor model")
                # Register forward hook on embedding layer for automatic injection
                # This works with FSDP since we're only hooking the embedding layer
                self._injection_hook_handle = self.embed_layer.register_forward_hook(self._embedding_injection_hook)
                log.debug("NLA ACTOR: Embedding injection hook registered successfully")

                # Wrap the actor's methods to manage injection
                if hasattr(self, "actor"):
                    # Wrap _forward_micro_batch for injection state management
                    if hasattr(self.actor, "_forward_micro_batch"):
                        log.debug("NLA ACTOR: Wrapping actor's _forward_micro_batch for injection state management")
                        log.debug(f"NLA ACTOR: self.actor type: {type(self.actor)}")
                        log.debug(f"NLA ACTOR: self.actor class: {self.actor.__class__.__name__}")
                        self._wrap_actor_forward_micro_batch()
                        log.debug("NLA ACTOR: ✓ Actor _forward_micro_batch wrapped successfully")

                    # Wrap compute_log_prob to add activation_vectors to select_keys
                    if hasattr(self.actor, "compute_log_prob"):
                        log.debug("NLA ACTOR: Wrapping actor's compute_log_prob to preserve activation_vectors")
                        self._wrap_actor_compute_log_prob_method()
                        log.debug("NLA ACTOR: ✓ Actor compute_log_prob wrapped successfully")

                    # Wrap update_policy to add activation_vectors to select_keys
                    if hasattr(self.actor, "update_policy"):
                        log.debug("NLA ACTOR: Wrapping actor's update_policy to preserve activation_vectors")
                        self._wrap_actor_update_policy_method()
                        log.debug("NLA ACTOR: ✓ Actor update_policy wrapped successfully")

                    # Initialize telemetry counters
                    self._injection_telemetry = {
                        "forward_micro_batch_calls": 0,
                        "forward_with_activations": 0,
                        "forward_without_activations": 0,
                        "hook_calls": 0,
                        "actual_injections": 0,
                    }
                else:
                    log.debug("NLA ACTOR: WARNING - Actor not found, cannot wrap methods")

    def _store_embedding_layer(self):
        """Store reference to the model's embedding layer for input embedding computation."""
        self.embed_layer = None

        # Try to get embedding layer from actor model
        if hasattr(self, "actor_module_fsdp"):
            model = self.actor_module_fsdp

            # Try different access patterns for different model architectures
            if hasattr(model, "get_input_embeddings"):
                self.embed_layer = model.get_input_embeddings()
            elif hasattr(model, "embed_tokens"):
                self.embed_layer = model.embed_tokens
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_layer = model.model.embed_tokens
            elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                self.embed_layer = model.transformer.wte  # GPT-style
            elif hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
                self.embed_layer = model.bert.embeddings.word_embeddings  # BERT-style

            if self.embed_layer is not None:
                log.debug("Successfully stored embedding layer from actor model")
            else:
                log.debug("WARNING: Could not find embedding layer in actor model")

    def _embedding_injection_hook(self, module, input, output):
        """Forward hook that injects activation vectors into embeddings.

        This hook is registered on the embedding layer and automatically injects
        activation vectors when they are available in the injection state.
        """
        # Track hook calls
        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["hook_calls"] += 1

        if not hasattr(self, "_current_activation_vectors") or self._current_activation_vectors is None:
            return output

        activation_vectors = self._current_activation_vectors
        injection_positions = self._current_injection_positions

        log.debug(
            f"🔥 NLA HOOK FIRING: Injecting activations - batch_size={output.shape[0]}, activation_vectors.shape={activation_vectors.shape}"
        )

        # Clone output to avoid in-place modification issues with autograd
        output = output.clone()

        # Inject activations at specified positions
        batch_size = output.shape[0]
        injections_performed = 0

        for batch_idx in range(batch_size):
            if batch_idx >= len(injection_positions):
                continue

            pos = injection_positions[batch_idx]
            if pos is None:
                continue

            # Handle tensor positions (from _find_injection_positions)
            if isinstance(pos, torch.Tensor):
                if pos.numel() == 0:
                    continue
                # If multiple positions, only use first one and warn
                if pos.numel() > 1:
                    log.debug(
                        f"WARNING: Multiple injection positions found for batch {batch_idx}: {pos.tolist()}. Using first position only."
                    )
                pos = pos[0].item()

            # Inject activation at position
            if 0 <= pos < output.shape[1]:
                activation = activation_vectors[batch_idx]

                # Handle dimension mismatch with pre-created projection layer
                if activation.shape[-1] != output.shape[-1]:
                    if hasattr(self, "_activation_projection") and self._activation_projection is not None:
                        activation = self._activation_projection(activation)
                    else:
                        raise RuntimeError(
                            f"Activation dimension mismatch: got {activation.shape[-1]}, expected {output.shape[-1]}. "
                            f"No projection layer configured. Check NLA config activation_dim setting."
                        )

                # Convert to correct device AND dtype (critical fix)
                output[batch_idx, pos] = activation.to(device=output.device, dtype=output.dtype)
                injections_performed += 1

        # Safe logging of injection positions
        formatted_positions = self._format_injection_positions_for_logging(injection_positions[:batch_size])
        log.debug(f"✓ NLA HOOK: Performed {injections_performed}/{batch_size} injections at positions {formatted_positions}")

        # Track actual injections
        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["actual_injections"] += injections_performed

        return output

    def _set_injection_state(self, activation_vectors: torch.Tensor, injection_positions: list):
        """Set activation vectors and positions for the next forward pass."""
        self._current_activation_vectors = activation_vectors
        self._current_injection_positions = injection_positions

    def _clear_injection_state(self):
        """Clear activation injection state after forward pass."""
        self._current_activation_vectors = None
        self._current_injection_positions = None

    def _find_injection_positions_from_ids(self, input_ids: torch.Tensor) -> list:
        """Find positions of injection tokens in input_ids.

        Returns list of positions (one per batch item), where each position
        is either None (no injection token found) or a tensor of positions.
        """
        positions = []
        injection_token_id = self.injection_cfg.injection_token_id

        for i in range(input_ids.shape[0]):
            # Find all occurrences of the injection token
            batch_positions = (input_ids[i] == injection_token_id).nonzero(as_tuple=False).squeeze(-1)
            positions.append(batch_positions if batch_positions.numel() > 0 else None)

        return positions

    def _format_injection_positions_for_logging(
        self, injection_positions: list[Optional[torch.Tensor]]
    ) -> list[int | list[int] | None]:
        """Format injection positions for safe logging without tensor conversion errors."""
        formatted_positions = []
        for pos in injection_positions:
            if pos is None:
                formatted_positions.append(None)
            elif isinstance(pos, torch.Tensor):
                if pos.numel() == 0:
                    formatted_positions.append(None)
                elif pos.numel() == 1:
                    formatted_positions.append(pos.item())
                else:
                    # Multiple positions - convert to list
                    formatted_positions.append(pos.tolist())
            else:
                formatted_positions.append(pos)
        return formatted_positions

    def _log_multiple_injection_tokens_to_wandb(
        self, input_ids: torch.Tensor, injection_positions: list[Optional[torch.Tensor]], step: Optional[int] = None
    ):
        """Log sequences with multiple injection tokens to wandb for debugging."""
        try:
            # Check if wandb is available and we're on rank 0
            if os.environ.get("RANK", "0") != "0":
                return

            import wandb

            # Find sequences with multiple injection tokens
            sequences_to_log = []
            for i, pos in enumerate(injection_positions):
                if pos is not None and isinstance(pos, torch.Tensor) and pos.numel() > 1:
                    # Decode the sequence if tokenizer is available
                    sequence_text = None
                    if hasattr(self, "tokenizer") and self.tokenizer is not None:
                        try:
                            sequence_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                        except Exception as e:
                            log.warning(f"Failed to decode sequence for wandb logging: {e}")

                    sequences_to_log.append(
                        {
                            "batch_idx": i,
                            "positions": pos.tolist(),
                            "sequence_text": sequence_text,
                            "input_ids": input_ids[i].tolist(),
                        }
                    )

            # Log to wandb if we found multiple injection tokens
            if sequences_to_log:
                wandb.log(
                    {
                        "debug/multiple_injection_tokens": wandb.Table(
                            columns=["batch_idx", "positions", "sequence_text", "input_ids"],
                            data=[
                                (
                                    seq["batch_idx"],
                                    str(seq["positions"]),
                                    seq["sequence_text"] or "N/A",
                                    str(seq["input_ids"]),
                                )
                                for seq in sequences_to_log
                            ],
                        )
                    },
                    step=step,
                )

                log.warning(f"Found {len(sequences_to_log)} sequences with multiple injection tokens. Logged to wandb.")

        except ImportError:
            log.debug("wandb not available for logging multiple injection tokens")
        except Exception as e:
            log.warning(f"Failed to log multiple injection tokens to wandb: {e}")

    def _wrap_actor_compute_log_prob_method(self):
        """Wrap actor's compute_log_prob to add activation_vectors to select_keys."""
        original_compute_log_prob = self.actor.compute_log_prob

        def compute_log_prob_with_activation_vectors(data: DataProto, calculate_entropy=False):
            """Wrapped version that adds activation_vectors to select_keys."""
            # Check if activation_vectors present
            has_activations = "activation_vectors" in data.batch.keys()

            if has_activations:
                log.debug("🔧 Wrapped compute_log_prob: Found activation_vectors in batch")
                # The actor's compute_log_prob will do data.select() with hardcoded keys
                # We need to intercept and add activation_vectors BEFORE that happens
                # Unfortunately we can't easily intercept the select() call inside the method
                # So we monkey-patch data.select temporarily

                original_select = data.select

                def select_with_activation_vectors(batch_keys=None, non_tensor_batch_keys=None):
                    """Modified select that preserves activation_vectors."""
                    if batch_keys is not None and "activation_vectors" not in batch_keys:
                        batch_keys = list(batch_keys) + ["activation_vectors"]
                        log.debug("✓ Added 'activation_vectors' to select_keys in compute_log_prob")
                    return original_select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

                # Temporarily replace select method
                data.select = select_with_activation_vectors

            try:
                # Call original method
                return original_compute_log_prob(data, calculate_entropy=calculate_entropy)
            finally:
                # Restore original select if we modified it
                if has_activations:
                    data.select = original_select

        self.actor.compute_log_prob = compute_log_prob_with_activation_vectors

    def _wrap_actor_update_policy_method(self):
        """Wrap actor's update_policy to add activation_vectors to select_keys."""
        original_update_policy = self.actor.update_policy

        def update_policy_with_activation_vectors(data: DataProto):
            """Wrapped version that adds activation_vectors to select_keys."""
            has_activations = "activation_vectors" in data.batch.keys()

            if has_activations:
                log.debug("🔧 Wrapped update_policy: Found activation_vectors in batch")

                original_select = data.select

                def select_with_activation_vectors(batch_keys=None, non_tensor_batch_keys=None):
                    """Modified select that preserves activation_vectors."""
                    if batch_keys is not None and "activation_vectors" not in batch_keys:
                        batch_keys = list(batch_keys) + ["activation_vectors"]
                        log.debug("✓ Added 'activation_vectors' to select_keys in update_policy")
                    return original_select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

                data.select = select_with_activation_vectors

            try:
                return original_update_policy(data)
            finally:
                if has_activations:
                    data.select = original_select

        self.actor.update_policy = update_policy_with_activation_vectors

    def _wrap_actor_forward_micro_batch(self):
        """Wrap the actor's _forward_micro_batch to manage injection state.

        This wraps the DataParallelPPOActor's _forward_micro_batch method to:
        1. Extract activation vectors from the micro_batch
        2. Set up injection state before forward pass
        3. Clear injection state after forward pass
        """
        # Store reference to original method
        original_forward_micro_batch = self.actor._forward_micro_batch

        # Create wrapped version
        def _forward_micro_batch_with_injection(micro_batch, temperature, calculate_entropy=False):
            """Wrapped version that handles activation injection."""
            # Track all calls
            if hasattr(self, "_injection_telemetry"):
                self._injection_telemetry["forward_micro_batch_calls"] += 1

            # Check if this micro-batch has activation vectors
            has_activations = "activation_vectors" in micro_batch

            # Determine which code path we're in by inspecting the stack
            import traceback

            stack = traceback.extract_stack()
            calling_function = None
            for frame in reversed(stack):
                if "compute_log_prob" in frame.name:
                    calling_function = "compute_log_prob"
                    break
                elif "update_policy" in frame.name:
                    calling_function = "update_policy"
                    break

            log.debug(f"\n{'=' * 80}")
            log.debug("📊 WRAPPED _forward_micro_batch CALLED")
            log.debug(f"   Called from: {calling_function or 'UNKNOWN'}")
            log.debug(f"   Has activation_vectors: {has_activations}")
            log.debug(f"   calculate_entropy: {calculate_entropy}")
            log.debug(f"   Temperature: {temperature}")
            if has_activations:
                activation_vectors = micro_batch["activation_vectors"]
                log.debug(f"   Activation vectors shape: {activation_vectors.shape}")
                log.debug(f"   Input IDs shape: {micro_batch['input_ids'].shape}")
            log.debug(f"{'=' * 80}\n")

            if has_activations:
                if hasattr(self, "_injection_telemetry"):
                    self._injection_telemetry["forward_with_activations"] += 1

                activation_vectors = micro_batch["activation_vectors"]
                input_ids = micro_batch["input_ids"]

                # Find injection positions in this micro-batch
                injection_positions = self._find_injection_positions_from_ids(input_ids)

                # Log multiple injection tokens to wandb for debugging
                self._log_multiple_injection_tokens_to_wandb(input_ids, injection_positions)

                # Safe logging of injection positions
                formatted_positions = self._format_injection_positions_for_logging(injection_positions)
                log.debug(f"🎯 Found injection positions: {formatted_positions}")

                # Set injection state for the hook
                self._set_injection_state(activation_vectors, injection_positions)
                log.debug("✓ Injection state set - hook will fire during embedding forward pass")

                # Remove activation_vectors from micro_batch to avoid passing to model
                # (the hook will handle injection)
                micro_batch = {k: v for k, v in micro_batch.items() if k != "activation_vectors"}
            else:
                if hasattr(self, "_injection_telemetry"):
                    self._injection_telemetry["forward_without_activations"] += 1
                log.debug("⚠️  No activation vectors in this micro-batch")

            try:
                # Call original method - the hook will inject during forward pass
                result = original_forward_micro_batch(micro_batch, temperature, calculate_entropy)
                log.debug("✓ Original _forward_micro_batch completed")
                return result
            finally:
                # Always clear injection state
                if has_activations:
                    self._clear_injection_state()
                    log.debug("✓ Injection state cleared\n")

        # Replace the actor's method with our wrapped version
        self.actor._forward_micro_batch = _forward_micro_batch_with_injection

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
            batch_keys = getattr(data.batch, "keys", lambda: [])()
            if "activation_vectors" in batch_keys:
                activation_vectors = data.batch["activation_vectors"]

        if activation_vectors is None and data.meta_info and data.meta_info.get("activation_vectors") is not None:
            raise ValueError("Activation vectors must be provided via batch['activation_vectors']")

        return activation_vectors

    def _prepare_input_embeddings(
        self, data: DataProto, input_ids: torch.Tensor, activation_vectors: torch.Tensor
    ) -> torch.Tensor:
        if self.embed_layer is None:
            raise RuntimeError("Embedding layer not available. Cannot compute input embeddings.")

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
                    input_ids, full_weight, padding_idx=getattr(self.embed_layer, "padding_idx", None)
                )
            else:
                # Regular embedding layer (not sharded)
                input_embeds = self.embed_layer(input_ids)

        activation_vectors = activation_vectors.to(input_embeds.device)
        batch_size = activation_vectors.shape[0]
        injection_positions = torch.ones(batch_size, dtype=torch.long, device=activation_vectors.device)

        # Check for injection token in the config
        injection_token_id = None
        if hasattr(self, "injection_cfg") and self.injection_cfg.injection_token_id is not None:
            injection_token_id = self.injection_cfg.injection_token_id
        elif hasattr(self, "nla_wrapper") and self.nla_wrapper.injection_config.injection_token_id is not None:
            injection_token_id = self.nla_wrapper.injection_config.injection_token_id

        if injection_token_id is not None:
            injection_mask = input_ids == injection_token_id
            if injection_mask.any():
                injection_positions = injection_mask.float().argmax(dim=1)
                log.debug(f"NLA Actor: Found injection tokens at positions: {injection_positions}")

        hidden_dim = input_embeds.shape[-1]

        # Validate activation vector dimensions match model hidden size
        if activation_vectors.shape[-1] != hidden_dim:
            raise ValueError(
                f"Activation vector dimension mismatch: activation_vectors have shape {activation_vectors.shape} "
                f"(dimension {activation_vectors.shape[-1]}), but model hidden_dim is {hidden_dim}. "
                f"activation_dim must equal model hidden_size. Check your config and dataset."
            )

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
                log.debug(
                    f"NLA Actor: Injected activation at position {pos} for {len(idxs)} sequence(s) at token # input_ids[{idxs}, {pos}] = {token}, counting left-padding."
                )

        attention_mask = data.batch["attention_mask"]

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
        print(f"[NLA ACTOR WORKER] generate_sequences ENTRY (rank={self.rank})")
        """
        Generate sequences with activation injection.

        For SGLang: Prepares activation vectors and injection positions
        to be used as custom input embeddings.
        """
        log.debug("=" * 80)
        log.debug("NLA ACTOR: generate_sequences() called")
        log.debug("=" * 80)
        log.debug(f"NLA ACTOR: data.batch type: {type(data.batch)}")
        log.debug(f"NLA ACTOR: data.batch keys: {list(data.batch.keys()) if data.batch is not None else 'None'}")
        log.debug(
            f"NLA ACTOR: data.meta_info keys: {list(data.meta_info.keys()) if data.meta_info is not None else 'None'}"
        )

        activation_vectors = self._extract_activation_vectors(data)
        log.debug(
            f"NLA ACTOR: Extracted activation_vectors: {activation_vectors.shape if activation_vectors is not None else None}"
        )

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
        input_ids = data.batch["input_ids"]
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        log.debug(
            f"NLA Actor: Prepared input_embeds for SGLang with shape: {input_embeds[0].shape}, second item shape: {input_embeds[1].shape if len(input_embeds) > 1 else 'None'}"
        )

        return super().generate_sequences(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_input_embeddings(self, data: DataProto) -> DataProto:
        """Compute and attach input embeddings without triggering generation."""
        activation_vectors = self._extract_activation_vectors(data)
        if activation_vectors is None:
            return data

        input_ids = data.batch["input_ids"]
        input_embeds = self._prepare_input_embeddings(data, input_ids, activation_vectors)
        data.non_tensor_batch.update({"input_embeds": input_embeds})
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        """Compute log probabilities with activation injection.

        Ensures activation vectors are in batch, then delegates to parent.
        Parent calls self.actor.compute_log_prob (which we wrapped!).
        """
        log.debug(f"\n{'#' * 80}")
        log.debug("🎯 NLA ACTOR WORKER: compute_log_prob() CALLED")
        log.debug(f"{'#' * 80}")

        # Extract activation vectors and ensure they're in batch
        activation_vectors = self._extract_activation_vectors(data)

        if activation_vectors is not None:
            if "activation_vectors" not in data.batch.keys():
                data.batch["activation_vectors"] = activation_vectors
            log.debug(
                f"✓ NLA ACTOR WORKER: compute_log_prob with activation vectors of shape: {activation_vectors.shape}"
            )
            self._print_injection_telemetry("BEFORE compute_log_prob")
        else:
            log.debug("⚠️  NLA ACTOR WORKER: No activation vectors found in data")

        # Call parent - it will delegate to self.actor.compute_log_prob (which is wrapped!)
        # The wrapped actor.compute_log_prob will add activation_vectors to select_keys
        log.debug("→ Delegating to parent compute_log_prob (will call wrapped actor.compute_log_prob)")
        log.debug(f"{'#' * 80}\n")

        result = super().compute_log_prob(data)

        if activation_vectors is not None:
            self._print_injection_telemetry("AFTER compute_log_prob")

        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def update_policy(self, data: DataProto):
        """Update policy with activation injection.

        Ensures activation vectors are in batch, then delegates to parent.
        Parent calls self.actor.update_policy (which we wrapped!).
        """
        log.debug(f"\n{'#' * 80}")
        log.debug("🎯 NLA ACTOR WORKER: update_policy() CALLED")
        log.debug(f"{'#' * 80}")

        # Extract activation vectors and ensure they're in batch
        activation_vectors = self._extract_activation_vectors(data)

        if activation_vectors is not None:
            if "activation_vectors" not in data.batch.keys():
                data.batch["activation_vectors"] = activation_vectors
            log.debug(f"✓ NLA ACTOR WORKER: update_policy with activation vectors of shape: {activation_vectors.shape}")
            self._print_injection_telemetry("BEFORE update_policy")
        else:
            log.debug("⚠️  NLA ACTOR WORKER: No activation vectors found in data")

        # Call parent - it will delegate to self.actor.update_policy (which is wrapped!)
        # The wrapped actor.update_policy will add activation_vectors to select_keys
        log.debug("→ Delegating to parent update_policy (will call wrapped actor.update_policy)")
        log.debug(f"{'#' * 80}\n")

        result = super().update_policy(data)

        if activation_vectors is not None:
            self._print_injection_telemetry("AFTER update_policy")

        return result

    def _print_injection_telemetry(self, label: str = ""):
        """Print injection telemetry for debugging."""
        if hasattr(self, "_injection_telemetry"):
            log.debug(f"\n{'=' * 60}")
            log.debug(f"📊 INJECTION TELEMETRY {label}")
            log.debug(f"{'=' * 60}")
            for key, value in self._injection_telemetry.items():
                log.debug(f"  {key}: {value}")
            log.debug(f"{'=' * 60}\n")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def verify_injection_setup(self):
        """Verify that injection hooks and wrappers are properly installed.

        This method can be called after init to verify everything is set up correctly.
        """
        log.debug(f"\n{'=' * 80}")
        log.debug("🔍 NLA INJECTION SETUP VERIFICATION")
        log.debug(f"{'=' * 80}")

        checks = []

        # Check 1: Embedding layer
        if hasattr(self, "embed_layer") and self.embed_layer is not None:
            checks.append(("✓", "Embedding layer found", str(type(self.embed_layer))))
        else:
            checks.append(("✗", "Embedding layer NOT found", "MISSING"))

        # Check 2: Injection hook
        if hasattr(self, "_injection_hook_handle"):
            checks.append(("✓", "Injection hook registered", "OK"))
        else:
            checks.append(("✗", "Injection hook NOT registered", "MISSING"))

        # Check 3: Actor exists
        if hasattr(self, "actor") and self.actor is not None:
            checks.append(("✓", "Actor exists", self.actor.__class__.__name__))
        else:
            checks.append(("✗", "Actor does NOT exist", "MISSING"))

        # Check 4: Actor has _forward_micro_batch
        if hasattr(self, "actor") and hasattr(self.actor, "_forward_micro_batch"):
            checks.append(("✓", "Actor has _forward_micro_batch", "OK"))

            # Check if it's wrapped (wrapped functions have __name__ = '_forward_micro_batch_with_injection')
            method_name = self.actor._forward_micro_batch.__name__
            if "injection" in method_name:
                checks.append(("✓", "Actor _forward_micro_batch is WRAPPED", method_name))
            else:
                checks.append(("⚠️", "Actor _forward_micro_batch may NOT be wrapped", method_name))
        else:
            checks.append(("✗", "Actor _forward_micro_batch NOT found", "MISSING"))

        # Check 5: Injection config
        if hasattr(self, "injection_cfg"):
            token_id = self.injection_cfg.injection_token_id
            checks.append(("✓", f"Injection config exists (token_id={token_id})", "OK"))
        else:
            checks.append(("✗", "Injection config NOT found", "MISSING"))

        # Check 6: Telemetry
        if hasattr(self, "_injection_telemetry"):
            checks.append(("✓", "Telemetry initialized", "OK"))
        else:
            checks.append(("⚠️", "Telemetry NOT initialized", "WARN"))

        # Check 7: Projection layer
        if hasattr(self, "_activation_projection"):
            if self._activation_projection is not None:
                checks.append(("✓", "Projection layer created", str(self._activation_projection)))
            else:
                checks.append(("ℹ️", "No projection layer (dims match)", "OK"))
        else:
            checks.append(("⚠️", "Projection layer attribute missing", "WARN"))

        # Print all checks
        for status, check, detail in checks:
            log.debug(f"  {status} {check}: {detail}")

        # Overall status
        failed = sum(1 for s, _, _ in checks if s == "✗")
        warnings = sum(1 for s, _, _ in checks if s in ["⚠️", "ℹ️"])
        passed = sum(1 for s, _, _ in checks if s == "✓")

        log.debug(f"\n  Summary: {passed} passed, {warnings} warnings, {failed} failed")

        if failed > 0:
            log.debug("  ❌ INJECTION SETUP INCOMPLETE - INJECTION WILL NOT WORK")
        elif warnings > 0:
            log.debug("  ⚠️  INJECTION SETUP OK (with warnings)")
        else:
            log.debug("  ✅ INJECTION SETUP COMPLETE - READY TO INJECT")

        log.debug(f"{'=' * 80}\n")

        return {"passed": passed, "warnings": warnings, "failed": failed}
