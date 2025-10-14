"""NLA-enabled Megatron SFT Trainer with activation injection support.

This module extends the base SFT trainer to support activation vector injection
during Megatron training with full support for:
- Tensor Parallelism (TP): Model layers sharded across GPUs
- Pipeline Parallelism (PP): Model stages distributed across GPUs
- Data Parallelism (DP): Batches distributed across GPU groups

ARCHITECTURE OVERVIEW - Activation Injection for SFT Training
==============================================================

This trainer implements the same activation injection mechanism as the
NLAMegatronActorRolloutRefWorker but adapted for SFT training:

1. EMBEDDING-LAYER HOOK APPROACH
   - Forward hook registered on embedding layer (PP stage 0 only)
   - Hook modifies embeddings during forward pass
   - Integrates with Megatron's forward_backward_func scheduler

2. MICRO-BATCH TRACKING (CRITICAL FOR PP)
   - Megatron splits global batches into micro-batches
   - Hook fires ONCE PER MICRO-BATCH
   - We track which micro-batch via _current_micro_batch_offset
   - Each hook uses correct slice of activation vectors

3. STATE MANAGEMENT FLOW
   - Before each batch:
     * Set injection state with all activation vectors
     * Reset offset to 0
   - During forward pass:
     * Hook fires for each micro-batch
     * Uses activations[offset:offset+micro_batch_size]
     * Increments offset after each micro-batch
   - After batch completes:
     * Clear injection state

4. BACKWARD COMPATIBILITY
   - If no NLA config present: standard SFT training
   - If NLA config present: activation injection enabled
   - All changes guarded by config checks

VALIDATION CHECKLIST
--------------------
- Hook only registers on mpu.is_pipeline_first_stage()
- Hook uses _current_micro_batch_offset to index vectors
- Offset resets in _set_injection_state
- Offset increments after each hook invocation
- State clears after training batch completes
"""

import logging
import os

import torch
import torch.distributed
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer, run_sft

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

log = logging.getLogger(__name__)
log.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class NLAMegatronSFTTrainer(SFTTrainer):
    """
    NLA-enabled Megatron SFT Trainer with activation injection support.

    Extends base SFTTrainer with activation injection capabilities for Megatron models.
    Fully backward compatible - works as standard SFT trainer when NLA config is absent.

    Key features:
    - Activation injection via embedding layer hooks (like NLAMegatronActorRolloutRefWorker)
    - Full support for TP/PP/DP parallelism
    - Micro-batch tracking for pipeline parallelism
    - Automatic injection token management
    - Projection layer for dimension mismatches
    """

    def __init__(self, config):
        """Initialize NLA Megatron SFT trainer.

        Args:
            config: Hydra config with optional 'nla' section containing:
                - injection.mode: "replace" or "add" (default: "replace")
                - injection.layer_indices: layers to inject at (default: [0])
                - injection.projection_dim: activation dim if != hidden_dim
                - injection.injection_token: token for injection position (default: None)
                - activation_dim: dimension of activation vectors
        """
        # Check if NLA config exists
        self.nla_enabled = hasattr(config, "nla") and config.get("nla", {}).get("enabled", True)

        if self.nla_enabled:
            log.info("=" * 80)
            log.info("NLA MEGATRON SFT TRAINER: NLA injection enabled")
            log.info("=" * 80)

        # Call parent constructor
        super().__init__(config)

        # Setup NLA components after parent initialization
        if self.nla_enabled:
            self._setup_nla_injection()

    def _build_dataloader(self):
        """Override to use NLA collator that handles variable-length responses."""
        if not self.nla_enabled:
            # Use parent's dataloader construction
            return super()._build_dataloader()

        # Import NLA collator
        from verl.nla.data.nla_sft_dataset import NLASFTCollator

        # Get tokenizer for pad_token_id
        tokenizer = self.model_config.tokenizer if hasattr(self.model_config, "tokenizer") else None
        pad_token_id = tokenizer.pad_token_id if tokenizer and hasattr(tokenizer, "pad_token_id") else 0

        # Create collator
        collate_fn = NLASFTCollator(pad_token_id=pad_token_id)

        # Build dataloader with custom collator
        from torch.utils.data import DistributedSampler
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.utils.device import get_device_name

        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = self.config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
            collate_fn=collate_fn,  # Use NLA collator
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
            collate_fn=collate_fn,  # Use NLA collator
        )

        log.info("NLA MEGATRON SFT: Using NLASFTCollator for DataLoader")

    def _setup_nla_injection(self):
        """Setup NLA injection components after model initialization."""
        log.info("NLA MEGATRON SFT: Setting up activation injection")

        # Only setup on first pipeline stage (where embeddings exist)
        if not mpu.is_pipeline_first_stage():
            log.info("NLA MEGATRON SFT: Not on first pipeline stage, skipping injection setup")
            return

        # Get NLA configuration
        nla_config = self.config.get("nla", {})
        injection_config_dict = nla_config.get("injection", {})

        # Import injection config
        from verl.nla.models.nla_wrapper import InjectionConfig
        from verl.nla.utils.injection_manager import InjectionTokenManager

        # Configure injection settings
        self.injection_cfg = InjectionConfig(
            mode=injection_config_dict.get("mode", "replace"),
            layer_indices=injection_config_dict.get("layer_indices", [0]),
            projection_dim=injection_config_dict.get("projection_dim", None),
            injection_token=injection_config_dict.get("injection_token", None),
        )

        # Setup tokenizer and injection token manager
        tokenizer = self.model_config.tokenizer if hasattr(self.model_config, "tokenizer") else None
        if tokenizer is not None:
            injection_manager = InjectionTokenManager(tokenizer, self.injection_cfg.injection_token)
            self.injection_cfg.injection_token_id = injection_manager.token_id
            self.injection_cfg.injection_character = injection_manager.character
            self.injection_manager = injection_manager
            log.info(
                f"NLA MEGATRON SFT: Injection token: '{injection_manager.character}' (ID: {injection_manager.token_id})"
            )
        elif self.injection_cfg.injection_token_id is not None and self.injection_cfg.injection_token_id >= 0:
            self.injection_manager = None
        else:
            raise ValueError(
                "Either provide a tokenizer for auto-selection or specify injection_token_id in config."
            )

        # Store reference to embedding layer
        self._store_embedding_layer()

        if self.embed_layer is None:
            log.warning("NLA MEGATRON SFT: Could not find embedding layer, injection disabled")
            self.nla_enabled = False
            return

        # Determine dimensions for projection layer
        hidden_dim = None
        activation_dim = nla_config.get("activation_dim", None)

        # Try to get hidden_dim from engine's model
        if hasattr(self.engine, "module") and len(self.engine.module) > 0:
            model = self.engine.module[0]
            if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
                hidden_dim = model.config.hidden_size

        if activation_dim is None:
            activation_dim = hidden_dim

        # Create projection layer if dimensions mismatch
        if activation_dim is not None and hidden_dim is not None and activation_dim != hidden_dim:
            log.info(f"NLA MEGATRON SFT: Creating projection layer: {activation_dim} -> {hidden_dim}")
            self._activation_projection = torch.nn.Linear(activation_dim, hidden_dim, bias=False)
            # Move to same device as embedding layer
            device = self.embed_layer.weight.device if hasattr(self.embed_layer, "weight") else None
            if device is not None:
                self._activation_projection = self._activation_projection.to(device)
        else:
            self._activation_projection = None

        # Register forward hook on embedding layer
        log.info("NLA MEGATRON SFT: Registering embedding injection hook")
        self._injection_hook_handle = self.embed_layer.register_forward_hook(self._embedding_injection_hook)

        # Initialize state tracking
        self._current_activation_vectors = None
        self._current_injection_positions = None
        self._current_micro_batch_offset = 0

        # Initialize telemetry
        self._injection_telemetry = {
            "train_batches": 0,
            "batches_with_activations": 0,
            "hook_calls": 0,
            "actual_injections": 0,
        }

        log.info("NLA MEGATRON SFT: Activation injection setup complete")

    def _store_embedding_layer(self):
        """Store reference to Megatron model's embedding layer (first pipeline stage only)."""
        self.embed_layer = None

        # Only first pipeline stage has the embedding layer
        if not mpu.is_pipeline_first_stage():
            return

        # Megatron models are stored in engine.module (list)
        if hasattr(self.engine, "module") and len(self.engine.module) > 0:
            model = self.engine.module[0]

            # Try Megatron-specific patterns first
            if hasattr(model, "embedding") and hasattr(model.embedding, "word_embeddings"):
                self.embed_layer = model.embedding.word_embeddings
                log.info(f"NLA MEGATRON SFT: Found Megatron embedding layer: {type(self.embed_layer)}")
            # Fallback to HF patterns
            elif hasattr(model, "get_input_embeddings"):
                self.embed_layer = model.get_input_embeddings()
                log.info("NLA MEGATRON SFT: Found embedding layer via get_input_embeddings")
            elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                self.embed_layer = model.model.embed_tokens
                log.info("NLA MEGATRON SFT: Found embedding layer via model.embed_tokens")
            else:
                log.warning("NLA MEGATRON SFT: Could not find embedding layer in model")
        else:
            log.warning("NLA MEGATRON SFT: engine.module not found or empty")

    def _embedding_injection_hook(self, module, input, output):
        """Forward hook that injects activation vectors into embeddings (first pipeline stage only).

        CRITICAL: This hook fires once per micro-batch during Megatron's pipeline execution.
        We must track which micro-batch we're processing to use the correct activation vectors.

        Args:
            module: The embedding module
            input: Input tuple to the embedding layer
            output: Output embeddings tensor [micro_batch_size, seq_len, hidden_dim]

        Returns:
            Modified output with activation vectors injected
        """
        # This should only fire on first pipeline stage
        if not mpu.is_pipeline_first_stage():
            return output

        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["hook_calls"] += 1

        # Check if we have activation vectors to inject
        if not hasattr(self, "_current_activation_vectors") or self._current_activation_vectors is None:
            return output

        activation_vectors = self._current_activation_vectors
        injection_positions = self._current_injection_positions

        # Get current micro-batch offset to index into the global activation vectors
        micro_batch_offset = getattr(self, "_current_micro_batch_offset", 0)
        micro_batch_size = output.shape[0]

        log.debug(
            f"🔥 NLA MEGATRON SFT HOOK: offset={micro_batch_offset}, "
            f"micro_batch_size={micro_batch_size}, "
            f"total_activations={activation_vectors.shape[0]}"
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
            f"✓ NLA MEGATRON SFT HOOK: Performed {injections_performed}/{micro_batch_size} injections "
            f"for global indices [{micro_batch_offset}:{micro_batch_offset + micro_batch_size}]"
        )

        if hasattr(self, "_injection_telemetry"):
            self._injection_telemetry["actual_injections"] += injections_performed

        # Increment offset for next micro-batch
        self._current_micro_batch_offset = micro_batch_offset + micro_batch_size

        return output

    def _set_injection_state(self, activation_vectors: torch.Tensor, injection_positions: list):
        """Set activation vectors and positions for the next forward pass.

        This is called once per global batch. The hook will fire multiple times
        (once per micro-batch) and must track its progress.

        Args:
            activation_vectors: Tensor of shape [global_batch_size, activation_dim]
            injection_positions: List of injection positions, one per sample
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
        """Find positions of injection tokens in input_ids.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len]

        Returns:
            List of positions (one per sample). Each position is either:
            - A tensor with position index(es)
            - None if no injection token found
        """
        positions = []
        injection_token_id = self.injection_cfg.injection_token_id

        for i in range(input_ids.shape[0]):
            batch_positions = (input_ids[i] == injection_token_id).nonzero(as_tuple=False).squeeze(-1)
            positions.append(batch_positions if batch_positions.numel() > 0 else None)

        return positions

    def fit(self):
        """Override fit to wrap training loop with injection state management."""
        if not self.nla_enabled:
            # Standard SFT training
            return super().fit()

        # NLA-enabled training with injection
        # We need to wrap the engine's train_batch method to manage injection state
        if mpu.is_pipeline_first_stage():
            self._wrap_engine_train_batch()

        # Call parent fit
        result = super().fit()

        # Log telemetry on rank 0
        if torch.distributed.get_rank() == 0 and hasattr(self, "_injection_telemetry"):
            log.info("=" * 80)
            log.info("NLA MEGATRON SFT: Training telemetry")
            log.info(f"  Train batches processed: {self._injection_telemetry['train_batches']}")
            log.info(f"  Batches with activations: {self._injection_telemetry['batches_with_activations']}")
            log.info(f"  Hook invocations (micro-batches): {self._injection_telemetry['hook_calls']}")
            log.info(f"  Actual injections performed: {self._injection_telemetry['actual_injections']}")
            log.info("=" * 80)

        return result

    def _wrap_engine_train_batch(self):
        """Wrap engine's train_batch method to manage injection state."""
        if not hasattr(self.engine, "train_batch"):
            log.warning("NLA MEGATRON SFT: engine.train_batch not found, cannot wrap")
            return

        original_train_batch = self.engine.train_batch

        def wrapped_train_batch(data, loss_function=None, **kwargs):
            """Wrapped train_batch that manages injection state."""
            # Track calls
            if hasattr(self, "_injection_telemetry"):
                self._injection_telemetry["train_batches"] += 1

            # Check if batch has activation vectors
            has_activations = hasattr(data, "batch") and "activation_vectors" in data.batch
            should_inject = has_activations and mpu.is_pipeline_first_stage()

            if should_inject:
                if hasattr(self, "_injection_telemetry"):
                    self._injection_telemetry["batches_with_activations"] += 1

                activation_vectors = data.batch["activation_vectors"]
                input_ids = data.batch["input_ids"]
                injection_positions = self._find_injection_positions_from_ids(input_ids)

                log.debug("🎯 NLA MEGATRON SFT: Setting injection state for train_batch")
                log.debug(f"   Activations shape: {activation_vectors.shape}")
                log.debug(
                    f"   Injection positions: {[(p.item() if isinstance(p, torch.Tensor) and p.numel() > 0 else p) for p in injection_positions]}"
                )

                self._set_injection_state(activation_vectors, injection_positions)

            try:
                return original_train_batch(data, loss_function, **kwargs)
            finally:
                if should_inject:
                    self._clear_injection_state()
                    log.debug("✓ NLA MEGATRON SFT: Cleared injection state after train_batch")

        # Replace engine's train_batch method
        self.engine.train_batch = wrapped_train_batch
        log.info("NLA MEGATRON SFT: Wrapped engine.train_batch for injection state management")


def run_nla_megatron_sft(config):
    """Run NLA-enabled Megatron SFT training.

    This is a drop-in replacement for run_sft() that supports NLA activation injection.
    If NLA config is not present, behaves identically to standard SFT training.

    Args:
        config: Hydra config with optional 'nla' section
    """
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group()
    trainer = NLAMegatronSFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


# For backward compatibility, also export as main entry point
if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
    def main(config):
        run_nla_megatron_sft(config)

    main()

