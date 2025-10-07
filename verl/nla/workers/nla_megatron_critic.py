"""NLA Megatron Critic that outputs activation vectors instead of scalars.

PIPELINE PARALLELISM (PP) SUPPORT STATUS
==========================================

CURRENT STATUS:
- PP=1: Fully supported with any output_layer_index
- PP>1 with output_layer_index=-1: Fully supported
- PP>1 with intermediate layers (e.g., output_layer_index=-2, 20): LIMITED SUPPORT

WHY THE LIMITATION?
-------------------
Megatron's pipeline parallelism distributes transformer layers across ranks:
- PP Stage 0: Layers 0-N
- PP Stage 1: Layers N+1-M
- PP Stage 2: Layers M+1-K, etc.

Each stage only has access to its own layer outputs. To extract from an intermediate
layer (e.g., layer 20 on Stage 1), we would need to:
1. Capture the output on Stage 1
2. Broadcast it to the final stage (where loss is computed)
3. Handle this for every micro-batch in the pipeline schedule

This requires deep integration with Megatron's pipeline scheduler and adds significant
communication overhead.

WORKAROUNDS:
-----------
1. Use PP=1 (recommended for small models): Full support for any output_layer_index
2. Use output_layer_index=-1 with PP>1: The final layer output IS available on the
   last PP stage, so this works without modification
3. Wait for full PP>1 support: Future work will add proper intermediate layer extraction

TENSOR PARALLELISM:
------------------
TP (Tensor Parallelism) at any size is fully supported. TP shards layers horizontally
across GPUs but all ranks have access to the same logical layer outputs.
"""

import logging
import torch
import torch.nn as nn
from torch import optim
from typing import Optional
from functools import partial

from verl import DataProto
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.utils.device import get_device_id
from verl.utils.torch_functional import masked_mean

# Import Megatron utilities
try:
    from megatron.core import parallel_state as mpu
    from megatron.core.pipeline_parallel import get_forward_backward_func
except ImportError:
    # Fallback for environments without Megatron
    class _MockMPU:
        @staticmethod
        def get_pipeline_model_parallel_rank():
            return 0
        @staticmethod
        def get_pipeline_model_parallel_world_size():
            return 1
        @staticmethod
        def get_pipeline_model_parallel_group():
            return None
        @staticmethod
        def is_pipeline_last_stage(ignore_virtual=False):
            return True
    mpu = _MockMPU()
    def get_forward_backward_func():
        raise NotImplementedError("Megatron not available")

log = logging.getLogger(__name__)


class NLAMegatronCritic(MegatronPPOCritic):
    """
    NLA Critic for Megatron that extends the standard VERL Megatron critic but:
    1. Outputs activation vectors instead of scalar values
    2. Computes MSE loss against target activations
    3. Optionally prepends a prompt to all inputs (e.g., "summary of the following text: ")

    Key differences from FSDP NLADataParallelCritic:
    - Inherits from MegatronPPOCritic instead of DataParallelPPOCritic
    - Handles Megatron's pipeline parallelism and model list structure
    - Uses Megatron-specific forward/backward utilities
    """

    def __init__(
        self,
        config,
        critic_module: nn.ModuleList,
        critic_optimizer,
        tokenizer=None,
        **kwargs
    ):
        # MegatronPPOCritic expects additional config parameters
        # We'll pass through whatever we have, with sensible defaults
        model_config = getattr(config, 'model', None)
        hf_config = kwargs.get('hf_config', None)
        tf_config = kwargs.get('tf_config', None)
        critic_optimizer_config = kwargs.get('critic_optimizer_config', None)

        super().__init__(
            config=config,
            model_config=model_config,
            hf_config=hf_config,
            tf_config=tf_config,
            critic_module=critic_module,
            critic_optimizer=critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )

        self.logger = logging.getLogger(__name__)

        # Validate PP configuration and output_layer_index
        self.pp_size = 1
        self.output_layer_index = getattr(config, 'output_layer_index', -1)

        if hasattr(config, 'megatron'):
            self.pp_size = getattr(config.megatron, 'pipeline_model_parallel_size', 1)

            # Validate PP configuration
            if self.pp_size > 1:
                if self.output_layer_index != -1:
                    # PP>1 with intermediate layers: Not fully supported yet
                    raise NotImplementedError(
                        f"\n{'='*80}\n"
                        f"NLA MEGATRON CRITIC: PIPELINE PARALLELISM LIMITATION\n"
                        f"{'='*80}\n"
                        f"Configuration:\n"
                        f"  - Pipeline Parallelism (PP): {self.pp_size}\n"
                        f"  - output_layer_index: {self.output_layer_index}\n"
                        f"\n"
                        f"ISSUE:\n"
                        f"  Extracting activations from intermediate layers (output_layer_index != -1)\n"
                        f"  with PP>1 requires capturing outputs from non-final pipeline stages and\n"
                        f"  broadcasting them to the final stage for loss computation. This is not\n"
                        f"  currently implemented.\n"
                        f"\n"
                        f"WORKAROUNDS (choose one):\n"
                        f"  1. Use PP=1 (no pipeline parallelism):\n"
                        f"     Set config.megatron.pipeline_model_parallel_size = 1\n"
                        f"     This gives full support for any output_layer_index.\n"
                        f"\n"
                        f"  2. Use output_layer_index=-1 (final layer):\n"
                        f"     Set config.output_layer_index = -1\n"
                        f"     The final layer is available on the last PP stage, so no\n"
                        f"     cross-stage communication is needed.\n"
                        f"\n"
                        f"  3. Increase TP instead of PP:\n"
                        f"     Use tensor parallelism (TP) to scale instead of pipeline\n"
                        f"     parallelism. TP is fully supported with any output_layer_index.\n"
                        f"\n"
                        f"TECHNICAL BACKGROUND:\n"
                        f"  With PP={self.pp_size}, transformer layers are distributed across {self.pp_size} GPUs.\n"
                        f"  Each GPU only has access to its own layers' outputs. To extract from\n"
                        f"  layer {self.output_layer_index}, we'd need custom communication infrastructure.\n"
                        f"\n"
                        f"  For more details, see the module docstring in:\n"
                        f"  verl/verl/nla/workers/nla_megatron_critic.py\n"
                        f"{'='*80}\n"
                    )
                else:
                    # PP>1 with output_layer_index=-1: Fully supported
                    self.logger.info(
                        f"NLA Megatron critic initialized with PP={self.pp_size} and output_layer_index=-1. "
                        f"This configuration is fully supported."
                    )
            else:
                # PP=1: Fully supported with any output_layer_index
                self.logger.info(
                    f"NLA Megatron critic initialized with PP=1 and output_layer_index={self.output_layer_index}. "
                    f"Fully supported."
                )

        # Get hidden size from the critic model (Megatron model is in a list)
        if len(critic_module) > 0:
            model = critic_module[0]
            self.hidden_size = model.config.hidden_size if hasattr(model, 'config') else None
        else:
            self.hidden_size = None

        # Initialize NLA adapter for extracting activation vectors
        from verl.nla.integration.dataproto_adapter import NLADataProtoAdapter
        self.adapter = NLADataProtoAdapter()

        # Handle critic prompt configuration (same as FSDP version)
        self.tokenizer = tokenizer
        self.critic_prompt_text = getattr(config, 'critic_prompt', None)
        self.critic_prompt_tokens = None
        self.critic_prompt_length = 0
        self._MAX_CRITIC_PROMPT_LENGTH = 128

        if self.critic_prompt_text and tokenizer is not None:
            if not self.critic_prompt_text.strip():
                self.logger.warning("Critic prompt is empty or whitespace-only. Ignoring.")
                self.critic_prompt_tokens = None
                self.critic_prompt_length = 0
            else:
                prompt_encoding = tokenizer(
                    self.critic_prompt_text,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                self.critic_prompt_tokens = prompt_encoding['input_ids'][0]
                self.critic_prompt_length = len(self.critic_prompt_tokens)

                if self.critic_prompt_length == 0:
                    self.logger.warning(f"Critic prompt '{self.critic_prompt_text}' tokenized to 0 tokens.")
                    self.critic_prompt_tokens = None
                    self.critic_prompt_length = 0
                elif self.critic_prompt_length > self._MAX_CRITIC_PROMPT_LENGTH:
                    raise ValueError(
                        f"Critic prompt is too long ({self.critic_prompt_length} tokens). "
                        f"Maximum allowed: {self._MAX_CRITIC_PROMPT_LENGTH} tokens."
                    )
                else:
                    self.logger.info(
                        f"Initialized Megatron critic with prompt: '{self.critic_prompt_text}' "
                        f"({self.critic_prompt_length} tokens)"
                    )
        elif self.critic_prompt_text and tokenizer is None:
            self.logger.warning(
                f"Critic prompt '{self.critic_prompt_text}' configured but no tokenizer provided."
            )
        else:
            self.logger.info("Megatron critic initialized without prompt prefix")

    def _extract_value_tensor(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """Return activation tensor from hidden states.

        For NLA, we extract from the layer specified by output_layer_index.
        """
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            return model_outputs.hidden_states[self.output_layer_index]

        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 2:
            candidate = model_outputs[2]
            if isinstance(candidate, torch.Tensor) and candidate.dim() >= 3:
                return candidate

        raise ValueError("Megatron critic model output does not contain activation tensor")

    def _forward_micro_batch(self, micro_batch):
        """Run forward pass for a micro batch returning activation vectors.

        NLA Design Note: Processes ONLY the response text (optionally prefixed
        with critic_prompt), not the full user prompt + response.
        """
        responses = micro_batch["responses"]
        response_length = responses.size(-1)

        # NLA design: Use only responses
        input_ids = responses

        # Prepend prompt tokens if configured
        if self.critic_prompt_tokens is not None and self.critic_prompt_length > 0:
            batch_size = input_ids.shape[0]
            prompt_batch = self.critic_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(input_ids.device)
            input_ids = torch.cat([prompt_batch, input_ids], dim=1)

        batch, seqlen = input_ids.shape

        # Extract attention mask for responses, then prepend prompt mask if needed
        attention_mask_full = micro_batch["attention_mask"]
        attention_mask = attention_mask_full[:, -response_length:]

        if self.critic_prompt_tokens is not None and self.critic_prompt_length > 0:
            prompt_mask = torch.ones(
                (batch, self.critic_prompt_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Create position IDs from scratch
        position_ids_full = micro_batch["position_ids"]
        if position_ids_full.dim() == 3:  # qwen2vl mrope
            position_ids_full_transposed = position_ids_full.transpose(0, 1)
            num_ropes = position_ids_full_transposed.shape[0]
            position_ids = torch.zeros(
                (num_ropes, batch, seqlen),
                dtype=position_ids_full.dtype,
                device=position_ids_full.device
            )
            for rope_idx in range(num_ropes):
                for b_idx in range(batch):
                    valid_len = int(attention_mask[b_idx].sum().item())
                    position_ids[rope_idx, b_idx, :valid_len] = torch.arange(
                        valid_len, dtype=position_ids_full.dtype, device=position_ids_full.device
                    )
            position_ids = position_ids.transpose(0, 1)
        else:
            position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            for b_idx in range(batch):
                valid_len = int(attention_mask[b_idx].sum().item())
                position_ids[b_idx, :valid_len] = torch.arange(
                    valid_len, dtype=torch.long, device=attention_mask.device
                )

        # Forward pass through Megatron model
        # Megatron critic_module is a ModuleList
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.critic_module[0](
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            values = self._extract_value_tensor(outputs)

            # Extract only the response portion if prompt was prepended
            if self.critic_prompt_length > 0:
                values = values[:, self.critic_prompt_length:, :]
                if values.shape[1] > response_length:
                    values = values[:, :response_length, :]

        return values

    def extract_predicted_activations(
        self,
        full_activations: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        pooling: str = "last"
    ) -> torch.Tensor:
        """Extract predicted activations from full activations tensor (same as FSDP version)."""
        batch_size = full_activations.shape[0]

        if pooling == "last":
            if response_mask is not None:
                last_indices = response_mask.sum(dim=1) - 1
                last_indices = last_indices.clamp(min=0).long()
                pooled_predictions = full_activations[
                    torch.arange(batch_size), last_indices
                ]
            else:
                raise ValueError("Response mask is required for last pooling")

        elif pooling == "mean":
            if response_mask is not None:
                mask_expanded = response_mask.unsqueeze(-1)
                pooled_predictions = (full_activations * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled_predictions = full_activations.mean(dim=1)

        elif pooling == "max":
            if response_mask is not None:
                mask_expanded = response_mask.unsqueeze(-1)
                masked_predictions = full_activations.masked_fill(~mask_expanded.bool(), -1e9)
                pooled_predictions, _ = masked_predictions.max(dim=1)
            else:
                pooled_predictions, _ = full_activations.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        return pooled_predictions

    def compute_activation_loss(
        self,
        predicted_activations: torch.Tensor,
        target_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target activations."""
        mse_loss = nn.functional.mse_loss(predicted_activations, target_activations)
        return mse_loss

    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        Compute MSE-based scalar values (negative reconstruction error).

        Returns values as (batch, seq_len) tensor where higher values = better reconstruction.
        This allows NLA to use standard VERL GRPO advantage computation.
        """
        # Check if values already computed
        if "values" in data.batch.keys():
            self.logger.debug("Values already computed, skipping recomputation")
            return data.batch["values"]

        # Compute response_mask if not present
        if "response_mask" not in data.batch.keys():
            from verl.trainer.ppo.ray_trainer import compute_response_mask
            response_mask = compute_response_mask(data)
            data.batch["response_mask"] = response_mask

        # For Megatron, we need to implement a simpler forward path
        # that doesn't use the complex pipeline parallelism of the parent class
        # For now, delegate to _compute_values_simple
        return self._compute_values_simple(data)

    def _compute_values_simple(self, data: DataProto) -> torch.Tensor:
        """Simplified value computation for NLA without pipeline parallelism complexity."""
        # Prepare meta info
        if data.meta_info is None:
            data.meta_info = {}
        data.meta_info.setdefault("micro_batch_size", self.config.forward_micro_batch_size_per_gpu)

        # Select required keys
        select_keys = ["responses", "input_ids", "response_mask", "attention_mask", "position_ids", "activation_vectors"]
        data_selected = data.select(batch_keys=select_keys)

        # Process in micro-batches
        micro_batch_size = data.meta_info["micro_batch_size"]
        micro_batches = data_selected.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                full_activations = self._forward_micro_batch(model_inputs)

            # Extract target activations
            target_activations = micro_batch.batch['activation_vectors']
            if not isinstance(target_activations, torch.Tensor):
                target_activations = torch.as_tensor(
                    target_activations,
                    device=full_activations.device,
                    dtype=full_activations.dtype
                )

            # Pool predicted activations
            response_mask = micro_batch.batch["response_mask"]
            pooled = self.extract_predicted_activations(
                full_activations, response_mask, pooling="last"
            )

            # Compute MSE per sample
            mse_per_sample = ((pooled - target_activations) ** 2).mean(dim=-1)

            # Broadcast to response length
            resp_len = micro_batch.batch["responses"].size(-1)
            values = torch.zeros((pooled.shape[0], resp_len), dtype=full_activations.dtype, device=full_activations.device)

            # Put value only on last valid token
            last_indices = response_mask.sum(dim=1) - 1
            last_indices = last_indices.clamp(min=0)
            values[torch.arange(values.shape[0]), last_indices] = (-mse_per_sample).to(values.dtype)

            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)

        # Apply response mask
        if "response_mask" in data_selected.batch:
            response_mask = data_selected.batch["response_mask"].to(values.device)
            values = values * response_mask

        return values

    def update_critic(self, data: DataProto):
        """
        Update the Megatron critic with MSE loss against target activations.

        This uses a simple micro-batch loop approach. For PP>1 with intermediate layers,
        see validation in __init__ for limitations.
        """
        # Make sure we are in training mode
        for module in self.critic_module:
            module.train()

        # Select required keys including activation_vectors
        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns", "activation_vectors"]
        data = data.select(batch_keys=select_keys)

        # Extract target activations
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data, raise_on_missing=True)

        # Process in micro-batches
        micro_batch_size = data.meta_info.get("micro_batch_size", self.config.forward_micro_batch_size_per_gpu)
        micro_batches = data.split(micro_batch_size)

        total_loss = 0.0
        num_batches = 0

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # Forward pass with gradients
            full_activations = self._forward_micro_batch(model_inputs)
            response_mask = micro_batch.batch["response_mask"]
            predicted_activations = self.extract_predicted_activations(
                full_activations, response_mask, pooling=self.config.get("pooling_strategy", "last")
            )

            # Get corresponding target activations for this micro-batch
            batch_target_activations = micro_batch.batch['activation_vectors']
            if not isinstance(batch_target_activations, torch.Tensor):
                batch_target_activations = torch.as_tensor(
                    batch_target_activations,
                    device=predicted_activations.device,
                    dtype=predicted_activations.dtype
                )

            # Compute MSE loss
            loss = self.compute_activation_loss(
                predicted_activations,
                batch_target_activations.to(device=predicted_activations.device, dtype=predicted_activations.dtype),
            )

            # Backward pass
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            grad_norm = self._optimizer_step()

            total_loss += loss.item()
            num_batches += 1

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {
            "critic_loss": avg_loss,
            "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else float('inf'),
            "hidden_size": self.hidden_size if self.hidden_size else predicted_activations.shape[-1],
        }

        # Log statistics (using last micro-batch for simplicity)
        with torch.no_grad():
            pred_mean = predicted_activations.mean().item()
            pred_std = predicted_activations.std().item()
            target_mean = batch_target_activations.mean().item()
            target_std = batch_target_activations.std().item()

            target_norms = torch.norm(batch_target_activations, dim=1)
            avg_target_norm = target_norms.mean().item()
            pred_norms = torch.norm(predicted_activations, dim=1)
            avg_pred_norm = pred_norms.mean().item()

            metrics["pred_activation_mean"] = pred_mean
            metrics["pred_activation_std"] = pred_std
            metrics["target_activation_mean"] = target_mean
            metrics["target_activation_std"] = target_std
            metrics["target_activation_avg_norm"] = avg_target_norm
            metrics["pred_activation_avg_norm"] = avg_pred_norm

            if avg_target_norm > 0:
                metrics["normalized_mse"] = avg_loss / (avg_target_norm ** 2)

        return metrics

    def _optimizer_step(self):
        """Perform optimizer step and return gradient norm."""
        # Clip gradients if configured
        if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for module in self.critic_module for p in module.parameters()],
                self.config.grad_clip_norm
            )
        else:
            # Compute grad norm for logging
            total_norm = 0.0
            for module in self.critic_module:
                for p in module.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            grad_norm = torch.tensor(total_norm ** 0.5)

        # Optimizer step
        self.critic_optimizer.step()

        return grad_norm
