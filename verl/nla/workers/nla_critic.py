"""NLA Critic that outputs activation vectors instead of scalars."""

import logging

import torch
import torch.nn as nn
from torch import optim
from typing import Optional

from verl import DataProto
from verl.workers.critic.dp_critic import DataParallelPPOCritic
from verl.utils.device import get_device_id, is_cuda_available, is_npu_available
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.trainer.ppo.ray_trainer import compute_response_mask

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


class NLADataParallelCritic(DataParallelPPOCritic):
    """
    NLA Critic that extends the standard VERL critic but:
    1. Outputs activation vectors instead of scalar values
    2. Computes MSE loss against target activations
    """

    def __init__(
        self,
        config,
        critic_module: nn.Module,
        critic_optimizer: optim.Optimizer,
    ):
        super().__init__(config, critic_module, critic_optimizer)
        self.logger = logging.getLogger(__name__)
        # Get hidden size from the critic model
        self.hidden_size = critic_module.config.hidden_size if hasattr(critic_module, 'config') else None
        # Initialize NLA adapter for extracting activation vectors
        from verl.nla.integration.dataproto_adapter import NLADataProtoAdapter
        self.adapter = NLADataProtoAdapter()

    def _extract_value_tensor(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """Return activation tensor, preferring explicit value head else hidden states."""
        # print(f"model outputs keys: {model_outputs.keys()}" if isinstance(model_outputs, dict) else f"model outputs type: {type(model_outputs)}")
        # print("length of hidden states:", len(model_outputs.hidden_states) if hasattr(model_outputs, "hidden_states") else "hidden_states not found")

        # if hasattr(model_outputs, "value") and model_outputs.value is not None:
        #     value = model_outputs.value
        #     if value.dim() >= 3:
        #         return value
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            return model_outputs.hidden_states[self.config.output_layer_index]
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 2:
            candidate = model_outputs[2]
            if isinstance(candidate, torch.Tensor) and candidate.dim() >= 3:
                return candidate
        raise ValueError("Critic model output does not contain activation tensor")

    def _forward_micro_batch(self, micro_batch):
        """Run forward pass for a micro batch returning activation vectors."""
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # Use only responses as model input
            input_ids = responses
            batch, seqlen = input_ids.shape

            # Extract attention mask for responses only
            attention_mask_full = micro_batch["attention_mask"]
            attention_mask = attention_mask_full[:, -response_length:]

            # Create fresh position IDs starting from 0 for responses
            # This is needed for Flash Attention which expects position IDs to start from 0
            position_ids_full = micro_batch["position_ids"]
            if position_ids_full.dim() == 3:  # qwen2vl mrope
                # For multi-rope, create position IDs from scratch based on attention mask
                # Keep the same structure but reset values
                position_ids_full_transposed = position_ids_full.transpose(0, 1)
                num_ropes = position_ids_full_transposed.shape[0]
                position_ids = torch.zeros(
                    (num_ropes, batch, response_length),
                    dtype=position_ids_full.dtype,
                    device=position_ids_full.device
                )
                for rope_idx in range(num_ropes):
                    for b_idx in range(batch):
                        valid_len = attention_mask[b_idx].sum().item()
                        position_ids[rope_idx, b_idx, :valid_len] = torch.arange(
                            valid_len, dtype=position_ids_full.dtype, device=position_ids_full.device
                        )
                position_ids = position_ids.transpose(0, 1)
            else:
                # Standard case: create position IDs from 0 based on attention mask
                position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
                for b_idx in range(batch):
                    valid_len = attention_mask[b_idx].sum().item()
                    position_ids[b_idx, :valid_len] = torch.arange(
                        valid_len, dtype=torch.long, device=attention_mask.device
                    )

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, max_seqlen = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                outputs = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
                values_rmpad = self._extract_value_tensor(outputs)

                if values_rmpad.dim() == 3:
                    values_rmpad = rearrange(values_rmpad, "b s h -> (b s) h")
                else:
                    values_rmpad = values_rmpad.reshape(values_rmpad.shape[0], -1)

                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
            else:
                outputs = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
                values = self._extract_value_tensor(outputs)

        return values

    def _prepare_meta_info(self, data: DataProto) -> dict:
        if data.meta_info is None:
            data.meta_info = {}
        meta = data.meta_info
        meta.setdefault("micro_batch_size", self.config.forward_micro_batch_size_per_gpu)
        meta.setdefault("max_token_len", self.config.forward_max_token_len_per_gpu)
        meta.setdefault("use_dynamic_bsz", self.config.use_dynamic_bsz)
        return meta

    def _broadcast_response_mask(self, data: DataProto) -> Optional[torch.Tensor]:
        if data.batch is None or "response_mask" not in data.batch.keys():
            return None
        mask = data.batch["response_mask"]
        if mask is None or mask.dim() != 2:
            return None
        # DataParallelPPOCritic assumes scalar critics and multiplies response_mask directly.
        # Our activations are vectors, so temporarily unsqueeze to enable broadcasting.
        # TODO(sg): override compute_values fully to avoid this shim.
        self.logger.debug("Broadcasting response_mask for vector-valued critic")
        data.batch["response_mask"] = mask.unsqueeze(-1)
        return mask

    def _compute_values_internal(self, data: DataProto, *, enable_grads: bool) -> torch.Tensor:
        meta = self._prepare_meta_info(data)
        restore_mask = self._broadcast_response_mask(data)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data_selected = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        micro_batch_size = meta["micro_batch_size"]
        use_dynamic_bsz = meta["use_dynamic_bsz"]

        if use_dynamic_bsz:
            max_token_len = meta["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data_selected, max_token_len=max_token_len
            )
        else:
            micro_batches = data_selected.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            if enable_grads:
                values = self._forward_micro_batch(model_inputs)
            else:
                with torch.no_grad():
                    values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            values = restore_dynamic_batch(values, batch_idx_list)

        if "response_mask" in data_selected.batch:
            response_mask = data_selected.batch["response_mask"].to(values.device)
            values = values * response_mask

        if restore_mask is not None:
            data.batch["response_mask"] = restore_mask

        return values

    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        Compute MSE-based scalar values (negative reconstruction error).

        Returns values as (batch, seq_len) tensor where higher values = better reconstruction.
        This allows NLA to use standard VERL GRPO advantage computation.

        Also stores target_activations in data.batch for use by update_critic.

        If values already exist in data.batch (from earlier shim call), returns them immediately
        without recomputation.
        """
        # Check if values already computed (e.g., by reward shim)
        if "values" in data.batch.keys():
            self.logger.debug("Values already computed, skipping recomputation")
            return data.batch["values"]

        # Compute response_mask if not present (needed for proper masking)
        if "response_mask" not in data.batch.keys():
            response_mask = compute_response_mask(data)
            data.batch["response_mask"] = response_mask
            self.logger.debug(f"Computed response_mask with shape {response_mask.shape}")

        # Predicted activations for response tokens only
        full_activations = self._compute_values_internal(data, enable_grads=False)  # (b, L, H)

        # Targets (b, H)
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data, raise_on_missing=True)
        if not isinstance(target_activations, torch.Tensor):
            target_activations = torch.as_tensor(
                target_activations,
                device=full_activations.device,
                dtype=full_activations.dtype
            )
        else:
            target_activations = target_activations.to(
                device=full_activations.device,
                dtype=full_activations.dtype
            )

        # Pool last non-padded token for each response
        response_mask = data.batch.get("response_mask", None)  # (b, L)
        pooled = self.extract_predicted_activations(
            full_activations, response_mask, pooling="last"
        )  # (b, H)
        if response_mask.shape != full_activations.shape[:2]:
            raise ValueError(f"Response mask shape {response_mask.shape} does not match full activations shape {full_activations.shape[:2]}")
        print(f"Number of nonzero elements of response mask of 0th sequence: {response_mask[0].sum().item()}")
        print(f"first response: {data.batch['responses'][0]}")

        # Per-sample MSE on the pooled last token
        mse_per_sample = ((pooled - target_activations) ** 2).mean(dim=-1)  # (b,)

        # Broadcast scalar to all response positions for compatibility
        resp_len = data.batch["responses"].size(-1)
        values = (-mse_per_sample).unsqueeze(-1).expand(-1, resp_len).to(full_activations.dtype)

        # Put value only on last valid token
        if response_mask is not None:
            last_indices = response_mask.sum(dim=1) - 1
            last_indices = last_indices.clamp(min=0)
            values = torch.zeros_like(values)
            values[torch.arange(values.shape[0]), last_indices] = (-mse_per_sample).to(values.dtype)
        else:
            raise ValueError("Response mask is required for last pooling")

        # Log metrics
        with torch.no_grad():
            if response_mask is not None:
                mask_sum = response_mask.to(values.dtype).sum()
                if mask_sum > 0:
                    valid_mse = mse_per_sample.mean()
                else:
                    valid_mse = mse_per_sample.mean()
            else:
                valid_mse = mse_per_sample.mean()
            self.logger.debug(f"Computed values: mse_mean={valid_mse.item():.6f}, reward_mean={-valid_mse.item():.6f}")

        return values

    def extract_predicted_activations(self, full_activations: torch.Tensor, response_mask: Optional[torch.Tensor] = None, pooling: str = "last") -> torch.Tensor:
        """
        Args:
            full_activations: (batch, seq_len, hidden_size)
            response_mask: Optional mask for valid tokens
            pooling: How to pool sequence predictions ("last", "mean", "max")

        Returns:
            Predicted activations tensor (batch, hidden_size)
        Extract predicted activations from the full activations tensor.
        """
        batch_size = full_activations.shape[0]

        # Pool the sequence dimension based on strategy
        if pooling == "last":
            # Take the last token's prediction
            if response_mask is not None:
                # Find last valid token for each sequence
                last_indices = response_mask.sum(dim=1) - 1  # (batch,)
                last_indices = last_indices.clamp(min=0)

                # Gather predictions at last positions
                pooled_predictions = full_activations[
                    torch.arange(batch_size), last_indices
                ]  # (batch, activation_dim)
                # print(f"last indices (count of each): {last_indices.unique(return_counts=True)}, for shape of full_activations: {full_activations.shape}")
                # print(f"avg norms of extraction vector by position")
                # #for i in range(len(last_indices)):
                # #    print(f"position {i}: {torch.norm(full_activations[i, last_indices[i]]).item()}")
                # print(f"norms of first sequence: {[torch.norm(full_activations[0,i]) for i in range(full_activations.shape[1])]}")
                # print(f"norms of second sequence: {[torch.norm(full_activations[1,i]) for i in range(full_activations.shape[1])]}")
            else:
                raise ValueError("Response mask is required for last pooling")
                pooled_predictions = full_activations[:, -1, :]

        elif pooling == "mean":
            if response_mask is not None:
                # Masked mean over sequence
                mask_expanded = response_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                pooled_predictions = (full_activations * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled_predictions = full_activations.mean(dim=1)

        elif pooling == "max":
            if response_mask is not None:
                # Set masked positions to very negative value before max
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
        """
        Compute MSE loss between predicted and target activations.

        Args:
            predicted_activations: (batch, hidden_size)
            target_activations: (batch, hidden_size)
            response_mask: Optional mask for valid tokens
            pooling: How to pool sequence predictions ("last", "mean", "max")

        Returns:
            MSE loss scalar
        """

        # Compute MSE loss
        mse_loss = nn.functional.mse_loss(predicted_activations, target_activations)

        return mse_loss

    def update_critic(self, data: DataProto):
        """
        Update the critic with MSE loss against target activations.

        This overrides the parent's update_critic to use MSE loss
        instead of the standard PPO value loss.
        """
        # Make sure we are in training mode
        self.critic_module.train()

        metrics = {}

        # NLA-specific: include activation_vectors in select_keys
        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns", "activation_vectors"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Extract target activations from data
        # raise_on_missing=True ensures we fail fast if activations are missing
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data, raise_on_missing=True)

        # Compute predictions using parent's forward logic
        full_activations = self._compute_values_internal(data, enable_grads=True)
        predicted_activations = self.extract_predicted_activations(full_activations, data.batch["response_mask"], pooling=self.config.get("pooling_strategy", "last"))

        # Get response mask if available
        response_mask = data.batch.get("response_mask", None)

        # Compute MSE loss
        loss = self.compute_activation_loss(
            predicted_activations,
            target_activations.to(device=predicted_activations.device, dtype=predicted_activations.dtype),
        )

        # Backward pass
        self.critic_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping and optimizer step
        grad_norm = self._optimizer_step()

        # Log metrics
        metrics["critic_loss"] = loss.item()
        metrics["grad_norm"] = grad_norm.item() if torch.isfinite(grad_norm) else float('inf')
        metrics["hidden_size"] = self.hidden_size if self.hidden_size else predicted_activations.shape[-1]

        # Log some statistics about predictions
        with torch.no_grad():
            pred_mean = predicted_activations.mean().item()
            pred_std = predicted_activations.std().item()
            target_mean = target_activations.mean().item()
            target_std = target_activations.std().item()

            # Compute average norm of target activations (per sample)
            target_norms = torch.norm(target_activations, dim=1)  # (batch,)
            avg_target_norm = target_norms.mean().item()
            pred_norms = torch.norm(predicted_activations, dim=1)  # (batch,)
            avg_pred_norm = pred_norms.mean().item()    
            max_target_norm = target_norms.max().item()
            mse_with_zero_pred = nn.functional.mse_loss(torch.zeros_like(predicted_activations).to(target_activations.device), target_activations).item()
            mse_with_mean_pred_over_batch = nn.functional.mse_loss(target_activations.mean(dim=0).unsqueeze(0).expand_as(predicted_activations), target_activations).item()

            metrics["pred_activation_mean"] = pred_mean
            metrics["pred_activation_std"] = pred_std
            metrics["target_activation_mean"] = target_mean
            metrics["target_activation_std"] = target_std
            metrics["target_activation_avg_norm"] = avg_target_norm
            metrics["target_activation_max_norm"] = max_target_norm
            metrics["pred_activation_avg_norm"] = avg_pred_norm
            metrics["mse_with_zero_pred"] = mse_with_zero_pred
            metrics["mse_with_mean_pred_over_batch"] = mse_with_mean_pred_over_batch

            # Normalized MSE: mse / (average_norm^2)
            # This gives reconstruction error relative to the typical activation magnitude
            # Value of 1.0 means RMSE equals the average norm (poor reconstruction)
            # Value of 0.1 means RMSE is 10% of average norm (good reconstruction)
            if avg_target_norm > 0:
                normalized_mse = loss.item() / (avg_target_norm ** 2)
                metrics["normalized_mse"] = normalized_mse
            else:
                metrics["normalized_mse"] = float('inf')
            
            if mse_with_zero_pred > 0:
                normalized_mse_with_zero_pred = loss.item() / mse_with_zero_pred
                metrics["normalized_mse_by_zero"] = normalized_mse_with_zero_pred
            else:
                metrics["normalized_mse_by_zero"] = float('inf')

            if mse_with_mean_pred_over_batch > 0:
                normalized_mse_with_mean_pred_over_batch = loss.item() / mse_with_mean_pred_over_batch
                metrics["normalized_mse_by_mean_pred_over_batch"] = normalized_mse_with_mean_pred_over_batch
            else:
                metrics["normalized_mse_by_mean_pred_over_batch"] = float('inf')

        return metrics
