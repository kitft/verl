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

    def _extract_value_tensor(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """Return activation tensor, preferring explicit value head else hidden states."""
        if hasattr(model_outputs, "value") and model_outputs.value is not None:
            value = model_outputs.value
            if value.dim() >= 3:
                return value
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
            return model_outputs.hidden_states[-1]
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
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

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
                values = values[:, -response_length - 1 : -1, :]
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
                values = values[:, -response_length - 1 : -1, :]

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
        """Compute activation vector predictions without tracking gradients."""
        return self._compute_values_internal(data, enable_grads=False)

    def compute_activation_loss(
        self,
        predicted_activations: torch.Tensor,
        target_activations: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        pooling: str = "last"
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target activations.

        Args:
            predicted_activations: (batch, seq_len, hidden_size)
            target_activations: (batch, hidden_size)
            response_mask: Optional mask for valid tokens
            pooling: How to pool sequence predictions ("last", "mean", "max")

        Returns:
            MSE loss scalar
        """
        batch_size = predicted_activations.shape[0]

        # Pool the sequence dimension based on strategy
        if pooling == "last":
            # Take the last token's prediction
            if response_mask is not None:
                # Find last valid token for each sequence
                last_indices = response_mask.sum(dim=1) - 1  # (batch,)
                last_indices = last_indices.clamp(min=0)

                # Gather predictions at last positions
                pooled_predictions = predicted_activations[
                    torch.arange(batch_size), last_indices
                ]  # (batch, activation_dim)
            else:
                pooled_predictions = predicted_activations[:, -1, :]

        elif pooling == "mean":
            if response_mask is not None:
                # Masked mean over sequence
                mask_expanded = response_mask.unsqueeze(-1)  # (batch, seq_len, 1)
                pooled_predictions = (predicted_activations * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled_predictions = predicted_activations.mean(dim=1)

        elif pooling == "max":
            if response_mask is not None:
                # Set masked positions to very negative value before max
                mask_expanded = response_mask.unsqueeze(-1)
                masked_predictions = predicted_activations.masked_fill(~mask_expanded.bool(), -1e9)
                pooled_predictions, _ = masked_predictions.max(dim=1)
            else:
                pooled_predictions, _ = predicted_activations.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        # Compute MSE loss
        mse_loss = nn.functional.mse_loss(pooled_predictions, target_activations)

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

        # Get required data
        micro_batch_size = data.meta_info.get("micro_batch_size", 1)

        # Extract target activations from data
        # These should be provided in the DataProto
        if "target_activations" not in data.batch:
            raise ValueError("target_activations must be provided in data.batch for NLA critic training")

        target_activations = data.batch["target_activations"]  # (batch, hidden_size)

        # Compute predictions using parent's forward logic
        predicted_activations = self._compute_values_internal(data, enable_grads=True)

        # Get response mask if available
        response_mask = data.batch.get("response_mask", None)

        # Compute MSE loss
        loss = self.compute_activation_loss(
            predicted_activations,
            target_activations.to(predicted_activations.device),
            response_mask.to(predicted_activations.device) if response_mask is not None else None,
            pooling=self.config.get("pooling_strategy", "last")
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

            metrics["pred_activation_mean"] = pred_mean
            metrics["pred_activation_std"] = pred_std
            metrics["target_activation_mean"] = target_mean
            metrics["target_activation_std"] = target_std

        return metrics
