"""NLA Critic that outputs activation vectors instead of scalars."""

import torch
import torch.nn as nn
from torch import optim
from typing import Optional

from verl import DataProto
from verl.workers.critic.dp_critic import DataParallelPPOCritic
from verl.utils.torch_functional import masked_mean


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
        activation_dim: int = 8,
    ):
        super().__init__(config, critic_module, critic_optimizer)
        self.activation_dim = activation_dim

    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        Compute activation vector predictions.

        Returns:
            torch.Tensor: Predicted activation vectors (batch_size, seq_len, activation_dim)
        """
        # Call parent's compute_values which handles micro-batching and model forward
        # The parent expects scalar values, but our model outputs vectors
        values = super().compute_values(data)

        # Values should be (batch_size, response_len, activation_dim)
        # The parent class already handles extracting just the response portion
        return values

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
            predicted_activations: (batch, seq_len, activation_dim)
            target_activations: (batch, activation_dim)
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

        target_activations = data.batch["target_activations"]  # (batch, activation_dim)

        # Compute predictions using parent's forward logic
        predicted_activations = self.compute_values(data)  # (batch, seq_len, activation_dim)

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
        metrics["activation_dim"] = self.activation_dim

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