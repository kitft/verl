"""MSE-based reward computation for autoencoder training."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from verl.protocol import DataProto


@dataclass
class MSERewardConfig:
    """Configuration for MSE-based rewards."""

    normalize_rewards: bool = True
    reward_scale: float = 1.0
    reward_transform: str = "negative"  # "negative", "exp", "bounded"
    mse_reduction: str = "mean"  # "mean" or "sum"
    clip_rewards: Optional[Tuple[float, float]] = None  # (min, max)
    baseline_subtract: bool = False  # Subtract running mean
    temperature: float = 1.0  # For exp transform


class MSERewardComputer:
    """Computes rewards based on MSE between predicted and target activations."""

    def __init__(self, config: Optional[MSERewardConfig] = None):
        self.config = config or MSERewardConfig()
        self.running_mean = None
        self.running_std = None
        self.num_updates = 0

    def compute_mse(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between predictions and targets."""
        if self.config.mse_reduction == "mean":
            return F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
        elif self.config.mse_reduction == "sum":
            return F.mse_loss(predicted, target, reduction="none").sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction: {self.config.mse_reduction}")

    def transform_mse_to_reward(self, mse: torch.Tensor) -> torch.Tensor:
        """Transform MSE values to rewards."""
        if self.config.reward_transform == "negative":
            # Simple negative MSE
            rewards = -mse

        elif self.config.reward_transform == "exp":
            # Exponential decay (higher MSE = lower reward)
            rewards = torch.exp(-mse * self.config.temperature)

        elif self.config.reward_transform == "bounded":
            # Bounded between 0 and 1
            rewards = 1.0 / (1.0 + mse)

        else:
            raise ValueError(f"Unknown transform: {self.config.reward_transform}")

        return rewards * self.config.reward_scale

    def update_statistics(self, rewards: torch.Tensor):
        """Update running statistics for baseline subtraction."""
        if self.running_mean is None:
            self.running_mean = rewards.mean().item()
            self.running_std = rewards.std().item()
        else:
            # Exponential moving average
            alpha = 0.01
            self.running_mean = (1 - alpha) * self.running_mean + alpha * rewards.mean().item()
            self.running_std = (1 - alpha) * self.running_std + alpha * rewards.std().item()

        self.num_updates += 1

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards."""
        if self.config.baseline_subtract and self.running_mean is not None:
            # Subtract running baseline
            rewards = rewards - self.running_mean
            if self.running_std > 0:
                rewards = rewards / (self.running_std + 1e-8)

        elif self.config.normalize_rewards:
            # Standard normalization
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return rewards

    def compute_rewards(
        self,
        predicted_activations: torch.Tensor,
        target_activations: torch.Tensor,
        return_mse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards from predicted and target activations.

        Args:
            predicted_activations: Critic's predicted activation vectors
            target_activations: Original activation vectors
            return_mse: Whether to return raw MSE values

        Returns:
            Dictionary with rewards and optionally MSE values
        """
        # Compute MSE
        mse = self.compute_mse(predicted_activations, target_activations)

        # Transform to rewards
        rewards = self.transform_mse_to_reward(mse)

        # Update statistics if using baseline
        if self.config.baseline_subtract:
            self.update_statistics(rewards)

        # Normalize
        rewards = self.normalize(rewards)

        # Clip if configured
        if self.config.clip_rewards is not None:
            min_val, max_val = self.config.clip_rewards
            rewards = torch.clamp(rewards, min_val, max_val)

        result = {"rewards": rewards}
        if return_mse:
            result["mse"] = mse

        return result

    def compute_rewards_from_dataproto(
        self,
        data: DataProto,
        predicted_activations: torch.Tensor,
    ) -> DataProto:
        """
        Compute rewards from DataProto and add them back.

        Args:
            data: DataProto containing target activations
            predicted_activations: Critic's predictions

        Returns:
            DataProto with rewards added
        """
        # Extract target activations
        if data.batch is None:
            raise ValueError("DataProto batch missing activation vectors")

        meta_info = getattr(data, "meta_info", None)
        if "activation_vectors" in data.batch.keys():
            target_activations = data.batch["activation_vectors"]
        elif meta_info and "activation_vectors" in meta_info:
            raise ValueError("Activation vectors must be stored in batch['activation_vectors']")
        else:
            raise ValueError("No activation vectors found in DataProto")

        # Compute rewards
        reward_dict = self.compute_rewards(
            predicted_activations,
            target_activations,
            return_mse=True,
        )

        # Add rewards to DataProto
        data.batch.update({
            "rewards": reward_dict["rewards"],
            "mse_loss": reward_dict["mse"],
        })

        return data


class CriticSupervisedLoss:
    """Supervised loss for training the critic to predict activations."""

    def __init__(
        self,
        loss_type: str = "mse",  # "mse", "smooth_l1", "cosine"
        loss_weight: float = 1.0,
        auxiliary_losses: Optional[Dict[str, float]] = None,
    ):
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.auxiliary_losses = auxiliary_losses or {}

    def compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute supervised loss for critic training."""
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction="none")

        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted, target, reduction="none")

        elif self.loss_type == "cosine":
            # Cosine similarity loss (1 - cosine_similarity)
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            loss = 1.0 - cos_sim

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        loss = loss * self.loss_weight

        losses = {"critic_loss": loss}

        # Add auxiliary losses
        if "l2_reg" in self.auxiliary_losses:
            l2_loss = (predicted ** 2).mean() * self.auxiliary_losses["l2_reg"]
            losses["l2_reg"] = l2_loss
            losses["critic_loss"] = losses["critic_loss"] + l2_loss

        if "variance_reg" in self.auxiliary_losses:
            # Encourage diversity in predictions
            var_loss = -predicted.var(dim=0).mean() * self.auxiliary_losses["variance_reg"]
            losses["variance_reg"] = var_loss
            losses["critic_loss"] = losses["critic_loss"] + var_loss

        return losses
