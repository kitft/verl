"""Custom PPO trainer for NLA autoencoder-style training."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.torch_functional import masked_mean

from ..models.autoencoder_critic import NLAAutoencoderCritic
from ..rewards.mse_reward import MSERewardComputer, CriticSupervisedLoss
from ..integration.dataproto_adapter import NLADataProtoAdapter


@dataclass
class NLATrainerConfig:
    """Configuration for NLA PPO trainer."""

    # Critic training
    critic_supervised_weight: float = 1.0  # Weight for supervised MSE loss
    critic_learning_rate: float = 5e-5
    critic_train_epochs: int = 1  # Epochs for critic supervised training

    # Reward computation
    reward_normalize: bool = True
    reward_transform: str = "negative"  # How to transform MSE to reward
    reward_scale: float = 1.0

    # Actor training
    actor_learning_rate: float = 1e-5
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2

    # Activation settings
    activation_dim: int = 768
    use_pooling: str = "last"  # How to pool response hidden states


class NLAAutoencoderPPOTrainer(RayPPOTrainer):
    """
    PPO trainer for NLA with autoencoder-style critic.

    Key differences from standard PPO:
    1. Critic outputs activation vectors instead of scalar values
    2. Rewards are computed as negative MSE with original activations
    3. Critic is trained with supervised objective
    4. Actor is trained with RL objective using MSE-based rewards
    """

    def __init__(
        self,
        config: DictConfig,
        nla_config: Optional[NLATrainerConfig] = None,
    ):
        super().__init__(config)
        self.nla_config = nla_config or NLATrainerConfig()

        # Initialize NLA components
        self.reward_computer = MSERewardComputer()
        self.critic_loss_fn = CriticSupervisedLoss()
        self.adapter = NLADataProtoAdapter()

        # Track metrics
        self.nla_metrics = {
            "avg_mse": [],
            "avg_reward": [],
            "critic_loss": [],
        }

    def _create_critic_worker(self, worker_group: RayWorkerGroup) -> Any:
        """Create custom critic worker with autoencoder head."""

        class NLACriticWorker(worker_group.worker_cls):
            """Custom critic worker with vector output."""

            def init_model(self):
                # Initialize base model
                super().init_model()

                # Replace value head with autoencoder critic
                self.critic = NLAAutoencoderCritic(
                    base_model=self.model,
                    activation_dim=self.nla_config.activation_dim,
                    use_pooling=self.nla_config.use_pooling,
                )

                # Update optimizer for new parameters
                self.critic_optimizer = torch.optim.Adam(
                    self.critic.parameters(),
                    lr=self.nla_config.critic_learning_rate,
                )

            def compute_activation_predictions(
                self,
                response_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                """Compute predicted activation vectors for responses."""
                output = self.critic(
                    input_ids=response_ids,
                    attention_mask=attention_mask,
                )
                return output.predicted_activation

        return NLACriticWorker

    def _compute_rewards_from_critic(
        self,
        data: DataProto,
        critic_worker: Any,
    ) -> DataProto:
        """
        Compute rewards using critic's activation predictions.

        Args:
            data: DataProto with generated responses and original activations
            critic_worker: Critic worker instance

        Returns:
            DataProto with computed rewards
        """
        # Extract response tokens
        response_ids = data.data.get("response_ids", data.data["input_ids"])
        attention_mask = data.data.get("attention_mask")

        # Get critic's activation predictions
        with torch.no_grad():
            predicted_activations = critic_worker.compute_activation_predictions(
                response_ids=response_ids,
                attention_mask=attention_mask,
            )

        # Extract original activation vectors
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data)

        if target_activations is None:
            raise ValueError("No target activation vectors found in data")

        # Compute MSE-based rewards
        reward_dict = self.reward_computer.compute_rewards(
            predicted_activations=predicted_activations,
            target_activations=target_activations,
            return_mse=True,
        )

        # Add rewards to data
        data.data["rewards"] = reward_dict["rewards"]
        data.data["mse_values"] = reward_dict["mse"]
        data.data["predicted_activations"] = predicted_activations

        # Track metrics
        self.nla_metrics["avg_mse"].append(reward_dict["mse"].mean().item())
        self.nla_metrics["avg_reward"].append(reward_dict["rewards"].mean().item())

        return data

    def _train_critic_supervised(
        self,
        data: DataProto,
        critic_worker: Any,
    ) -> Dict[str, float]:
        """
        Train critic with supervised objective to predict activations.

        Args:
            data: DataProto with responses and target activations
            critic_worker: Critic worker instance

        Returns:
            Dictionary of training metrics
        """
        response_ids = data.data.get("response_ids", data.data["input_ids"])
        attention_mask = data.data.get("attention_mask")
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data)

        if target_activations is None:
            raise ValueError("No target activation vectors found")

        total_loss = 0.0
        num_updates = 0

        # Train for configured number of epochs
        for epoch in range(self.nla_config.critic_train_epochs):
            # Forward pass
            predicted_activations = critic_worker.compute_activation_predictions(
                response_ids=response_ids,
                attention_mask=attention_mask,
            )

            # Compute supervised loss
            losses = self.critic_loss_fn.compute_loss(
                predicted=predicted_activations,
                target=target_activations,
            )

            # Backward pass
            critic_worker.critic_optimizer.zero_grad()
            losses["critic_loss"].backward()
            critic_worker.critic_optimizer.step()

            total_loss += losses["critic_loss"].item()
            num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)
        self.nla_metrics["critic_loss"].append(avg_loss)

        return {"critic_supervised_loss": avg_loss}

    def _train_actor_with_ppo(
        self,
        data: DataProto,
        actor_worker: Any,
    ) -> Dict[str, float]:
        """
        Train actor with PPO using MSE-based rewards.

        This follows standard PPO but uses rewards computed from critic's
        activation predictions vs original activations.
        """
        # Ensure we have rewards
        if "rewards" not in data.data:
            raise ValueError("Rewards not computed. Run _compute_rewards_from_critic first.")

        # Standard PPO update with computed rewards
        metrics = actor_worker.update_actor(data)

        return metrics

    def training_step(
        self,
        prompts: DataProto,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Custom training step for NLA autoencoder training.

        Flow:
        1. Generate responses with activation injection
        2. Critic predicts activation vectors from responses
        3. Compute MSE-based rewards
        4. Train critic with supervised loss
        5. Train actor with RL loss using MSE rewards
        """
        metrics = {}

        # Step 1: Generate responses with activation injection
        with self.timer("generate_sequences"):
            responses = self.actor_rollout_ref_worker.generate_sequences(prompts)

        # Step 2 & 3: Compute rewards using critic
        with self.timer("compute_rewards"):
            responses = self._compute_rewards_from_critic(
                data=responses,
                critic_worker=self.critic_worker,
            )

        # Step 4: Train critic with supervised objective
        with self.timer("train_critic"):
            critic_metrics = self._train_critic_supervised(
                data=responses,
                critic_worker=self.critic_worker,
            )
            metrics.update(critic_metrics)

        # Step 5: Train actor with PPO using MSE-based rewards
        with self.timer("train_actor"):
            actor_metrics = self._train_actor_with_ppo(
                data=responses,
                actor_worker=self.actor_rollout_ref_worker,
            )
            metrics.update(actor_metrics)

        # Add NLA-specific metrics
        metrics["nla/avg_mse"] = sum(self.nla_metrics["avg_mse"]) / len(self.nla_metrics["avg_mse"])
        metrics["nla/avg_reward"] = sum(self.nla_metrics["avg_reward"]) / len(self.nla_metrics["avg_reward"])
        metrics["nla/critic_loss"] = sum(self.nla_metrics["critic_loss"]) / len(self.nla_metrics["critic_loss"])

        # Clear metrics for next iteration
        for key in self.nla_metrics:
            self.nla_metrics[key] = []

        return metrics

    def compute_advantage_and_returns(
        self,
        data: DataProto,
    ) -> DataProto:
        """
        Override to use MSE-based rewards instead of value estimates.

        In autoencoder training, advantages are based on MSE rewards
        rather than critic value predictions.
        """
        rewards = data.data["rewards"]

        # Simple advantage = reward - baseline
        # The baseline could be a running average of rewards
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Normalize advantages
        if self.config.trainer.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data.data["advantages"] = advantages
        data.data["returns"] = rewards  # In this setup, returns = rewards

        return data