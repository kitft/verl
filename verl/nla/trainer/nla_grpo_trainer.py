"""GRPO-style trainer for NLA with autoencoder critic.

This trainer implements Group Relative Policy Optimization (GRPO) where:
1. We generate N trajectories per prompt
2. Rewards are computed via critic's activation reconstruction MSE
3. Advantages are calculated within each group of N trajectories
4. Both actor and critic are trained simultaneously
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.torch_functional import masked_mean

from ..models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead
from ..rewards.mse_reward import MSERewardComputer, CriticSupervisedLoss
from ..integration.dataproto_adapter import NLADataProtoAdapter


@dataclass
class GRPOTrainerConfig:
    """Configuration for GRPO-style NLA trainer."""

    # GRPO-specific
    num_trajectories_per_prompt: int = 4  # N in GRPO - generate N responses per prompt
    group_normalize_advantages: bool = True  # Use group-wise advantage normalization

    # Critic training
    critic_supervised_weight: float = 1.0  # Weight for supervised MSE loss
    critic_learning_rate: float = 5e-5
    critic_train_epochs: int = 1  # Epochs for critic supervised training

    # Reward computation
    reward_normalize: bool = False  # Let GRPO handle normalization
    reward_transform: str = "negative"  # How to transform MSE to reward
    reward_scale: float = 1.0

    # Actor training
    actor_learning_rate: float = 1e-5
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2


class NLAGRPOTrainer(RayPPOTrainer):
    """
    GRPO-style trainer for NLA with autoencoder critic.

    Key features:
    1. Generate N trajectories per prompt for relative comparison
    2. Critic outputs activation vectors (not scalar values)
    3. Rewards are negative MSE between predicted and target activations
    4. Advantages are computed within each group of N trajectories
    5. Both actor and critic are trained simultaneously
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        grpo_config: Optional[GRPOTrainerConfig] = None,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
    ):
        # Initialize parent RayPPOTrainer with all required parameters
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls or RayWorkerGroup,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        self.grpo_config = grpo_config or GRPOTrainerConfig()

        # Initialize NLA components
        self.reward_computer = MSERewardComputer()
        self.critic_loss_fn = CriticSupervisedLoss()
        self.adapter = NLADataProtoAdapter()

        # Track metrics
        self.grpo_metrics = {
            "avg_mse": [],
            "avg_reward": [],
            "critic_loss": [],
            "group_advantage_std": [],
        }

    def _create_critic_worker(self, worker_group: RayWorkerGroup) -> Any:
        """Create custom critic worker with vector value head."""

        from ..models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead

        class GRPOCriticWorker(worker_group.worker_cls):
            """Custom critic worker with vector output for GRPO."""

            def init_model(self):
                # Initialize the critic as a copy of the base model with vector value head
                self.critic = AutoModelForCausalLMWithVectorValueHead.from_pretrained(
                    self.model_name_or_path,
                    dropout=0.1,
                    **self.model_kwargs
                )

                # Move to device
                self.critic = self.critic.to(self.device)

                # Update optimizer for new parameters
                self.critic_optimizer = torch.optim.Adam(
                    self.critic.parameters(),
                    lr=self.grpo_config.critic_learning_rate,
                )

            def compute_activation_predictions(
                self,
                response_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                """
                Compute predicted activation vectors for responses.

                The critic outputs a vector at the last token position.
                Returns: (batch_size, hidden_size)
                """
                output = self.critic(
                    input_ids=response_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                # Extract the last token's value (which is the activation vector)
                # The model already handles extracting the last valid token
                values = output.value  # (batch, seq_len, hidden_size)

                # Get the last valid position for each sequence
                if attention_mask is not None:
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = values.shape[0]
                    activation_vectors = values[torch.arange(batch_size), seq_lengths]
                else:
                    activation_vectors = values[:, -1]

                return activation_vectors  # (batch, hidden_size)

        return GRPOCriticWorker

    def _generate_multiple_trajectories(
        self,
        prompts: DataProto,
    ) -> DataProto:
        """
        Generate N trajectories per prompt for GRPO with activation injection.

        Args:
            prompts: DataProto with original prompts and activation vectors

        Returns:
            DataProto with expanded batch (batch_size * N) and group_ids
        """
        batch_size = prompts.data["input_ids"].shape[0]
        n_trajectories = self.grpo_config.num_trajectories_per_prompt

        # Extract activation vectors before expansion (if present)
        activation_vectors = None
        if "activation_vectors" in prompts.data:
            activation_vectors = prompts.data["activation_vectors"]
        elif hasattr(prompts, 'metadata') and prompts.metadata and "activation_vectors" in prompts.metadata:
            activation_vectors = prompts.metadata["activation_vectors"]

        # Expand prompts to generate N responses per prompt
        expanded_data = {}
        for key, value in prompts.data.items():
            if isinstance(value, torch.Tensor):
                # Repeat each prompt N times
                expanded_value = value.repeat_interleave(n_trajectories, dim=0)
                expanded_data[key] = expanded_value
            else:
                expanded_data[key] = value

        # Create group IDs to track which responses belong to which prompt
        group_ids = torch.arange(batch_size).repeat_interleave(n_trajectories)

        # Create expanded DataProto
        expanded_prompts = DataProto(
            data=expanded_data,
            metadata={
                "group_ids": group_ids,
                "original_batch_size": batch_size,
                "num_trajectories": n_trajectories,
            }
        )

        # Add activation vectors to expanded prompts if available
        if activation_vectors is not None:
            # Expand activation vectors to match the repeated prompts
            expanded_activation_vectors = activation_vectors.repeat_interleave(n_trajectories, dim=0)

            # Store in both data and metadata for compatibility
            expanded_prompts.data["activation_vectors"] = expanded_activation_vectors
            expanded_prompts.metadata["activation_vectors"] = expanded_activation_vectors

            # Also store original unexpanded vectors for reference
            expanded_prompts.metadata["original_activation_vectors"] = activation_vectors

        # Preserve any other metadata from original prompts
        if hasattr(prompts, 'metadata') and prompts.metadata:
            for key, value in prompts.metadata.items():
                if key not in expanded_prompts.metadata:
                    expanded_prompts.metadata[key] = value

        # Generate responses with the actor (with activation injection if vectors present)
        responses = self.actor_rollout_ref_worker.generate_sequences(expanded_prompts)

        # Preserve group information and activation vectors in responses
        responses.metadata["group_ids"] = group_ids
        responses.metadata["original_batch_size"] = batch_size
        responses.metadata["num_trajectories"] = n_trajectories

        # Preserve activation vectors for critic computation
        if activation_vectors is not None:
            responses.metadata["original_activation_vectors"] = activation_vectors
            if "activation_vectors" not in responses.data:
                responses.data["activation_vectors"] = expanded_activation_vectors

        return responses

    def _compute_rewards_from_critic(
        self,
        data: DataProto,
        critic_worker: Any,
    ) -> DataProto:
        """
        Compute rewards using critic's activation predictions.

        Works on expanded batch (batch_size * N).
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
        self.grpo_metrics["avg_mse"].append(reward_dict["mse"].mean().item())
        self.grpo_metrics["avg_reward"].append(reward_dict["rewards"].mean().item())

        return data

    def compute_advantage_and_returns(
        self,
        data: DataProto,
    ) -> DataProto:
        """
        Compute advantages using GRPO-style group-wise normalization.

        For each prompt, the rewards from its N trajectories are normalized
        relative to each other, providing a strong learning signal about
        which responses are better for that specific prompt.
        """
        rewards = data.data["rewards"]  # Shape: (batch_size * N,)

        # Get group information
        group_ids = data.metadata["group_ids"]  # Shape: (batch_size * N,)
        batch_size = data.metadata.get("original_batch_size", group_ids.max().item() + 1)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Process each group of N trajectories
        for i in range(batch_size):
            # Find all samples belonging to the same original prompt
            group_mask = (group_ids == i)
            group_rewards = rewards[group_mask]

            if group_rewards.numel() > 1:
                # Calculate group statistics
                group_mean = group_rewards.mean()
                group_std = group_rewards.std()

                # GRPO-style advantage: normalize within group
                if self.grpo_config.group_normalize_advantages and group_std > 1e-8:
                    group_advantages = (group_rewards - group_mean) / group_std
                else:
                    # Fallback to simple centering if std is too small
                    group_advantages = group_rewards - group_mean

                # Track group statistics for monitoring
                self.grpo_metrics["group_advantage_std"].append(group_std.item())

                # Returns in GRPO context can be the raw rewards or advantages
                group_returns = group_rewards
            else:
                # Single sample in group - no relative comparison possible
                group_advantages = torch.zeros_like(group_rewards)
                group_returns = group_rewards
                raise ValueError("Single sample in group - no relative comparison possible")

            # Store computed values
            advantages[group_mask] = group_advantages
            returns[group_mask] = group_returns

        # Optional: Global normalization after group normalization
        if self.config.trainer.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data.data["advantages"] = advantages
        data.data["returns"] = returns

        return data

    def _train_critic_supervised(
        self,
        data: DataProto,
        critic_worker: Any,
    ) -> Dict[str, float]:
        """
        Train critic with supervised objective to predict activations.

        The critic learns from all N trajectories per prompt.
        """
        response_ids = data.data.get("response_ids", data.data["input_ids"])
        attention_mask = data.data.get("attention_mask")
        target_activations = self.adapter.extract_activation_vectors_from_dataproto(data)

        if target_activations is None:
            raise ValueError("No target activation vectors found")

        total_loss = 0.0
        num_updates = 0

        # Train for configured number of epochs
        for epoch in range(self.grpo_config.critic_train_epochs):
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
        self.grpo_metrics["critic_loss"].append(avg_loss)

        return {"critic_supervised_loss": avg_loss}

    def _train_actor_with_ppo(
        self,
        data: DataProto,
        actor_worker: Any,
    ) -> Dict[str, float]:
        """
        Train actor with PPO using GRPO advantages.

        The actor learns to generate responses that are better than
        the other N-1 alternatives for the same prompt.
        """
        # Ensure we have advantages
        if "advantages" not in data.data:
            raise ValueError("Advantages not computed. Run compute_advantage_and_returns first.")

        # Standard PPO update with GRPO advantages
        metrics = actor_worker.update_actor(data)

        return metrics

    def training_step(
        self,
        prompts: DataProto,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        GRPO training step for NLA.

        Flow:
        1. Generate N responses per prompt with activation injection
        2. Critic predicts activation vectors from all N*batch_size responses
        3. Compute MSE-based rewards for each response
        4. Calculate GRPO advantages within each group of N
        5. Train critic with supervised loss on all responses
        6. Train actor with PPO using GRPO advantages
        """
        metrics = {}

        # Step 1: Generate N trajectories per prompt
        with self.timer("generate_multiple_trajectories"):
            responses = self._generate_multiple_trajectories(prompts)

        # Step 2 & 3: Compute rewards using critic
        with self.timer("compute_rewards"):
            responses = self._compute_rewards_from_critic(
                data=responses,
                critic_worker=self.critic_worker,
            )

        # Step 4: Compute GRPO advantages
        with self.timer("compute_grpo_advantages"):
            responses = self.compute_advantage_and_returns(responses)

        # Step 5: Train critic with supervised objective
        with self.timer("train_critic"):
            critic_metrics = self._train_critic_supervised(
                data=responses,
                critic_worker=self.critic_worker,
            )
            metrics.update(critic_metrics)

        # Step 6: Train actor with PPO using GRPO advantages
        with self.timer("train_actor"):
            actor_metrics = self._train_actor_with_ppo(
                data=responses,
                actor_worker=self.actor_rollout_ref_worker,
            )
            metrics.update(actor_metrics)

        # Add GRPO-specific metrics
        if self.grpo_metrics["avg_mse"]:
            metrics["grpo/avg_mse"] = sum(self.grpo_metrics["avg_mse"]) / len(self.grpo_metrics["avg_mse"])
        if self.grpo_metrics["avg_reward"]:
            metrics["grpo/avg_reward"] = sum(self.grpo_metrics["avg_reward"]) / len(self.grpo_metrics["avg_reward"])
        if self.grpo_metrics["critic_loss"]:
            metrics["grpo/critic_loss"] = sum(self.grpo_metrics["critic_loss"]) / len(self.grpo_metrics["critic_loss"])
        if self.grpo_metrics["group_advantage_std"]:
            metrics["grpo/group_advantage_std"] = sum(self.grpo_metrics["group_advantage_std"]) / len(self.grpo_metrics["group_advantage_std"])

        metrics["grpo/num_trajectories"] = self.grpo_config.num_trajectories_per_prompt

        # Clear metrics for next iteration
        for key in self.grpo_metrics:
            self.grpo_metrics[key] = []

        return metrics