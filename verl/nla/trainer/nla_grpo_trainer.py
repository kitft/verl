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
        batch_size = prompts.batch["input_ids"].shape[0]
        n_trajectories = self.grpo_config.num_trajectories_per_prompt

        # Extract activation vectors before expansion (if present)
        activation_vectors = None
        if "activation_vectors" in prompts.batch.keys():
            activation_vectors = prompts.batch["activation_vectors"]
        elif prompts.meta_info and "activation_vectors" in prompts.meta_info:
            activation_vectors = prompts.meta_info["activation_vectors"]

        # Expand prompts to generate N responses per prompt
        expanded_prompts = prompts.repeat(repeat_times=n_trajectories, interleave=True)

        # Create group IDs to track which responses belong to which prompt
        group_ids = torch.arange(batch_size).repeat_interleave(n_trajectories)

        # Clone meta info to avoid modifying original prompts
        expanded_meta = dict(expanded_prompts.meta_info) if expanded_prompts.meta_info else {}
        expanded_meta.update(
            {
                "group_ids": group_ids,
                "original_batch_size": batch_size,
                "num_trajectories": n_trajectories,
            }
        )
        if activation_vectors is not None:
            expanded_meta["original_activation_vectors"] = activation_vectors
        expanded_prompts.meta_info = expanded_meta

        # Generate responses with the actor (with activation injection if vectors present)
        responses = self.actor_rollout_ref_worker.generate_sequences(expanded_prompts)

        # Preserve group information and activation vectors in responses
        response_meta = dict(responses.meta_info) if responses.meta_info else {}
        response_meta.update(
            {
                "group_ids": group_ids,
                "original_batch_size": batch_size,
                "num_trajectories": n_trajectories,
            }
        )
        if activation_vectors is not None:
            response_meta["original_activation_vectors"] = activation_vectors
        responses.meta_info = response_meta

        # Ensure activation vectors stay available for downstream modules
        if activation_vectors is not None and "activation_vectors" not in responses.batch.keys():
            expanded_activation_vectors = activation_vectors.repeat_interleave(n_trajectories, dim=0)
            responses.batch.update({"activation_vectors": expanded_activation_vectors})

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
        response_ids = data.batch["response_ids"] if "response_ids" in data.batch.keys() else data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"] if "attention_mask" in data.batch.keys() else None

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
        data.batch.update(
            {
                "rewards": reward_dict["rewards"],
                "mse_values": reward_dict["mse"],
                "predicted_activations": predicted_activations,
            }
        )

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
        rewards = data.batch["rewards"]  # Shape: (batch_size * N,)

        # Get group information
        if not data.meta_info:
            raise ValueError("Group metadata missing from DataProto")

        group_ids = data.meta_info["group_ids"]  # Shape: (batch_size * N,)
        batch_size = data.meta_info.get("original_batch_size", group_ids.max().item() + 1)

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

        data.batch.update({
            "advantages": advantages,
            "returns": returns,
        })

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
        response_ids = data.batch["response_ids"] if "response_ids" in data.batch.keys() else data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"] if "attention_mask" in data.batch.keys() else None
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
        if "advantages" not in data.batch.keys():
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
