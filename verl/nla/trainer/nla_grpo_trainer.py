"""GRPO-style trainer for NLA with autoencoder critic.

This trainer implements Group Relative Policy Optimization (GRPO) where:
1. We generate N trajectories per prompt
2. Rewards are computed via critic's activation reconstruction MSE
3. Advantages are calculated within each group of N trajectories
4. Both actor and critic are trained simultaneously
"""

from typing import Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup


class _ValuesAsRewardsShim:
    """
    Minimal shim that converts critic values (MSE-based) to token_level_scores.

    NLA critic.compute_values() returns (batch, seq_len) MSE-based values.
    This shim computes them if not present, then returns as token_level_scores.

    This is called BEFORE the base fit() calls compute_values, so we need to
    compute them ourselves and cache in the batch.
    """
    def __init__(self, trainer: "NLAGRPOTrainer"):
        self.trainer = trainer

    def __call__(self, data: DataProto, return_dict: bool = False):
        # Check if values already computed (they won't be on first call from reward computation)
        if "values" not in data.batch:
            # Compute values using critic - this returns MSE-based per-token values
            # Shape: (batch, seq_len) where values = -MSE (higher is better)
            values_output = self.trainer.critic_wg.compute_values(data)
            # Mutate the original batch in-place so values are cached for line 1089
            data.batch.update(values_output.batch)

        # Use values as token_level_scores (they're already per-token rewards)
        token_level_scores = data.batch["values"]

        reward_extra_info = {}
        if return_dict:
            return {"reward_tensor": token_level_scores, "reward_extra_info": reward_extra_info}
        return token_level_scores

@dataclass
class GRPOTrainerConfig:
    """Configuration for GRPO-style NLA trainer.

    Note: N (number of trajectories per prompt) is controlled by
    config.actor_rollout_ref.rollout.n, NOT here.
    """

    # GRPO-specific
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

        # NOTE: grpo_config is not used - all GRPO logic is in the base fit() routine
        # self.grpo_config = grpo_config or GRPOTrainerConfig()

        # Use minimal shim to convert critic values to token_level_scores
        # (base class always calls reward_fn, so we need this adapter)
        if reward_fn is None:
            self.reward_fn = _ValuesAsRewardsShim(self)
        if val_reward_fn is None:
            self.val_reward_fn = _ValuesAsRewardsShim(self)


    # NOTE: _generate_multiple_trajectories is NOT NEEDED
    # The base fit() already handles N-way generation via rollout.n config parameter.
    # It expands the batch at line 996 and generates N responses per prompt automatically.

    # NOTE: All custom reward/advantage computation methods are NOT NEEDED
    # The base fit() already handles:
    # - Reward computation via reward_fn (our shim)
    # - Advantage computation via compute_advantage() with GRPO estimator
    # - Critic updates via critic_wg.update_critic()
    # - Actor updates via actor_rollout_wg.update_actor()

    # NOTE: training_step is NOT USED - we rely on the base fit() routine
    # The base RayPPOTrainer.fit() doesn't call training_step, it implements
    # the training loop inline. Our strategy is to make compute_values return
    # MSE-based values that work with the normal GRPO flow.

    # def training_step(
    #     self,
    #     prompts: DataProto,
    #     iteration: int,
    # ) -> Dict[str, Any]:
    #     """
    #     GRPO training step for NLA.
    #
    #     Flow:
    #     1. Generate N responses per prompt with activation injection
    #     2. Critic predicts activation vectors from all N*batch_size responses
    #     3. Compute MSE-based rewards for each response
    #     4. Calculate GRPO advantages within each group of N
    #     5. Train critic with supervised loss on all responses
    #     6. Train actor with PPO using GRPO advantages
    #     """
    #     metrics = {}
    #
    #     # Step 1: Generate N trajectories per prompt
    #     with self.timer("generate_multiple_trajectories"):
    #         responses = self._generate_multiple_trajectories(prompts)
    #
    #     # Step 2 & 3: Compute rewards using critic
    #     with self.timer("compute_rewards"):
    #         responses = self._compute_rewards_from_critic(
    #             data=responses,
    #             critic_worker=self.critic_worker,
    #         )
    #
    #     # Step 4: Compute GRPO advantages
    #     with self.timer("compute_grpo_advantages"):
    #         responses = self.compute_advantage_and_returns(responses)
    #
    #     # Step 5: Train critic with supervised objective
    #     with self.timer("train_critic"):
    #         critic_metrics = self._train_critic_supervised(
    #             data=responses,
    #             critic_worker=self.critic_worker,
    #         )
    #         metrics.update(critic_metrics)
    #
    #     # Step 6: Train actor with PPO using GRPO advantages
    #     with self.timer("train_actor"):
    #         actor_metrics = self._train_actor_with_ppo(
    #             data=responses,
    #             actor_worker=self.actor_rollout_ref_worker,
    #         )
    #         metrics.update(actor_metrics)
    #
    #     # Add GRPO-specific metrics
    #     if self.grpo_metrics["avg_mse"]:
    #         metrics["grpo/avg_mse"] = sum(self.grpo_metrics["avg_mse"]) / len(self.grpo_metrics["avg_mse"])
    #     if self.grpo_metrics["avg_reward"]:
    #         metrics["grpo/avg_reward"] = sum(self.grpo_metrics["avg_reward"]) / len(self.grpo_metrics["avg_reward"])
    #     if self.grpo_metrics["critic_loss"]:
    #         metrics["grpo/critic_loss"] = sum(self.grpo_metrics["critic_loss"]) / len(self.grpo_metrics["critic_loss"])
    #     if self.grpo_metrics["group_advantage_std"]:
    #         metrics["grpo/group_advantage_std"] = sum(self.grpo_metrics["group_advantage_std"]) / len(self.grpo_metrics["group_advantage_std"])
    #
    #     metrics["grpo/num_trajectories"] = self.grpo_config.num_trajectories_per_prompt
    #
    #     # Clear metrics for next iteration
    #     for key in self.grpo_metrics:
    #         self.grpo_metrics[key] = []
    #
    #     return metrics
