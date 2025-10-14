"""GRPO-style trainer for NLA with autoencoder critic.

This trainer implements Group Relative Policy Optimization (GRPO) where:
1. We generate N trajectories per prompt
2. Rewards are computed via critic's activation reconstruction MSE
3. Advantages are calculated within each group of N trajectories
4. Both actor and critic are trained simultaneously
"""

import logging
from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig

# Import NLA evaluation module
from verl.nla.trainer.nla_eval import create_evaluator
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

logger = logging.getLogger(__name__)


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
        print("=== _ValuesAsRewardsShim called ===")
        print(f"Batch size: {data.batch['input_ids'].shape[0] if 'input_ids' in data.batch else 'N/A'}")
        print(f"Has activation_vectors: {'activation_vectors' in data.batch}")
        if "activation_vectors" in data.batch:
            print(f"activation_vectors shape: {data.batch['activation_vectors'].shape}")

        if "values" not in data.batch:
            # Compute values using critic - returns (batch, seq_len) where values = -MSE (higher is better)
            print("Computing values via critic...")
            values_output = self.trainer.critic_wg.compute_values(data)
            # Cache values in batch for reuse
            data.batch.update(values_output.batch)
            print(f"Values computed, shape: {data.batch['values'].shape}")

        # Use values as token_level_scores
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

        # Initialize metadata tracking for RL
        self.best_reward_mean = float("-inf")
        self.final_reward_mean = None
        self.final_reward_std = None
        self.training_start_time = None

        # Initialize NLA string-match evaluator (if enabled in config)
        self.nla_evaluator = None
        try:
            # # Get injection token ID from config
            # injection_token_id = self.config.get("nla", {}).get("injection", {}).get("injection_token_id", None)
            # if injection_token_id is None:
            #     # Try to get from tokenizer if not in config
            #     injection_token = self.config.get("nla", {}).get("injection", {}).get("injection_token", "㊗")
            #     result = tokenizer.convert_tokens_to_ids(injection_token)
            #     # Handle both single token (int) and potential list return
            #     injection_token_id = result[0] if isinstance(result, list) else result

            #     # Validate token ID is not unknown token
            #     if injection_token_id == tokenizer.unk_token_id:
            #         logger.warning(
            #             f"Injection token '{injection_token}' not found in vocabulary. "
            #             f"Evaluator may not work correctly. Consider adding token to tokenizer."
            #         )

            self.nla_evaluator = create_evaluator(
                config=config,
                tokenizer=tokenizer,
            )
        except FileNotFoundError as e:
            logger.warning(f"NLA evaluator disabled: eval file not found: {e}")
        except KeyError as e:
            logger.warning(f"NLA evaluator disabled: missing config key: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing NLA evaluator: {e}", exc_info=True)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Override to include activation_vectors in generation batch."""

        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # Pop standard keys but NOT activation_vectors (leave them in original batch for critic)
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # Copy activation_vectors to gen_batch (don't pop from original)
        # Note: activation_vectors will be repeated/expanded along with gen_batch by the base trainer
        if "activation_vectors" in batch.batch:
            gen_batch.batch["activation_vectors"] = batch.batch["activation_vectors"]

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

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

    def _run_nla_string_match_eval(self) -> dict:
        """
        Run NLA string-match evaluation.

        Generates completions with activation injection and checks if expected
        strings appear in the output.

        Returns:
            Dictionary of evaluation metrics, or empty dict if evaluator not available
        """
        if self.nla_evaluator is None:
            return {}

        try:
            # Get evaluation config parameters
            eval_config = self.config.get("nla_eval", {})
            max_new_tokens = eval_config.get("max_new_tokens", 50)
            temperature = eval_config.get("temperature", 0.7)
            do_sample = eval_config.get("do_sample", True)
            num_generations = eval_config.get("num_generations_per_prompt", 1)

            # Run evaluation
            eval_metrics = self.nla_evaluator.evaluate(
                actor_rollout_wg=self.actor_rollout_wg,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_generations_per_prompt=num_generations,
                global_steps=self.global_steps,
                experiment_name=self.config.trainer.experiment_name,
            )

            return eval_metrics

        except Exception as e:
            logger.error(f"NLA evaluation failed: {e}", exc_info=True)
            return {}

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

    def fit(self):
        """Override fit to track training time and metrics."""
        import signal
        import sys
        import time

        import verl.utils.tracking

        self.training_start_time = time.time()

        # Setup signal handler for Ctrl-C to save checkpoint before exiting
        def signal_handler(sig, frame):
            print("\n[SIGINT] Ctrl-C detected. Checking if checkpoint should be saved...")
            if self.global_steps >= 100:
                print(f"[SIGINT] Training has progressed to step {self.global_steps}. Saving checkpoint...")
                try:
                    self._save_checkpoint()
                    print("[SIGINT] Checkpoint saved successfully.")
                except Exception as e:
                    print(f"[SIGINT] Failed to save checkpoint: {e}")
            else:
                print(f"[SIGINT] Only {self.global_steps} steps completed (< 100). Skipping checkpoint save.")
            print("[SIGINT] Exiting...")
            sys.exit(0)

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        # Monkey-patch Tracking.log to capture metrics
        # This is necessary because parent creates its own logger internally
        original_log_method = verl.utils.tracking.Tracking.log

        def wrapped_log(logger_self, data, step):
            # Extract reward metrics if present (before calling original)
            if "critic/rewards/mean" in data:
                reward_mean = data["critic/rewards/mean"]
                self.final_reward_mean = reward_mean
                if reward_mean > self.best_reward_mean:
                    self.best_reward_mean = reward_mean

            # Try to get std, or approximate from range
            if "critic/rewards/std" in data:
                self.final_reward_std = data["critic/rewards/std"]
            elif "critic/rewards/max" in data and "critic/rewards/min" in data:
                reward_range = data["critic/rewards/max"] - data["critic/rewards/min"]
                self.final_reward_std = reward_range / 4.0  # Rough approximation

            # Run NLA string-match evaluation if enabled and at right frequency
            if self.nla_evaluator is not None:
                eval_config = self.config.get("nla_eval", {})
                eval_freq = eval_config.get("freq", 100)  # Default: every 100 steps

                # Check if it's time to evaluate or if it's the final step
                is_final_step = step >= self.total_training_steps
                should_eval = (step % eval_freq == 0) or is_final_step or step == 1

                if should_eval:
                    start_time = time.time()
                    logger.info(f"Running NLA string-match evaluation at step {step}...")
                    eval_metrics = self._run_nla_string_match_eval()
                    end_time = time.time()
                    time_taken = end_time - start_time
                    logger.info(f"NLA string-match evaluation completed in {time_taken:.2f} seconds")
                    eval_metrics["nla_eval/evaluation_time"] = end_time - start_time
                    if eval_metrics:
                        data.update(eval_metrics)
                        accuracy = eval_metrics.get("nla_eval/string_match_accuracy")
                        if accuracy is not None:
                            logger.info(f"NLA eval completed: {accuracy:.2%} accuracy")
                        else:
                            logger.warning("NLA eval completed but no accuracy metric returned")

            # Call original
            return original_log_method(logger_self, data, step)

        # Temporarily replace the method
        verl.utils.tracking.Tracking.log = wrapped_log

        try:
            # Call parent's fit (which uses the patched logger)
            super().fit()
        finally:
            # Restore original method
            verl.utils.tracking.Tracking.log = original_log_method

    def _save_checkpoint(self):
        """Override to save NLA metadata with RL checkpoint."""
        import os
        import time

        import torch

        from verl.utils.checkpoint.nla_metadata import (
            create_rl_metadata_from_config,
            save_metadata,
        )
        from verl.utils.fs import local_mkdir_safe

        # Prepare metadata before calling parent
        training_time_hours = None
        if self.training_start_time is not None:
            training_time_hours = (time.time() - self.training_start_time) / 3600.0

        # Get WandB info if available
        wandb_run_id = None
        wandb_run_url = None
        try:
            import wandb

            if wandb.run is not None:
                wandb_run_id = wandb.run.id
                wandb_run_url = wandb.run.get_url()
        except Exception:
            pass

        # Extract lineage from SFT checkpoints if they exist
        actor_checkpoint_path = self.config.actor_rollout_ref.model.get("path")
        critic_checkpoint_path = self.config.critic.model.get("path")

        # Create metadata with lineage
        metadata = create_rl_metadata_from_config(
            config=self.config,
            global_step=self.global_steps,
            actor_checkpoint_path=actor_checkpoint_path,
            critic_checkpoint_path=critic_checkpoint_path,
            final_reward_mean=self.final_reward_mean,
            final_reward_std=self.final_reward_std,
            best_reward_mean=self.best_reward_mean if self.best_reward_mean != float("-inf") else None,
            training_time_hours=training_time_hours,
            wandb_run_id=wandb_run_id,
            wandb_run_url=wandb_run_url,
        )

        # Duplicate parent logic to insert metadata save at the right point
        # (before workers handle HDFS copy internally)
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {checkpoint_dir}")
        actor_local_path = os.path.join(checkpoint_dir, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # CRITICAL: Save metadata BEFORE worker checkpoints (which handle HDFS copy)
        # This ensures metadata is in the folder when workers copy to HDFS
        local_mkdir_safe(checkpoint_dir)
        save_metadata(checkpoint_dir, metadata)
        print(f"Saved NLA RL metadata for step {self.global_steps}")

        # Now save actor/critic checkpoints (workers will copy entire folder to HDFS)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(checkpoint_dir, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # Save dataloader
        local_mkdir_safe(checkpoint_dir)
        dataloader_local_path = os.path.join(checkpoint_dir, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # Latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # Cleanup old checkpoints based on retention policy
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """
        Cleanup old checkpoints to save disk space.

        Retention policy:
        - Keep the last 3 checkpoints (rolling window)
        - Keep every 10th checkpoint permanently (milestones)

        Example with save_freq=100:
        - Milestones: steps 1000, 2000, 3000, ... (kept permanently)
        - Rolling: last 3 checkpoints regardless of step
        """
        import glob
        import re
        import shutil
        from pathlib import Path

        save_freq = self.config.trainer.save_freq
        if save_freq <= 0:
            return  # No cleanup if checkpointing is disabled

        checkpoint_base_dir = Path(self.config.trainer.default_local_dir)
        if not checkpoint_base_dir.exists():
            return

        # Find all checkpoint directories
        checkpoint_pattern = str(checkpoint_base_dir / "global_step_*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)

        if len(checkpoint_dirs) <= 3:
            return  # Keep all if we have 3 or fewer

        # Extract step numbers and sort
        checkpoint_steps = []
        for dir_path in checkpoint_dirs:
            match = re.search(r"global_step_(\d+)", dir_path)
            if match:
                step = int(match.group(1))
                checkpoint_steps.append((step, dir_path))

        checkpoint_steps.sort()

        # Determine which checkpoints to keep
        milestone_interval = 10 * save_freq
        keep_steps = set()

        # Keep milestones (every 10x save_freq)
        for step, _ in checkpoint_steps:
            if step % milestone_interval == 0:
                keep_steps.add(step)

        # Keep last 3 checkpoints
        for step, _ in checkpoint_steps[-3:]:
            keep_steps.add(step)

        # Delete checkpoints not in keep set
        deleted_count = 0
        for step, dir_path in checkpoint_steps:
            if step not in keep_steps:
                try:
                    shutil.rmtree(dir_path)
                    print(f"[Checkpoint Cleanup] Deleted old checkpoint: {dir_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[Checkpoint Cleanup] Failed to delete {dir_path}: {e}")

        if deleted_count > 0:
            print(
                f"[Checkpoint Cleanup] Deleted {deleted_count} old checkpoint(s). "
                f"Kept {len(keep_steps)} checkpoints (last 3 + milestones every {milestone_interval} steps)"
            )
