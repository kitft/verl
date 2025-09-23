#!/usr/bin/env python
"""
Main entry point for NLA GRPO training.
Based on main_ppo.py but adapted for NLA with autoencoder critic.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.nla.data.nla_rl_dataset import create_nla_rl_dataset

# Import NLA components
from verl.nla.trainer.nla_grpo_trainer import GRPOTrainerConfig, NLAGRPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


@hydra.main(config_path="config", config_name="nla_grpo_tiny", version_base=None)
def main(config):
    """Main entry point for NLA training with Hydra configuration management."""
    run_nla_grpo(config)


def run_nla_grpo(config) -> None:
    """Initialize Ray cluster and run distributed NLA GRPO training process."""

    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Get default runtime env and merge with config
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create and run task runner
    runner = NLATaskRunner.remote()
    ray.get(runner.run.remote(config))

    # Optional timeline profiling
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class NLATaskRunner:
    """Ray remote class for executing NLA GRPO training tasks."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker."""
        from verl.single_controller.ray import RayWorkerGroup

        # Check if we should use NLA workers
        if config.get("nla", {}).get("use_nla_datasets", False):
            # Use NLA actor worker for activation injection
            from verl.nla.workers.nla_actor_worker import create_nla_actor_worker

            # Create NLA worker class based on strategy
            base_strategy = config.actor_rollout_ref.actor.strategy
            actor_rollout_cls = create_nla_actor_worker(config)
        else:
            # Use standard workers based on strategy
            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import ActorRolloutRefWorker

                actor_rollout_cls = ActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import ActorRolloutRefWorker

                actor_rollout_cls = ActorRolloutRefWorker
            else:
                raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config):
        """Add critic worker."""
        # Use standard critic workers - NLA logic is handled in the trainer
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker

            critic_cls = CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

            critic_cls = CriticWorker
        else:
            raise NotImplementedError(f"Strategy {config.critic.strategy} not supported")

        self.role_worker_mapping[Role.Critic] = ray.remote(critic_cls)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id

        # Add reward model pool if enabled
        if config.reward_model.enable and config.reward_model.enable_resource_pool:
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def create_nla_datasets(self, config, tokenizer):
        """Create NLA datasets with activation vectors."""
        if not config.get("nla", {}).get("use_nla_datasets", False):
            # Use standard dataset creation
            raise ValueError("Standard dataset creation is not supported for NLA")
            from verl.utils.dataset.rl_dataset import RLHFDataset

            train_dataset = RLHFDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                config=config.data,
            )
            val_dataset = None
            if config.data.val_files:
                val_dataset = RLHFDataset(
                    data_files=config.data.val_files,
                    tokenizer=tokenizer,
                    config=config.data,
                )
        else:
            # Create NLA datasets
            nla_config = config.get("nla", {})
            dataset_config = {
                "prompt_key": config.data.prompt_key,
                "activation_vector_key": nla_config.get("activation_vector_key", "activation_vector"),
                "max_prompt_length": config.data.max_prompt_length,
                "activation_dim": nla_config.get("activation_dim", 768),
                "injection_position": nla_config.get("injection", {}).get("position", "manual"),
            }

            train_dataset = create_nla_rl_dataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                config=dataset_config,
            )

            val_dataset = None
            if config.data.val_files:
                val_dataset = create_nla_rl_dataset(
                    data_files=config.data.val_files,
                    tokenizer=tokenizer,
                    config=dataset_config,
                )

        return train_dataset, val_dataset

    def run(self, config):
        """Execute the main NLA GRPO training workflow."""
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"NLATaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Add workers
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # Add reward model if enabled (for NLA, typically disabled)
        if config.reward_model.enable:
            from verl.trainer.ppo.ray_trainer import Role

            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

        # Download checkpoint
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Setup tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Add simple chat template for tiny model if needed
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{{ messages[-1]['content'] }}"
            print("Set simple chat template for tokenizer")

        # Create datasets
        train_dataset, val_dataset = self.create_nla_datasets(config, tokenizer)
        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"Val dataset size: {len(val_dataset)}")

        # Create reward functions
        if config.reward_model.enable:
            from verl.trainer.ppo.reward import load_reward_manager

            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )
        else:
            # For NLA, rewards come from critic
            reward_fn = None
            val_reward_fn = None

        # Initialize resource pool
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Create sampler
        import torch
        from torch.utils.data import RandomSampler, SequentialSampler

        if config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(config.data.get("seed", 1))
            train_sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            train_sampler = SequentialSampler(data_source=train_dataset)

        # Create collate function
        from verl.utils.dataset.rl_dataset import collate_fn

        # Check if we should use NLA trainer
        if config.get("nla", {}).get("use_nla_datasets", False):
            # Create GRPO config from YAML
            grpo_settings = config.get("nla", {}).get("grpo", {})
            grpo_config = GRPOTrainerConfig(
                num_trajectories_per_prompt=grpo_settings.get("num_trajectories_per_prompt", 4),
                group_normalize_advantages=grpo_settings.get("group_normalize_advantages", True),
                critic_supervised_weight=grpo_settings.get("critic_supervised_weight", 1.0),
                critic_learning_rate=grpo_settings.get("critic_learning_rate", 5e-5),
                critic_train_epochs=grpo_settings.get("critic_train_epochs", 1),
                reward_normalize=grpo_settings.get("reward_normalize", False),
                reward_transform=grpo_settings.get("reward_transform", "negative"),
                reward_scale=grpo_settings.get("reward_scale", 1.0),
                actor_learning_rate=config.actor_rollout_ref.actor.lr,
                ppo_epochs=config.trainer.get("ppo_epochs", 1),
                ppo_clip_ratio=config.trainer.get("clip_ratio", 0.2),
            )

            # Initialize NLA GRPO trainer with all required parameters
            trainer = NLAGRPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                grpo_config=grpo_config,
                ray_worker_group_cls=ray_worker_group_cls,
                processor=processor,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
        else:
            # Use standard PPO trainer
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer

            trainer = RayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )

        # Initialize workers
        trainer.init_workers()

        # Start training
        trainer.fit()


if __name__ == "__main__":
    main()
