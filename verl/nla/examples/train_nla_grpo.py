#!/usr/bin/env python
"""
Example script for training NLA with GRPO.

This script demonstrates how to:
1. Set up NLA datasets with activation vectors
2. Configure the GRPO trainer for autoencoder-style training
3. Use custom workers for activation injection
4. Train both actor and critic with the GRPO objective
"""

import torch
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.config import DictConfig as VerlDictConfig

# Import NLA components
from verl.nla.data.nla_rl_dataset import create_nla_rl_dataset
from verl.nla.trainer.nla_grpo_trainer import NLAGRPOTrainer, GRPOTrainerConfig
from verl.nla.workers.nla_actor_worker import create_nla_actor_worker
from verl.nla.integration.dataproto_adapter import NLADataProtoAdapter


def create_config() -> DictConfig:
    """Create configuration for NLA GRPO training."""
    config = OmegaConf.create({
        # Model configuration
        "model": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "activation_dim": 768,
            "injection": {
                "mode": "replace",
                "layer_indices": [0],
                "injection_token": "<INJECT>",
                "injection_token_id": None,  # Will be auto-assigned
                "projection_dim": None,
            }
        },

        # GRPO configuration
        "grpo": {
            "num_trajectories_per_prompt": 4,
            "group_normalize_advantages": True,
            "critic_supervised_weight": 1.0,
            "critic_learning_rate": 5e-5,
            "critic_train_epochs": 1,
            "reward_normalize": False,
            "reward_transform": "negative",
            "reward_scale": 1.0,
            "activation_dim": 768,
            "use_pooling": "last",
        },

        # Data configuration
        "data": {
            "train_files": ["data/nla_train.parquet"],
            "val_files": ["data/nla_val.parquet"],
            "train_batch_size": 32,
            "val_batch_size": 32,
            "prompt_key": "prompt",
            "activation_vector_key": "activation_vector",
            "max_prompt_length": 512,
            "injection_token": "<INJECT>",
            "injection_position": "end",
            "activation_dim": 768,
        },

        # Training configuration
        "trainer": {
            "total_epochs": 3,
            "total_training_steps": None,
            "save_freq": 100,
            "val_freq": 50,
            "project_name": "nla_grpo",
            "experiment_name": "autoencoder_training",
            "device": "cuda",
        },

        # Algorithm configuration
        "algorithm": {
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
        },

        # Actor/Rollout configuration
        "actor_rollout_ref": {
            "model": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "lora_rank": 0,  # Can use LoRA if needed
            },
            "rollout": {
                "n": 1,  # This is overridden by GRPO's num_trajectories_per_prompt
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
                "do_sample": True,
                "max_new_tokens": 128,
            },
            "actor": {
                "optim": {
                    "lr": 1e-5,
                    "weight_decay": 0.01,
                },
            },
        },

        # Critic configuration
        "critic": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "pooling": "last",
            "dropout": 0.1,
            "projection_layers": 2,
        },

        # Resource configuration
        "resource_pool": {
            "actor_rollout": [4],  # 4 GPUs for actor
            "critic": [2],  # 2 GPUs for critic
        },
    })

    return config


def setup_tokenizer(model_name: str, injection_token: str):
    """Set up tokenizer with injection token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add injection token
    if injection_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [injection_token]})
        print(f"Added injection token '{injection_token}' to tokenizer")

    return tokenizer


def create_datasets(config: DictConfig, tokenizer):
    """Create NLA datasets with activation vectors."""
    train_dataset = create_nla_rl_dataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        config=OmegaConf.to_container(config.data),
    )

    val_dataset = create_nla_rl_dataset(
        data_files=config.data.val_files,
        tokenizer=tokenizer,
        config=OmegaConf.to_container(config.data),
    )

    return train_dataset, val_dataset


def create_worker_mapping(config: DictConfig):
    """
    Create role-to-worker mapping with NLA customization.

    This is where we inject our custom NLA actor worker.
    """
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.fsdp_workers import CriticWorker

    # Create NLA-wrapped actor worker
    nla_actor_worker = create_nla_actor_worker(
        base_worker_cls=ActorRolloutRefWorker,
        config=OmegaConf.to_container(config),
    )

    role_worker_mapping = {
        Role.ActorRollout: nla_actor_worker,
        Role.Critic: CriticWorker,  # Standard critic worker (GRPO will customize it)
    }

    return role_worker_mapping


def create_resource_pool_manager(config: DictConfig) -> ResourcePoolManager:
    """Create resource pool manager for distributed training."""
    resource_pool_spec = {
        "actor_rollout": config.resource_pool.actor_rollout,
        "critic": config.resource_pool.critic,
    }

    mapping = {
        Role.ActorRollout: "actor_rollout",
        Role.Critic: "critic",
    }

    return ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )


def main():
    """Main training function."""
    print("Starting NLA GRPO Training")

    # Create configuration
    config = create_config()

    # Set up tokenizer
    tokenizer = setup_tokenizer(
        model_name=config.model.model_name,
        injection_token=config.model.injection.injection_token,
    )

    # Update config with injection token ID
    injection_token_id = tokenizer.convert_tokens_to_ids(config.model.injection.injection_token)
    config.model.injection.injection_token_id = injection_token_id

    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Create worker mapping
    role_worker_mapping = create_worker_mapping(config)

    # Create resource pool manager
    resource_pool_manager = create_resource_pool_manager(config)

    # Create GRPO configuration
    grpo_config = GRPOTrainerConfig(
        num_trajectories_per_prompt=config.grpo.num_trajectories_per_prompt,
        group_normalize_advantages=config.grpo.group_normalize_advantages,
        critic_supervised_weight=config.grpo.critic_supervised_weight,
        critic_learning_rate=config.grpo.critic_learning_rate,
        critic_train_epochs=config.grpo.critic_train_epochs,
        reward_normalize=config.grpo.reward_normalize,
        reward_transform=config.grpo.reward_transform,
        reward_scale=config.grpo.reward_scale,
        activation_dim=config.grpo.activation_dim,
        use_pooling=config.grpo.use_pooling,
    )

    # Create NLA GRPO trainer
    trainer = NLAGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        grpo_config=grpo_config,
    )

    # Initialize workers
    print("Initializing distributed workers...")
    trainer.init_workers()

    # Start training
    print("Starting GRPO training loop...")
    trainer.fit()

    print("Training completed!")


if __name__ == "__main__":
    # Initialize Ray if needed
    import ray
    if not ray.is_initialized():
        ray.init()

    main()