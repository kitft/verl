#!/usr/bin/env python
"""
Example script for training NLA with GRPO using Hydra configuration.

This script demonstrates how to:
1. Set up NLA datasets with activation vectors
2. Configure the GRPO trainer for autoencoder-style training
3. Use custom workers for activation injection
4. Train both actor and critic with the GRPO objective
"""

import os
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
from pathlib import Path

from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.config import DictConfig as VerlDictConfig

# Import NLA components
from verl.nla.data.nla_rl_dataset import create_nla_rl_dataset
from verl.nla.trainer.nla_grpo_trainer import NLAGRPOTrainer, GRPOTrainerConfig
from verl.nla.workers.nla_actor_worker import create_nla_actor_worker
from verl.nla.integration.dataproto_adapter import NLADataProtoAdapter


def setup_tokenizer(model_name: str, injection_token: str = None):
    """Set up tokenizer with injection token."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add injection token if specified (otherwise InjectionManager will auto-select)
    if injection_token and injection_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [injection_token]})
        print(f"Added injection token '{injection_token}' to tokenizer")

    return tokenizer


def create_datasets(cfg: DictConfig, tokenizer):
    """Create NLA datasets with activation vectors."""

    # Convert OmegaConf lists to regular Python lists
    train_files = list(cfg.data.train_dataset.parquet_files)
    val_files = list(cfg.data.val_dataset.parquet_files) if cfg.data.val_dataset.parquet_files else None

    train_dataset = create_nla_rl_dataset(
        data_files=train_files,
        tokenizer=tokenizer,
        config={
            "prompt_key": cfg.data.train_dataset.prompt_key,
            "activation_vector_key": cfg.data.train_dataset.activation_vector_key,
            "max_prompt_length": cfg.data.train_dataset.max_prompt_length,
            "injection_token": cfg.data.train_dataset.injection_token,
            "injection_position": cfg.data.train_dataset.injection_position,
            "activation_dim": cfg.data.train_dataset.activation_dim,
        }
    )

    val_dataset = None
    if val_files:
        val_dataset = create_nla_rl_dataset(
            data_files=val_files,
            tokenizer=tokenizer,
            config={
                "prompt_key": cfg.data.val_dataset.prompt_key,
                "activation_vector_key": cfg.data.val_dataset.activation_vector_key,
                "max_prompt_length": cfg.data.val_dataset.max_prompt_length,
                "injection_token": cfg.data.val_dataset.injection_token,
                "injection_position": cfg.data.val_dataset.injection_position,
                "activation_dim": cfg.data.val_dataset.activation_dim,
            }
        )

    return train_dataset, val_dataset


def setup_resource_pool(cfg: DictConfig):
    """Set up Ray resource pool for distributed training."""
    # Get GPU settings from config
    actor_gpus = cfg.get('distributed', {}).get('actor_gpus', 1)
    critic_gpus = cfg.get('distributed', {}).get('critic_gpus', 1)

    # Create resource pool
    ray_resource_pool = RayResourcePool(
        actor_rollout=[actor_gpus],
        critic=[critic_gpus],
    )

    # Set up worker groups
    manager = ResourcePoolManager(ray_resource_pool)

    # Create actor worker with NLA support
    actor_worker_cls = create_nla_actor_worker(cfg.model)

    # Create worker group
    worker_group = RayWorkerGroup(
        resource_pool=ray_resource_pool,
        ray_actor_cls=actor_worker_cls,
        ray_init_config={"num_gpus": actor_gpus},
    )

    return manager, worker_group


def create_trainer(cfg: DictConfig, train_dataset, val_dataset, tokenizer):
    """Create NLA GRPO trainer."""

    # Convert config to format expected by trainer
    trainer_config = GRPOTrainerConfig(
        # Model settings
        model_name=cfg.model.model_name,
        critic_model_name=cfg.model.critic.model_name,
        activation_dim=cfg.model.actor.get('activation_dim', cfg.data.train_dataset.activation_dim),

        # Training settings
        num_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        batch_size=cfg.data.batch_size,
        eval_batch_size=cfg.data.eval_batch_size,
        learning_rate=cfg.training.learning_rate,

        # GRPO specific
        num_responses_per_prompt=cfg.grpo.num_responses_per_prompt,
        temperature=cfg.grpo.temperature,
        max_new_tokens=cfg.grpo.max_new_tokens,
        use_nla_reward=cfg.grpo.use_nla_reward,
        reconstruction_weight=cfg.grpo.reconstruction_weight,

        # PPO settings
        ppo_epochs=cfg.grpo.ppo_epochs,
        eps_clip=cfg.grpo.eps_clip,
        value_loss_coef=cfg.grpo.value_loss_coef,
        entropy_coef=cfg.grpo.entropy_coef,

        # Output settings
        output_dir=cfg.training.output_dir,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,

        # Debug
        debug=cfg.debug.enabled,
    )

    # Create trainer
    trainer = NLAGRPOTrainer(
        config=trainer_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        nla_config=cfg.model.injection,
    )

    return trainer


@hydra.main(version_base=None, config_path="../configs", config_name="test_tiny_grpo_config")
def main(cfg: DictConfig):
    """Main training function."""
    print("="*60)
    print("Starting NLA GRPO Training with Hydra Configuration")
    print("="*60)

    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = cfg.get('device', 'cuda:0')
    if cfg.get('use_cpu', False):
        device = 'cpu'
    torch.cuda.set_device(device) if 'cuda' in device else None

    # Set up tokenizer
    print(f"\n1. Setting up tokenizer for {cfg.model.model_name}")
    tokenizer = setup_tokenizer(
        model_name=cfg.model.model_name,
        injection_token=cfg.model.injection.injection_token,
    )

    # Update config with injection token ID
    if cfg.model.injection.injection_token:
        injection_token_id = tokenizer.convert_tokens_to_ids(cfg.model.injection.injection_token)
        cfg.model.injection.injection_token_id = injection_token_id
        print(f"   Injection token ID: {injection_token_id}")

    # Create datasets
    print("\n2. Loading datasets")
    train_dataset, val_dataset = create_datasets(cfg, tokenizer)
    print(f"   Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"   Val dataset size: {len(val_dataset)}")

    # Set up Ray if using distributed training
    if cfg.distributed.world_size > 1:
        print("\n3. Setting up Ray for distributed training")
        import ray
        if not ray.is_initialized():
            ray.init()
        manager, worker_group = setup_resource_pool(cfg)
    else:
        print("\n3. Using single-device training")
        manager, worker_group = None, None

    # Create trainer
    print("\n4. Creating NLA GRPO trainer")
    trainer = create_trainer(cfg, train_dataset, val_dataset, tokenizer)

    # Start training
    print("\n5. Starting training")
    print("="*60)

    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        if cfg.distributed.world_size > 1 and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()