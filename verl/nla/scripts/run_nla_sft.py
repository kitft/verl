#!/usr/bin/env python
"""Launch script for NLA SFT training."""

import os
import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

from verl.nla.data.nla_sft_dataset import NLASFTDataset, NLASFTCollator
from verl.nla.trainer.nla_sft_trainer import NLASFTTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NLA models with SFT")
    parser.add_argument(
        "--config",
        type=str,
        default="verl/trainer/config/nla_sft_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["actor", "critic", "both"],
        default="both",
        help="Training mode"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="World size for distributed training"
    )
    return parser.parse_args()


def setup_distributed(args):
    """Setup distributed training."""
    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=args.local_rank,
            world_size=args.world_size
        )
        torch.cuda.set_device(args.local_rank)


def create_datasets(config: DictConfig, tokenizer, train_mode: str):
    """Create training and validation datasets."""
    # Training dataset
    train_dataset = NLASFTDataset(
        parquet_files=config.data.train_dataset.parquet_files,
        tokenizer=tokenizer,
        config=config.data.train_dataset,
        mode=train_mode
    )

    # Validation dataset (optional)
    val_dataset = None
    if config.data.get("val_dataset"):
        val_dataset = NLASFTDataset(
            parquet_files=config.data.val_dataset.parquet_files,
            tokenizer=tokenizer,
            config=config.data.val_dataset,
            mode=train_mode
        )

    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)

    # Override train mode if specified
    if args.train_mode:
        config.trainer.train_mode = args.train_mode

    # Setup distributed training if needed
    if args.world_size > 1:
        setup_distributed(args)
        config.trainer.world_size = args.world_size
        config.trainer.local_rank = args.local_rank

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        config,
        tokenizer,
        config.trainer.train_mode
    )

    # Create device meshes for FSDP
    # Simplified version - in production would use proper mesh creation
    from torch.distributed.device_mesh import DeviceMesh
    device_mesh = DeviceMesh("cuda", list(range(args.world_size)))
    ulysses_device_mesh = None  # Optional for Ulysses parallelism

    # Initialize trainer
    trainer = NLASFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_mode=config.trainer.train_mode
    )

    # Training loop
    print(f"Starting NLA SFT training in {config.trainer.train_mode} mode")
    print(f"Total training steps: {config.trainer.total_training_steps}")

    from torch.utils.data import DataLoader

    # Create data collator
    collator = NLASFTCollator(pad_token_id=tokenizer.pad_token_id)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Set to 0 for debugging
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )

    # Training loop
    global_step = 0
    for epoch in range(config.trainer.num_epochs):
        print(f"Epoch {epoch + 1}/{config.trainer.num_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            # Training step
            metrics = trainer.training_step(batch)

            # Logging
            if global_step % config.trainer.logging_steps == 0:
                print(f"Step {global_step}: {metrics}")

            # Validation
            if val_dataloader and global_step % config.trainer.val_steps == 0:
                val_metrics = {}
                for val_batch in val_dataloader:
                    batch_metrics = trainer.validation_step(val_batch)
                    for k, v in batch_metrics.items():
                        if k not in val_metrics:
                            val_metrics[k] = []
                        val_metrics[k].append(v)

                # Average validation metrics
                val_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}
                print(f"Validation at step {global_step}: {val_metrics}")

            # Checkpointing
            if global_step % config.trainer.save_steps == 0:
                checkpoint_dir = Path(config.trainer.save_dir) / f"step_{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save models
                if config.trainer.train_mode in ["actor", "both"]:
                    torch.save(
                        trainer.fsdp_model.state_dict(),
                        checkpoint_dir / "actor_model.pt"
                    )

                if config.trainer.train_mode in ["critic", "both"]:
                    torch.save(
                        trainer.fsdp_critic.state_dict(),
                        checkpoint_dir / "critic_model.pt"
                    )

                print(f"Saved checkpoint at step {global_step}")

            global_step += 1
            if global_step >= config.trainer.total_training_steps:
                break

        if global_step >= config.trainer.total_training_steps:
            break

    print("Training completed!")


if __name__ == "__main__":
    main()