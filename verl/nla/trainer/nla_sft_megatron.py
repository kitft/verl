#!/usr/bin/env python
"""
NLA SFT Trainer with Megatron backend and activation injection.

Uses NLA Megatron actor for activation injection via forward hooks.
Simpler than RL - just prompt→response pairs with standard SFT loss.
"""

import os
from functools import partial

import hydra
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.nla.data.nla_sft_dataset import NLASFTDataset
from verl.nla.workers.nla_megatron_worker import NLAMegatronActorRolloutRefWorker
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.tracking import Tracking


class NLAMegatronSFTTrainer:
    """
    SFT trainer for NLA with Megatron backend.

    Uses NLA Megatron actor worker which handles:
    - Activation injection via forward hooks
    - Tensor/pipeline parallelism
    - Distributed optimizer
    """

    def __init__(self, config):
        self.config = config
        self.rank = torch.distributed.get_rank()

        # Initialize Megatron parallel groups BEFORE creating worker
        self._init_megatron_parallel()

        # Build datasets
        self._build_datasets()

        # Build NLA Megatron worker (handles model + injection)
        self._build_worker()

        # Build dataloader
        self._build_dataloader()

        # Setup loss function
        from verl.workers.roles.utils.losses import sft_loss
        self.loss_fn = partial(sft_loss, config=None)

    def _init_megatron_parallel(self):
        """Initialize Megatron model parallel groups."""
        from megatron.core import parallel_state as mpu

        # Check if already initialized
        if torch.distributed.is_initialized():
            try:
                # Try to get group - if this works, already initialized
                mpu.get_tensor_model_parallel_group()
                print(f"[Rank {self.rank}] Megatron parallel groups already initialized")
                return
            except:
                pass  # Not initialized, continue

        # Get parallelism settings from config
        tensor_mp_size = self.config.actor.megatron.tensor_model_parallel_size
        pipeline_mp_size = self.config.actor.megatron.pipeline_model_parallel_size
        virtual_pipeline_mp_size = self.config.actor.megatron.get("virtual_pipeline_model_parallel_size", None)
        context_parallel_size = self.config.actor.megatron.get("context_parallel_size", 1)
        expert_mp_size = self.config.actor.megatron.get("expert_model_parallel_size", 1)
        expert_tensor_parallel_size = self.config.actor.megatron.get("expert_tensor_parallel_size", 1)

        print(f"[Rank {self.rank}] Initializing Megatron parallel groups:")
        print(f"  TP={tensor_mp_size}, PP={pipeline_mp_size}, VPP={virtual_pipeline_mp_size}")
        print(f"  CP={context_parallel_size}, EP={expert_mp_size}, ETP={expert_tensor_parallel_size}")

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tensor_mp_size,
            pipeline_model_parallel_size=pipeline_mp_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_mp_size,
            use_sharp=False,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_mp_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
        )

    def _build_datasets(self):
        """Build NLA datasets with activation vectors."""
        from transformers import AutoConfig, AutoTokenizer

        model_path = self.config.model.path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_config = AutoConfig.from_pretrained(model_path)

        # NLA dataset config
        dataset_config = OmegaConf.create({
            "prompt_key": self.config.data.get("prompt_key", "prompt"),
            "response_key": self.config.data.get("response_key", "response"),
            "activation_vector_key": self.config.data.get("activation_vector_key", "activation_vector"),
            "max_prompt_length": self.config.data.get("max_prompt_length", 512),
            "max_response_length": self.config.data.get("max_response_length", 512),
            "max_length": self.config.data.get("max_length", 1024),
            "activation_dim": model_config.hidden_size,
            "truncation": self.config.data.get("truncation", "error"),
        })

        # Create datasets (mode="actor" for SFT with injection)
        self.train_dataset = NLASFTDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=tokenizer,
            config=dataset_config,
            mode="actor",
        )

        self.val_dataset = None
        if self.config.data.get("val_files"):
            self.val_dataset = NLASFTDataset(
                parquet_files=self.config.data.val_files,
                tokenizer=tokenizer,
                config=dataset_config,
                mode="actor",
            )

    def _build_worker(self):
        """Build NLA Megatron worker for activation injection."""
        # Create worker config that matches what Ray workers expect
        worker_config = self.config

        # Initialize worker
        self.worker = NLAMegatronActorRolloutRefWorker(
            config=worker_config,
            role="actor",  # SFT only needs actor
        )

        # Initialize model
        self.worker.init_model()

    def _build_dataloader(self):
        """Build distributed dataloader (matching RL trainer pattern)."""
        from verl.utils.device import get_device_name
        from verl.nla.data.nla_sft_dataset import NLASFTCollatorWithRLFields

        # Get tokenizer for collator with RL fields (for update_actor path)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.path)
        collate_fn = NLASFTCollatorWithRLFields(pad_token_id=tokenizer.pad_token_id)

        device_name = get_device_name()

        # Use global batch size - worker will handle DP splitting internally
        # (like RL trainer does)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            # Use same collator with RL fields for validation
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.config.data.train_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                pin_memory_device=device_name,
            )

    def fit(self):
        """Run training loop."""
        is_logging = self.rank == 0

        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        total_epochs = self.config.trainer.total_epochs
        steps_per_epoch = len(self.train_dataloader)
        total_steps = total_epochs * steps_per_epoch

        save_freq = self.config.trainer.get("save_freq", steps_per_epoch)
        test_freq = self.config.trainer.get("test_freq", steps_per_epoch)

        global_step = 0

        print(f"[Rank {self.rank}] Starting training: {total_epochs} epochs, {steps_per_epoch} steps/epoch")

        for epoch in range(total_epochs):
            for step_in_epoch, batch_dict in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Epoch {epoch + 1}/{total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # Convert batch dict to DataProto (like RL trainer does)
                from verl.protocol import DataProto

                data = DataProto.from_single_dict(batch_dict)

                # Add meta_info for micro-batching
                # Count tokens per sample for performance metrics (flops counter needs list of lengths)
                if "response_mask" in data.batch:
                    # Sum per sample (dim=1) to get list of token counts
                    global_token_num = data.batch["response_mask"].sum(dim=1).cpu().tolist()
                else:
                    global_token_num = data.batch["attention_mask"].sum(dim=1).cpu().tolist()

                data.meta_info = {
                    "micro_batch_size": self.config.data.micro_batch_size_per_gpu,
                    "use_dynamic_bsz": self.config.data.get("use_dynamic_bsz", True),
                    "max_token_len_per_gpu": self.config.data.get("max_token_len_per_gpu", 4096),
                    "temperature": 1.0,
                    "global_batch_size": self.config.data.train_batch_size,
                    "global_token_num": global_token_num,  # List of per-sample token counts
                }

                # Training step with activation injection
                # Note: We use update_actor() which calls actor.update_policy() with dummy old_log_probs (all zeros)
                # and disabled clipping to make it equivalent to SFT
                with Timer(name="train_step", logger=None) as timer:
                    output = self.worker.update_actor(data)

                # Log metrics
                if is_logging and output.meta_info and "metrics" in output.meta_info:
                    metrics = output.meta_info["metrics"]
                    metrics["train/step"] = global_step
                    metrics["train/epoch"] = epoch
                    tracking.log(data=metrics, step=global_step)

                # Validation
                if test_freq > 0 and global_step % test_freq == 0:
                    self._validate(global_step, tracking if is_logging else None)

                # Save checkpoint
                if save_freq > 0 and global_step % save_freq == 0:
                    self._save_checkpoint(global_step)

                if global_step >= total_steps:
                    break

            if global_step >= total_steps:
                break

        print(f"[Rank {self.rank}] Training complete!")

    def _validate(self, global_step, tracking=None):
        """Run validation."""
        if not self.val_dataset:
            return

        from verl.protocol import DataProto

        val_losses = []
        for batch_dict in self.val_dataloader:
            data = DataProto.from_single_dict(batch_dict)
            data.meta_info = {
                "micro_batch_size": self.config.data.micro_batch_size_per_gpu,
                "use_dynamic_bsz": self.config.data.get("use_dynamic_bsz", True),
                "max_token_len_per_gpu": self.config.data.get("max_token_len_per_gpu", 4096),
                "temperature": 1.0,
            }

            # Compute validation loss (no gradient update)
            output = self.worker.compute_log_prob(data)
            if output.meta_info and "metrics" in output.meta_info:
                val_losses.append(output.meta_info["metrics"].get("loss", 0.0))

        if val_losses and tracking:
            avg_val_loss = sum(val_losses) / len(val_losses)
            tracking.log(data={"val/loss": avg_val_loss}, step=global_step)

    def _save_checkpoint(self, global_step):
        """Save checkpoint."""
        checkpoint_dir = self.config.trainer.get("default_local_dir", "./checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}")

        if self.rank == 0:
            print(f"Saving checkpoint to {checkpoint_path}")

        self.worker.save_checkpoint(
            checkpoint_path=checkpoint_path,
            global_step=global_step,
        )


def run_nla_sft_megatron(config):
    """Run NLA SFT with Megatron backend."""
    initialize_global_process_group()
    trainer = NLAMegatronSFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="../../trainer/config", config_name="nla_sft_megatron_30b", version_base=None)
def main(config):
    """Main entry point."""
    print("=" * 80)
    print("NLA SFT Training with Megatron + Activation Injection")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    run_nla_sft_megatron(config)


if __name__ == "__main__":
    main()
