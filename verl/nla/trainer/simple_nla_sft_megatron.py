#!/usr/bin/env python
"""
Simple NLA SFT trainer for Megatron backend.
Uses verl's Engine abstraction - works with both FSDP and Megatron via config.

This is a minimal SFT trainer that:
1. Uses standard SFT loss (no RL, no complex workers)
2. Loads NLA datasets (with activation vectors)
3. Uses Megatron backend via Engine abstraction
"""

import hydra
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


class NLASFTTrainer(SFTTrainer):
    """
    NLA SFT trainer that extends base SFTTrainer.

    Key difference: Uses NLA datasets which include activation vectors.
    The base SFTTrainer + Megatron Engine handles everything else.
    """

    def _build_dataset(self):
        """Override to use NLA dataset instead of standard SFT dataset."""
        from verl.nla.data.nla_sft_dataset import NLASFTDataset

        config = self.config
        tokenizer = self.model_config.tokenizer

        # Get model config to determine activation_dim
        from transformers import AutoConfig

        model_path = config.model.path
        model_config = AutoConfig.from_pretrained(model_path)

        # NLA dataset config - matches what's in the parquet files
        dataset_config = OmegaConf.create({
            "prompt_key": config.data.get("prompt_key", "prompt"),
            "response_key": config.data.get("response_key", "response"),
            "activation_vector_key": config.data.get("activation_vector_key", "activation_vector"),
            "max_prompt_length": config.data.get("max_prompt_length", 512),
            "max_response_length": config.data.get("max_response_length", 512),
            "max_length": config.data.get("max_length", 1024),
            "activation_dim": model_config.hidden_size,
            "truncation": config.data.get("truncation", "error"),
        })

        # Create NLA datasets (mode="actor" for standard SFT)
        train_dataset = NLASFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=tokenizer,
            config=dataset_config,
            mode="actor",  # Actor mode: prompt + response with injection
        )

        val_dataset = None
        if config.data.val_files:
            val_dataset = NLASFTDataset(
                parquet_files=config.data.val_files,
                tokenizer=tokenizer,
                config=dataset_config,
                mode="actor",
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


def run_nla_sft(config):
    """Run NLA SFT training with Megatron backend."""
    initialize_global_process_group()
    trainer = NLASFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="../config", config_name="nla_sft_megatron", version_base=None)
def main(config):
    """Main entry point for NLA SFT training."""
    print("=" * 80)
    print("NLA SFT Training with Megatron Backend")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    run_nla_sft(config)


if __name__ == "__main__":
    main()
