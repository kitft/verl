#!/usr/bin/env python
"""Entry point for NLA Megatron SFT trainer.

This script wraps the NLA Megatron SFT trainer and should be run with torchrun:

    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
        --config-name=nla_sft_30b_8gpu_megatron \
        nla.train_mode=actor
"""

import hydra
from omegaconf import DictConfig

from verl.nla.trainer.nla_megatron_sft_trainer import run_nla_megatron_sft


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config: DictConfig):
    """Main entry point for NLA Megatron SFT training."""
    run_nla_megatron_sft(config)


if __name__ == "__main__":
    main()
