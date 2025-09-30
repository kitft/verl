#!/usr/bin/env python
"""
Main entry point for NLA GRPO training.
Based on main_ppo.py but adapted for NLA with autoencoder critic.
"""

import os
import re
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional

import hydra
import ray
from omegaconf import OmegaConf

from verl.nla.data.nla_rl_dataset import create_nla_rl_dataset

# Import NLA components
from verl.nla.trainer.nla_grpo_trainer import GRPOTrainerConfig, NLAGRPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


def _slugify(value: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    """Sanitize the provided text into a wandb-friendly slug."""
    text = str(value).strip() if value is not None else ""
    if not text and fallback is not None:
        text = str(fallback).strip()

    if not text:
        return None

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if slug:
        return slug

    if fallback is not None:
        fallback_slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(fallback)).strip("-").lower()
        return fallback_slug or None

    return None


def _convert_fsdp_to_hf_if_needed(model_path: str) -> str:
    """
    Auto-convert FSDP checkpoint to HuggingFace format if needed.

    Args:
        model_path: Path to model (can be HF format or FSDP checkpoint)

    Returns:
        Path to HuggingFace format model (either existing or newly converted)
    """
    path = Path(model_path)

    # If it's already pointing to huggingface subdirectory, use as-is
    if path.name == "huggingface" or (path / "huggingface").exists():
        if path.name == "huggingface":
            return str(path)
        else:
            return str(path / "huggingface")

    # Check if this is an FSDP checkpoint directory
    # FSDP checkpoints have model_world_size_*_rank_*.pt files and fsdp_config.json
    if path.is_dir():
        has_fsdp_shards = any(path.glob("model_world_size_*_rank_*.pt"))
        has_fsdp_config = (path / "fsdp_config.json").exists()
        has_hf_subdir = (path / "huggingface").exists()

        if has_fsdp_shards and has_fsdp_config:
            # This is an FSDP checkpoint
            if has_hf_subdir:
                print(f"Found existing HuggingFace checkpoint at {path / 'huggingface'}")
                return str(path / "huggingface")
            else:
                print(f"WARNING: FSDP checkpoint at {path} does not contain huggingface/ subdirectory.")
                print("Please re-train with checkpoint.save_contents including 'hf_model', or manually convert.")
                print("For now, attempting to use FSDP checkpoint directly (may fail)...")
                return str(path)

    # Otherwise assume it's a HuggingFace model path (could be local or HF Hub)
    return str(path)


def _detect_profile_label(config) -> Optional[str]:
    """Return a profile label (currently only distinguishes the tiny preset)."""
    # Check model name
    model_path = config.actor_rollout_ref.model.path
    if "tiny" in str(model_path).lower():
        return "tiny"

    # Check total training steps
    total_steps = getattr(config.trainer, "total_training_steps", None)
    if total_steps is not None and total_steps <= 100:
        return "tiny"

    return None


def _detect_dataset_variant(train_files) -> Optional[str]:
    """Infer whether training uses canonical or random activations."""
    if isinstance(train_files, str):
        paths: Iterable[str] = [train_files]
    elif isinstance(train_files, Iterable):
        paths = [str(p) for p in train_files]
    else:
        return None

    paths = list(paths)
    if not paths:
        return None

    if any("_random" in path for path in paths):
        return "random"
    return "canonical"


def _extract_base_model_name(model_path: str) -> Optional[str]:
    """Extract the base model name from checkpoint or model path."""
    from pathlib import Path
    import json

    # If it's a checkpoint path, try to read config.json
    path_obj = Path(model_path)
    if path_obj.exists():
        config_json = path_obj / "config.json"
        if config_json.exists():
            try:
                with open(config_json) as f:
                    model_config = json.load(f)
                    base_name = model_config.get("_name_or_path")
                    if base_name and not base_name.startswith(".") and "checkpoint" not in base_name.lower():
                        # Extract just the model name (e.g., "Qwen2.5-0.5B-Instruct" from full path)
                        return base_name.split("/")[-1] if "/" in base_name else base_name
            except Exception:
                pass

    # If path looks like a checkpoint, extract meaningful parts
    # e.g., "checkpoints/nla_sft_full/canonical/global_step_156/huggingface" -> "nla-sft-full"
    path_str = str(model_path)
    if "checkpoint" in path_str.lower() or "global_step" in path_str:
        parts = path_str.split("/")
        for i, part in enumerate(parts):
            if "checkpoint" in part.lower() and i + 1 < len(parts):
                # Return the next meaningful directory name
                return parts[i + 1]

    # Otherwise just return the last component
    return path_str.split("/")[-1] if "/" in path_str else path_str


def _build_experiment_name(config, user_label: Optional[str]) -> str:
    """Construct a descriptive run name for tracking backends."""
    model_path = config.actor_rollout_ref.model.path
    model_id = _extract_base_model_name(str(model_path))

    # Get training mode from NLA config (actor or critic focus)
    mode = config.get("nla", {}).get("train_mode", "rl")

    profile_label = _detect_profile_label(config)
    dataset_variant = _detect_dataset_variant(config.data.train_files)

    components: List[str] = []
    seen = set()

    def add_component(value: Optional[str], *, fallback: Optional[str] = None) -> None:
        slug = _slugify(value, fallback)
        if slug and slug not in seen:
            components.append(slug)
            seen.add(slug)

    add_component(model_id, fallback="model")
    add_component(mode, fallback="rl")
    add_component(profile_label)
    add_component(dataset_variant)
    add_component(user_label)

    return "-".join(components)


def _ensure_tracking_defaults(config) -> None:
    """Populate default tracking project/run names when missing."""
    project_name = getattr(config.trainer, "project_name", None)
    if not project_name:
        model_path = config.actor_rollout_ref.model.path
        model_id = _extract_base_model_name(str(model_path))
        mode = config.get("nla", {}).get("train_mode", "rl")
        model_slug = _slugify(model_id, fallback="model") or "model"
        mode_slug = _slugify(mode, fallback="rl") or "rl"
        config.trainer.project_name = f"{model_slug}-{mode_slug}-grpo"

    # Check if user provided a custom experiment_name in config
    user_label = getattr(config.trainer, "experiment_name", None)
    # If it looks like one of our auto-generated names or is generic, regenerate
    if not user_label or user_label in ["qwen-tiny-canonical", "nla-grpo", "nla-rl", "grpo_training"]:
        user_label = None

    run_name = _build_experiment_name(config, user_label)
    config.trainer.experiment_name = run_name


@hydra.main(config_path="config", config_name="runs/qwen_tiny/rl_grpo", version_base=None)
def main(config):
    """Main entry point for NLA training with Hydra configuration management."""
    run_nla_grpo(config)


def run_nla_grpo(config) -> None:
    """Initialize Ray cluster and run distributed NLA GRPO training process."""

    # Auto-convert FSDP checkpoints to HuggingFace format if needed
    print("Checking model paths for FSDP→HF conversion...")
    actor_path = config.actor_rollout_ref.model.path
    critic_path = config.critic.model.path

    converted_actor_path = _convert_fsdp_to_hf_if_needed(actor_path)
    converted_critic_path = _convert_fsdp_to_hf_if_needed(critic_path)

    if converted_actor_path != actor_path:
        print(f"Actor model path converted: {actor_path} → {converted_actor_path}")
        config.actor_rollout_ref.model.path = converted_actor_path

    if converted_critic_path != critic_path:
        print(f"Critic model path converted: {critic_path} → {converted_critic_path}")
        config.critic.model.path = converted_critic_path

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

        # NLA only supports FSDP strategy currently
        strategy = config.actor_rollout_ref.actor.strategy
        if strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError(f"NLA only supports FSDP strategy, got '{strategy}'")

        # Use NLA actor worker for activation injection
        from verl.nla.workers.nla_actor_worker import NLAActorRolloutRefWorker

        actor_rollout_cls = NLAActorRolloutRefWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config):
        """Add critic worker."""
        # NLA GRPO uses our custom critic worker with vector value head
        # we have actually disabled the value head linear map for now
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.nla.workers.nla_critic_worker import NLACriticWorker

            critic_cls = NLACriticWorker
        elif config.critic.strategy == "megatron":
            from verl.nla.workers.nla_critic_worker import NLACriticWorker

            critic_cls = NLACriticWorker
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
        # Create NLA datasets
        nla_config = config.get("nla", {})

        # Get model config to determine activation_dim
        from transformers import AutoConfig

        model_path = config.actor_rollout_ref.model.path
        model_config = AutoConfig.from_pretrained(model_path)

        dataset_config = {
            "prompt_key": config.data.prompt_key,
            "activation_vector_key": nla_config.get("activation_vector_key", "activation_vector"),
            "max_prompt_length": config.data.max_prompt_length,
            "activation_dim": model_config.hidden_size,  # Use model's hidden_size
            "injection_position": nla_config.get("injection", {}).get("position", "manual"),
            # Ensure rollout receives the raw prompt messages required by SGLang
            "return_raw_chat": True,
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

        # Ensure tracking defaults are set (auto-generate experiment names)
        _ensure_tracking_defaults(config)
        print(f"Project: {config.trainer.project_name}, Experiment: {config.trainer.experiment_name}")

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

        # NOTE: GRPOTrainerConfig is not used - all GRPO logic is in the base fit() routine
        # The only config needed is:
        # - config.actor_rollout_ref.rollout.n (number of trajectories per prompt)
        # - config.algorithm.adv_estimator = "grpo" (use GRPO advantage estimation)

        # Initialize NLA GRPO trainer with all required parameters
        trainer = NLAGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            grpo_config=None,  # Not used
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
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
