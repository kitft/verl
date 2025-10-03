"""
NLA Checkpoint Metadata Management.

Automatically saves metadata alongside checkpoints to track:
- Training configuration and hyperparameters
- Dataset information
- Model architecture details
- Training metrics and results
- Lineage (SFT → RL training chain)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class SFTMetadata:
    """Metadata for SFT (Supervised Fine-Tuning) checkpoints."""

    # Training run info
    timestamp: str
    wandb_run_id: Optional[str] = None
    wandb_run_url: Optional[str] = None
    wandb_project: Optional[str] = None

    # Base model
    base_model: str = ""
    model_architecture: str = ""
    hidden_size: int = 0
    num_layers: int = 0

    # Training configuration
    training_mode: str = "actor"  # "actor" or "critic"
    train_batch_size: int = 0
    micro_batch_size: int = 0
    learning_rate: float = 0.0
    total_epochs: int = 0
    global_step: int = 0

    # Datasets
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    dataset_variant: Optional[str] = None  # "canonical", "random", etc.

    # NLA configuration
    activation_dim: int = 0
    injection_mode: str = "replace"  # "replace", "add", "project"
    injection_layer_indices: List[int] = field(default_factory=list)
    injection_token: Optional[str] = None

    # Results
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    training_time_hours: Optional[float] = None

    # Full config
    hydra_config_name: Optional[str] = None
    full_hydra_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata_version": "1.0",
            "checkpoint_type": "sft",
            "training_run": {
                "timestamp": self.timestamp,
                "wandb_run_id": self.wandb_run_id,
                "wandb_run_url": self.wandb_run_url,
                "wandb_project": self.wandb_project,
            },
            "base_model": {
                "name": self.base_model,
                "architecture": self.model_architecture,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            "training_config": {
                "mode": self.training_mode,
                "train_batch_size": self.train_batch_size,
                "micro_batch_size": self.micro_batch_size,
                "learning_rate": self.learning_rate,
                "total_epochs": self.total_epochs,
                "global_step": self.global_step,
            },
            "datasets": {
                "train": self.train_files,
                "val": self.val_files,
                "variant": self.dataset_variant,
            },
            "nla_config": {
                "activation_dim": self.activation_dim,
                "injection_mode": self.injection_mode,
                "layer_indices": self.injection_layer_indices,
                "injection_token": self.injection_token,
            },
            "results": {
                "final_train_loss": self.final_train_loss,
                "final_val_loss": self.final_val_loss,
                "best_val_loss": self.best_val_loss,
                "training_time_hours": self.training_time_hours,
            },
            "hydra": {
                "config_name": self.hydra_config_name,
                "full_config": self.full_hydra_config,
            },
        }


@dataclass
class RLMetadata:
    """Metadata for RL (Reinforcement Learning) checkpoints."""

    # Training run info
    timestamp: str
    wandb_run_id: Optional[str] = None
    wandb_run_url: Optional[str] = None
    wandb_project: Optional[str] = None

    # Algorithm
    algorithm: str = "grpo"  # "grpo", "ppo", etc.
    global_step: int = 0
    total_training_steps: int = 0

    # SFT lineage (critical for tracking where models came from)
    actor_lineage: Optional[Dict[str, Any]] = None
    critic_lineage: Optional[Dict[str, Any]] = None

    # Model paths
    actor_model_path: str = ""
    critic_model_path: str = ""

    # Training configuration
    ppo_mini_batch_size: int = 0
    ppo_epochs: int = 0
    steps_per_update: int = 0
    actor_lr: float = 0.0
    critic_lr: float = 0.0

    # Datasets
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)

    # NLA configuration
    injection_mode: str = "replace"
    injection_layer_indices: List[int] = field(default_factory=list)
    critic_truncate_at_layer: Optional[int] = None

    # Results
    final_reward_mean: Optional[float] = None
    final_reward_std: Optional[float] = None
    best_reward_mean: Optional[float] = None
    training_time_hours: Optional[float] = None

    # Full config
    hydra_config_name: Optional[str] = None
    full_hydra_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata_version": "1.0",
            "checkpoint_type": "rl",
            "training_run": {
                "timestamp": self.timestamp,
                "wandb_run_id": self.wandb_run_id,
                "wandb_run_url": self.wandb_run_url,
                "wandb_project": self.wandb_project,
            },
            "algorithm": {
                "name": self.algorithm,
                "global_step": self.global_step,
                "total_training_steps": self.total_training_steps,
            },
            "lineage": {
                "actor": self.actor_lineage,
                "critic": self.critic_lineage,
            },
            "models": {
                "actor_path": self.actor_model_path,
                "critic_path": self.critic_model_path,
            },
            "training_config": {
                "ppo_mini_batch_size": self.ppo_mini_batch_size,
                "ppo_epochs": self.ppo_epochs,
                "steps_per_update": self.steps_per_update,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
            },
            "datasets": {
                "train": self.train_files,
                "val": self.val_files,
            },
            "nla_config": {
                "injection_mode": self.injection_mode,
                "layer_indices": self.injection_layer_indices,
                "critic_truncate_at_layer": self.critic_truncate_at_layer,
            },
            "results": {
                "final_reward_mean": self.final_reward_mean,
                "final_reward_std": self.final_reward_std,
                "best_reward_mean": self.best_reward_mean,
                "training_time_hours": self.training_time_hours,
            },
            "hydra": {
                "config_name": self.hydra_config_name,
                "full_config": self.full_hydra_config,
            },
        }


def save_metadata(
    checkpoint_dir: Union[str, Path], metadata: Union[SFTMetadata, RLMetadata], filename: str = "nla_metadata.yaml"
) -> None:
    """
    Save metadata to checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        metadata: Metadata object (SFTMetadata or RLMetadata)
        filename: Filename for metadata (default: nla_metadata.yaml)
    """
    checkpoint_path = Path(checkpoint_dir)

    # Create directory if it doesn't exist (defensive programming)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    metadata_path = checkpoint_path / filename

    metadata_dict = metadata.to_dict()

    # Convert to JSON-serializable format (handles complex nested structures)
    # Then convert back to dict for YAML
    metadata_json_str = json.dumps(metadata_dict, default=str, indent=2)
    metadata_dict = json.loads(metadata_json_str)

    # Save as YAML for human readability with atomic write
    temp_metadata_path = checkpoint_path / f"{filename}.tmp"
    with open(temp_metadata_path, "w") as f:
        yaml.dump(metadata_dict, f, default_flow_style=False, sort_keys=False)

    # Atomic rename
    import os
    os.rename(temp_metadata_path, metadata_path)

    print(f"Saved checkpoint metadata to: {metadata_path}")


def load_metadata(checkpoint_dir: Union[str, Path], filename: str = "nla_metadata.yaml") -> Optional[Dict[str, Any]]:
    """
    Load metadata from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        filename: Filename for metadata (default: nla_metadata.yaml)

    Returns:
        Metadata dictionary or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / filename

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        return yaml.safe_load(f)


def extract_lineage_from_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Extract SFT lineage information from a checkpoint directory.

    This reads the metadata from an SFT checkpoint to create lineage info
    that can be embedded in RL checkpoints.

    Args:
        checkpoint_path: Path to SFT checkpoint

    Returns:
        Lineage dictionary with key training info, or None if metadata doesn't exist
    """
    metadata = load_metadata(checkpoint_path)

    if metadata is None:
        return None

    # Only works for SFT checkpoints
    if metadata.get("checkpoint_type") != "sft":
        return None

    # Extract key information for lineage tracking
    lineage = {
        "checkpoint_path": str(checkpoint_path),
        "training_mode": metadata.get("training_config", {}).get("mode"),
        "base_model": metadata.get("base_model", {}).get("name"),
        "global_step": metadata.get("training_config", {}).get("global_step"),
        "final_val_loss": metadata.get("results", {}).get("final_val_loss"),
        "dataset_variant": metadata.get("datasets", {}).get("variant"),
        "wandb_run_id": metadata.get("training_run", {}).get("wandb_run_id"),
        "timestamp": metadata.get("training_run", {}).get("timestamp"),
    }

    return lineage


def create_sft_metadata_from_config(
    config: DictConfig,
    global_step: int,
    final_train_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    training_time_hours: Optional[float] = None,
    model_config: Optional[Any] = None,
    wandb_run_id: Optional[str] = None,
    wandb_run_url: Optional[str] = None,
) -> SFTMetadata:
    """
    Create SFT metadata from Hydra config and training results.

    Args:
        config: Hydra DictConfig
        global_step: Current training step
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        best_val_loss: Best validation loss seen
        training_time_hours: Total training time
        model_config: Model configuration (from transformers)
        wandb_run_id: WandB run ID
        wandb_run_url: WandB run URL

    Returns:
        SFTMetadata object
    """
    # Extract dataset variant from file paths
    train_files = config.data.get("train_files", [])
    if isinstance(train_files, str):
        train_files = [train_files]

    dataset_variant = None
    if train_files:
        train_file = train_files[0]
        if "random" in train_file.lower():
            dataset_variant = "random"
        elif "canonical" in train_file.lower() or "layer" in train_file.lower():
            dataset_variant = "canonical"

    # Get model info
    base_model = config.model.get("partial_pretrain", "")
    hidden_size = model_config.hidden_size if model_config else 0
    num_layers = getattr(model_config, "num_hidden_layers", 0)
    model_architecture = model_config.__class__.__name__ if model_config else ""

    # Get NLA config
    nla_config = config.get("nla", {})
    injection_config = nla_config.get("injection", {})

    return SFTMetadata(
        timestamp=datetime.now().isoformat(),
        wandb_run_id=wandb_run_id,
        wandb_run_url=wandb_run_url,
        wandb_project=config.trainer.get("project_name"),
        base_model=base_model,
        model_architecture=model_architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
        training_mode=nla_config.get("train_mode", "actor"),
        train_batch_size=config.data.get("train_batch_size", 0),
        micro_batch_size=config.data.get("micro_batch_size_per_gpu", 0),
        learning_rate=config.optim.get("lr", 0.0),
        total_epochs=config.trainer.get("total_epochs", 0),
        global_step=global_step,
        train_files=train_files,
        val_files=[config.data.get("val_files", "")],
        dataset_variant=dataset_variant,
        activation_dim=hidden_size,  # Same as hidden_size
        injection_mode=injection_config.get("mode", "replace"),
        injection_layer_indices=injection_config.get("layer_indices", []),
        injection_token=injection_config.get("injection_token"),
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        best_val_loss=best_val_loss,
        training_time_hours=training_time_hours,
        hydra_config_name=config.get("hydra_config_name"),
        full_hydra_config=OmegaConf.to_container(config, resolve=True),
    )


def create_rl_metadata_from_config(
    config: DictConfig,
    global_step: int,
    actor_checkpoint_path: Optional[str] = None,
    critic_checkpoint_path: Optional[str] = None,
    final_reward_mean: Optional[float] = None,
    final_reward_std: Optional[float] = None,
    best_reward_mean: Optional[float] = None,
    training_time_hours: Optional[float] = None,
    wandb_run_id: Optional[str] = None,
    wandb_run_url: Optional[str] = None,
) -> RLMetadata:
    """
    Create RL metadata from Hydra config and training results.

    Automatically extracts SFT lineage if actor/critic paths point to SFT checkpoints.

    Args:
        config: Hydra DictConfig
        global_step: Current training step
        actor_checkpoint_path: Path to actor SFT checkpoint (for lineage)
        critic_checkpoint_path: Path to critic SFT checkpoint (for lineage)
        final_reward_mean: Final mean reward
        final_reward_std: Final reward std dev
        best_reward_mean: Best mean reward seen
        training_time_hours: Total training time
        wandb_run_id: WandB run ID
        wandb_run_url: WandB run URL

    Returns:
        RLMetadata object
    """
    # Get dataset files
    train_files = config.data.get("train_files", [])
    val_files = config.data.get("val_files", [])
    if isinstance(train_files, str):
        train_files = [train_files]
    if isinstance(val_files, str):
        val_files = [val_files]

    # Extract lineage from SFT checkpoints
    actor_lineage = None
    critic_lineage = None

    if actor_checkpoint_path:
        actor_lineage = extract_lineage_from_checkpoint(actor_checkpoint_path)

    if critic_checkpoint_path:
        critic_lineage = extract_lineage_from_checkpoint(critic_checkpoint_path)

    # Get model paths from config
    actor_model_path = config.actor_rollout_ref.model.get("path", "")
    critic_model_path = config.critic.model.get("path", "")

    # Get training config
    actor_lr = config.actor_rollout_ref.actor.optim.get("lr", 0.0)
    critic_lr = config.critic.optim.get("lr", 0.0)

    # Get NLA config
    nla_config = config.get("nla", {})
    injection_config = nla_config.get("injection", {})

    return RLMetadata(
        timestamp=datetime.now().isoformat(),
        wandb_run_id=wandb_run_id,
        wandb_run_url=wandb_run_url,
        wandb_project=config.trainer.get("project_name"),
        algorithm=config.algorithm.get("adv_estimator", "grpo"),
        global_step=global_step,
        total_training_steps=config.trainer.get("total_training_steps", 0),
        actor_lineage=actor_lineage,
        critic_lineage=critic_lineage,
        actor_model_path=actor_model_path,
        critic_model_path=critic_model_path,
        ppo_mini_batch_size=config.trainer.get("ppo_mini_batch_size", 0),
        ppo_epochs=config.trainer.get("ppo_epochs", 0),
        steps_per_update=config.trainer.get("steps_per_update", 0),
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        train_files=train_files,
        val_files=val_files,
        injection_mode=injection_config.get("mode", "replace"),
        injection_layer_indices=injection_config.get("layer_indices", []),
        critic_truncate_at_layer=config.critic.model.get("truncate_at_layer"),
        final_reward_mean=final_reward_mean,
        final_reward_std=final_reward_std,
        best_reward_mean=best_reward_mean,
        training_time_hours=training_time_hours,
        hydra_config_name=config.get("hydra_config_name"),
        full_hydra_config=OmegaConf.to_container(config, resolve=True),
    )
