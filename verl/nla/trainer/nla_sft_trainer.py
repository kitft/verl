"""NLA SFT Trainer with activation injection and autoencoder critic training."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Union
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributed import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.torch_functional import masked_mean
from ..models.nla_wrapper import NLAModelWrapper, InjectionConfig
from ..models.autoencoder_critic import NLAAutoencoderCritic


class NLASFTTrainer(FSDPSFTTrainer):
    """
    SFT trainer for NLA that supports both actor and critic training.

    Features:
    - Actor training with activation injection via NLAModelWrapper
    - Critic training with supervised MSE loss for activation prediction
    - Can train both models jointly or separately
    - Fully compatible with FSDP for distributed training
    """

    def __init__(
        self,
        config: DictConfig,
        device_mesh: Any,
        ulysses_device_mesh: Any,
        tokenizer: Any,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        train_mode: str = "both",  # "actor", "critic", or "both"
    ):
        """
        Initialize NLA SFT trainer.

        Args:
            config: Training configuration
            device_mesh: Device mesh for FSDP
            ulysses_device_mesh: Device mesh for Ulysses parallelism
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            train_mode: Training mode - "actor", "critic", or "both"
        """
        self.train_mode = train_mode

        # Store NLA-specific config
        self.activation_dim = config.model.get("activation_dim", 768)
        self.injection_config = InjectionConfig(
            mode=config.model.injection.get("mode", "replace"),
            layer_indices=config.model.injection.get("layer_indices", [0]),
            projection_dim=config.model.injection.get("projection_dim", None),
            injection_token_id=config.model.injection.get("injection_token_id", -1),
        )

        # Critic-specific config
        self.critic_config = config.model.get("critic", {})
        self.critic_lr = config.optim.get("critic_lr", config.optim.lr)
        self.train_critic_epochs = config.trainer.get("critic_epochs", 1)

        # Initialize base trainer
        super().__init__(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    def _build_model_optimizer(self) -> None:
        """
        Build NLA models and optimizers.

        Overrides base method to:
        1. Wrap actor model with NLAModelWrapper
        2. Initialize critic model if needed
        3. Create separate optimizers
        """
        # Build base model first
        super()._build_model_optimizer()

        # Wrap the actor model with NLA wrapper if training actor
        if self.train_mode in ["actor", "both"]:
            # The base class already created self.fsdp_model
            # We need to extract the base model and rewrap it
            base_model = self.model  # This is the unwrapped model before FSDP

            # Wrap with NLA wrapper
            wrapped_model = NLAModelWrapper(
                base_model=base_model,
                injection_config=self.injection_config,
                hidden_dim=base_model.config.hidden_size,
                activation_dim=self.activation_dim,
            )

            # Re-apply FSDP wrapping
            self.fsdp_model = FSDP(
                wrapped_model,
                device_mesh=self.device_mesh,
                use_orig_params=True,
                **self._get_fsdp_config()
            )

            # Recreate optimizer for wrapped model
            self.optimizer = torch.optim.AdamW(
                self.fsdp_model.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                betas=(self.config.optim.beta1, self.config.optim.beta2),
            )

        # Initialize critic if needed
        if self.train_mode in ["critic", "both"]:
            self._build_critic()

    def _build_critic(self) -> None:
        """Initialize critic model and optimizer."""
        # Load critic base model (can be smaller than actor)
        critic_model_name = self.critic_config.get("model_name", self.config.model.model_name)
        critic_base = AutoModelForCausalLM.from_pretrained(
            critic_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if self.config.model.enable_flashattn else "eager",
        )

        # Create autoencoder critic
        self.critic = NLAAutoencoderCritic(
            base_model=critic_base,
            activation_dim=self.activation_dim,
            hidden_dim=critic_base.config.hidden_size,
            use_pooling=self.critic_config.get("pooling", "last"),
            dropout=self.critic_config.get("dropout", 0.1),
            num_projection_layers=self.critic_config.get("projection_layers", 2),
        )

        # Wrap critic with FSDP
        self.fsdp_critic = FSDP(
            self.critic,
            device_mesh=self.device_mesh,
            use_orig_params=True,
            **self._get_fsdp_config()
        )

        # Create critic optimizer
        self.critic_optimizer = torch.optim.AdamW(
            self.fsdp_critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
        )

        # Create critic scheduler if needed
        if self.config.optim.warmup_steps > 0:
            from transformers import get_cosine_schedule_with_warmup
            self.critic_scheduler = get_cosine_schedule_with_warmup(
                self.critic_optimizer,
                num_warmup_steps=self.config.optim.warmup_steps,
                num_training_steps=self.config.trainer.total_training_steps,
            )
        else:
            self.critic_scheduler = None

    def _get_fsdp_config(self) -> Dict:
        """Get FSDP configuration dict."""
        # This would normally come from config
        # Simplified version here
        return {
            "mixed_precision": self.fsdp_mixed_precision_policy,
            "sharding_strategy": self.fsdp_sharding_strategy,
        }

    def _compute_loss_and_backward(
        self,
        batch: Union[TensorDict, Dict],
        do_backward: bool = True,
        n_micro_batches: int = 1
    ) -> torch.Tensor:
        """
        Compute actor loss with activation injection.

        Overrides base method to pass activation vectors during forward pass.
        """
        # Extract tensors from batch
        if isinstance(batch, TensorDict):
            input_ids = batch["input_ids"].to(self.device_name)
            labels = batch["input_ids"].to(self.device_name)
            loss_mask = batch.get("loss_mask", torch.ones_like(input_ids)).to(self.device_name)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device_name)
            activation_vectors = batch.get("activation_vectors")
        else:
            input_ids = batch["input_ids"].to(self.device_name)
            labels = batch["input_ids"].to(self.device_name)
            loss_mask = batch.get("loss_mask", torch.ones_like(input_ids)).to(self.device_name)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device_name)
            activation_vectors = batch.get("activation_vectors")

        if activation_vectors is not None:
            activation_vectors = activation_vectors.to(self.device_name)

        # Shift labels for next token prediction
        labels = labels[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        # Forward pass with activation injection
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16, enabled=True):
            output = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                activation_vectors=activation_vectors,  # Pass activation vectors
                use_cache=False,
            )

            # Compute cross-entropy loss
            logits = output.logits if hasattr(output, "logits") else output[0]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none"
            )
            loss = loss.reshape(labels.shape)

            # Apply loss mask
            loss = masked_mean(loss, loss_mask, axis=-1).mean()

            # Scale loss for gradient accumulation
            loss = loss / n_micro_batches

        if do_backward:
            loss.backward()

        return loss

    def _train_critic_step(self, batch: Union[TensorDict, Dict]) -> Dict[str, float]:
        """
        Train critic to predict activation vectors from responses.

        Args:
            batch: Batch containing response_ids and target activation vectors

        Returns:
            Dict with critic training metrics
        """
        # Extract critic training data
        if isinstance(batch, TensorDict):
            response_ids = batch["response_ids"].to(self.device_name)
            response_mask = batch["response_attention_mask"].to(self.device_name)
            target_activations = batch["activation_vectors"].to(self.device_name)
        else:
            response_ids = batch["response_ids"].to(self.device_name)
            response_mask = batch["response_attention_mask"].to(self.device_name)
            target_activations = batch["activation_vectors"].to(self.device_name)

        self.fsdp_critic.train()
        total_loss = 0.0

        for _ in range(self.train_critic_epochs):
            self.critic_optimizer.zero_grad()

            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16, enabled=True):
                # Get predicted activations from critic
                critic_output = self.fsdp_critic(
                    input_ids=response_ids,
                    attention_mask=response_mask,
                )
                predicted_activations = critic_output.predicted_activation

                # Compute MSE loss
                loss = F.mse_loss(predicted_activations, target_activations)

            loss.backward()

            # Clip gradients
            self.fsdp_critic.clip_grad_norm_(self.config.optim.max_norm)

            self.critic_optimizer.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.train_critic_epochs
        return {"critic_loss": avg_loss}

    def training_step(self, batch: Union[TensorDict, Dict]) -> Dict[str, Any]:
        """
        Perform a training step for actor and/or critic.

        Overrides base method to support dual training.
        """
        metrics = {}

        # Train actor if needed
        if self.train_mode in ["actor", "both"]:
            self.fsdp_model.train()
            self.optimizer.zero_grad()

            # Split batch for gradient accumulation
            if self.config.data.micro_batch_size_per_gpu < batch["input_ids"].shape[0]:
                micro_batches = self._split_batch(batch)
            else:
                micro_batches = [batch]

            actor_loss = 0.0
            for micro_batch in micro_batches:
                loss = self._compute_loss_and_backward(
                    batch=micro_batch,
                    n_micro_batches=len(micro_batches)
                )
                actor_loss += loss.item()

            # Clip gradients and step optimizer
            self.fsdp_model.clip_grad_norm_(self.config.optim.max_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

            metrics["train/actor_loss"] = actor_loss
            metrics["train/actor_lr"] = self.optimizer.param_groups[0]["lr"]

        # Train critic if needed
        if self.train_mode in ["critic", "both"]:
            critic_metrics = self._train_critic_step(batch)
            metrics["train/critic_loss"] = critic_metrics["critic_loss"]
            if hasattr(self, "critic_optimizer"):
                metrics["train/critic_lr"] = self.critic_optimizer.param_groups[0]["lr"]

        # Update global step
        self.global_step += 1

        return metrics

    def _split_batch(self, batch: Union[TensorDict, Dict]) -> list:
        """
        Split batch into micro-batches for gradient accumulation.
        """
        micro_batch_size = self.config.data.micro_batch_size_per_gpu
        batch_size = batch["input_ids"].shape[0] if isinstance(batch, dict) else batch["input_ids"].shape[0]

        micro_batches = []
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            if isinstance(batch, TensorDict):
                micro_batch = batch[i:end_idx]
            else:
                micro_batch = {k: v[i:end_idx] if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}
            micro_batches.append(micro_batch)

        return micro_batches

    def validation_step(self, batch: Union[TensorDict, Dict]) -> Dict[str, Any]:
        """
        Perform a validation step.
        """
        metrics = {}

        with torch.no_grad():
            # Validate actor if needed
            if self.train_mode in ["actor", "both"]:
                self.fsdp_model.eval()
                loss = self._compute_loss_and_backward(batch, do_backward=False)
                metrics["val/actor_loss"] = loss.item()

            # Validate critic if needed
            if self.train_mode in ["critic", "both"] and hasattr(self, "fsdp_critic"):
                self.fsdp_critic.eval()

                if isinstance(batch, TensorDict):
                    response_ids = batch["response_ids"].to(self.device_name)
                    response_mask = batch["response_attention_mask"].to(self.device_name)
                    target_activations = batch["activation_vectors"].to(self.device_name)
                else:
                    response_ids = batch["response_ids"].to(self.device_name)
                    response_mask = batch["response_attention_mask"].to(self.device_name)
                    target_activations = batch["activation_vectors"].to(self.device_name)

                critic_output = self.fsdp_critic(
                    input_ids=response_ids,
                    attention_mask=response_mask,
                )
                predicted_activations = critic_output.predicted_activation
                critic_loss = F.mse_loss(predicted_activations, target_activations)
                metrics["val/critic_loss"] = critic_loss.item()

        return metrics