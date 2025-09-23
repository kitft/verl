"""NLA Critic Worker with vector value head support."""

import torch
from typing import Dict, Optional, Any
from verl.workers.fsdp_workers import CriticWorker
from verl.nla.models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs


class NLACriticWorker(CriticWorker):
    """Critic worker that loads vector value head model for NLA."""

    def _build_critic_model_optimizer(self, config):
        """Override model initialization to load vector value head model."""
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType
        from verl.utils import hf_tokenizer, hf_processor
        from transformers import AutoConfig
        import warnings
        from omegaconf import OmegaConf

        use_shm = config.model.get("use_shm", False)
        from verl.utils.fs import copy_to_local
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # Load tokenizer for injection token management
        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm) if config.model.get("tokenizer_path") else local_path
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"NLA Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Load AutoConfig for model configuration
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )

        # Load NLA critic model with vector value head instead of standard value head
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create the NLA critic model with vector value head
            self.critic_module = AutoModelForCausalLMWithVectorValueHead(
                pretrained_model_name_or_path=local_path,
                dropout=config.model.get("dropout", 0.1),
                trust_remote_code=config.model.get("trust_remote_code", False),
                torch_dtype=torch_dtype,
            )

            if torch.cuda.is_available():
                self.critic_module = self.critic_module.cuda()

            print(f"Loaded NLA critic model with vector value head (hidden_size={self.critic_module.hidden_size})")

        # Set up FSDP if needed
        from verl.utils.model import get_init_weight_context_manager
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from verl.utils.fs import copy_local_path_from_hdfs

        self.use_meta_tensor = not critic_model_config.tie_word_embeddings

        # Apply FSDP
        fsdp_config = self._get_fsdp_config()
        self.critic = FSDP(
            self.critic_module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch_dtype,
                reduce_dtype=torch_dtype,
                buffer_dtype=torch_dtype,
            ),
            use_orig_params=self.use_orig_params,
            device_mesh=self.device_mesh,
        )

        # Set up optimizer for critic
        critic_lr = config.optim.get("lr", 5e-5)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        print_model_size(self.critic, "NLA Critic")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the NLA critic model."""
        self._build_critic_model_optimizer(self.config)

    def compute_values(self, batch):
        """
        Override value computation to handle vector outputs.

        Returns:
            torch.Tensor: Vector values of shape (batch_size, seq_len, hidden_size)
        """
        # Forward pass through model
        outputs = self.critic(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Extract vector values (not scalar)
        # The model outputs values of shape (batch, seq_len, hidden_size)
        values = outputs.value if hasattr(outputs, 'value') else outputs[2]

        return values

    def compute_activation_predictions(self, response_ids, attention_mask):
        """
        Compute predicted activation vectors from responses.

        This is used by the GRPO trainer for MSE reward computation.

        Args:
            response_ids: Response token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            torch.Tensor: Predicted activations (batch_size, hidden_size)
        """
        # Create batch dict for compute_values
        batch = {
            "input_ids": response_ids,
            "attention_mask": attention_mask,
        }

        values = self.compute_values(batch)

        # Extract last token's prediction
        if attention_mask is not None:
            # Find last valid token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = values.shape[0]
            last_token_values = values[torch.arange(batch_size), seq_lengths]
        else:
            last_token_values = values[:, -1, :]

        return last_token_values

    def _get_fsdp_config(self):
        """Get FSDP configuration."""
        from torch.distributed.fsdp import ShardingStrategy

        sharding_strategy = self.config.model.fsdp_config.get("sharding_strategy", "FULL_SHARD")
        if hasattr(ShardingStrategy, sharding_strategy):
            sharding_strategy = getattr(ShardingStrategy, sharding_strategy)
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        return {
            "sharding_strategy": sharding_strategy,
            "cpu_offload": self.config.model.fsdp_config.get("param_offload", False),
            "backward_prefetch": self.config.model.fsdp_config.get("backward_prefetch", None),
            "forward_prefetch": self.config.model.fsdp_config.get("forward_prefetch", False),
            "limit_all_gathers": self.config.model.fsdp_config.get("limit_all_gathers", True),
            "use_orig_params": self.use_orig_params,
            "sync_module_states": self.config.model.fsdp_config.get("sync_module_states", True),
        }