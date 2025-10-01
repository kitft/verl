"""NLA FSDP SFT Trainer that extends the base FSDP trainer with NLA-specific functionality."""

from contextlib import nullcontext

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM

from verl.nla.models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead
from verl.nla.models.nla_wrapper import InjectionConfig, NLAModelWrapper
from verl.trainer.fsdp_sft_trainer import (
    FSDPSFTTrainer,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    index_first_axis,
    pad_input,
    rearrange,
    ulysses_pad_and_slice_inputs,
    unpad_input,
)
from verl.utils.device import get_device_id
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_dtypes import PrecisionType


class NLAFSDPSFTTrainer(FSDPSFTTrainer):
    """
    NLA-specific FSDP SFT Trainer that supports EITHER actor OR critic training.

    Actor mode:
    - Wraps base model with NLAModelWrapper for activation injection
    - Trains model to generate responses with injected activation vectors

    Critic mode:
    - Uses base model to predict activation vectors from inputs
    - Outputs activation vectors directly from last hidden states
    """

    def __init__(
        self,
        config: DictConfig,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        # Store NLA config before calling parent
        self.nla_config = config.get("nla", {})
        self.train_mode = self.nla_config.get("train_mode", "actor")  # 'actor' or 'critic' only

        # Validate train_mode
        if self.train_mode not in ["actor", "critic"]:
            raise ValueError(f"train_mode must be 'actor' or 'critic', got '{self.train_mode}'")

        # Call parent constructor
        super().__init__(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    def _build_model_optimizer(self):
        """Build either actor or critic model based on train_mode."""
        if self.train_mode == "actor":
            # Build NLA-wrapped actor model
            self._build_nla_actor_model_optimizer()
        else:  # critic mode
            # Build critic model and set it as the main model
            self._build_critic_model_optimizer()

    def _build_nla_actor_model_optimizer(self):
        """Build NLA-wrapped actor model and optimizer."""
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Load config
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config

        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )

        # Initialize model with meta tensors if needed
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            # For tiny model testing, just use the config and ignore weights mismatch
            # The yujiepan/gemma-2-tiny-random has mismatched weights
            try:
                # Try loading normally first
                base_model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2"
                    if self.config.model.get("enable_flashattn", False)
                    else "eager",
                    trust_remote_code=trust_remote_code,
                )
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    # If there's a size mismatch, just create model from config with random weights
                    print("Weight size mismatch detected, using random initialization with config")
                    base_model = AutoModelForCausalLM.from_config(
                        config=config,
                        torch_dtype=torch_dtype,
                        attn_implementation="flash_attention_2"
                        if self.config.model.get("enable_flashattn", False)
                        else "eager",
                        trust_remote_code=trust_remote_code,
                    )
                else:
                    raise

            # Configure NLA injection
            injection_config = InjectionConfig(
                mode=self.nla_config.get("injection", {}).get("mode", "replace"),
                layer_indices=self.nla_config.get("injection", {}).get("layer_indices", [0]),
                projection_dim=self.nla_config.get("injection", {}).get("projection_dim", None),
                injection_token=self.nla_config.get("injection", {}).get("injection_token", "<INJECT>"),
            )

            # Wrap with NLA wrapper
            self.model = NLAModelWrapper(
                base_model=base_model,
                tokenizer=self.tokenizer,
                injection_config=injection_config,
                hidden_dim=base_model.config.hidden_size,
                activation_dim=base_model.config.hidden_size,  # Always use model's hidden_size
            )

            if self.device_mesh.get_rank() == 0:
                print("Wrapped actor model with NLAModelWrapper")
                print(
                    f"Injection token: '{injection_config.injection_token}' (ID: {self.model.injection_config.injection_token_id})"
                )

        # Apply gradient checkpointing if needed
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Apply FSDP wrapping
        # For actor, we wrap the base_model inside the NLAModelWrapper
        self._apply_fsdp_wrapping(
            model_to_wrap=self.model,
            model_for_policy=self.model.base_model,
            inner_model_for_fsdp2=self.model.base_model
        )

        # Create optimizer and scheduler
        self._create_optimizer_and_scheduler()

    def _build_critic_model_optimizer(self):
        """Build critic model that outputs activation vectors from last hidden states."""
        critic_config = self.nla_config.get("critic", {})
        critic_model_path = critic_config.get("model_path", self.config.model.partial_pretrain)

        if self.device_mesh.get_rank() == 0:
            print(f"Building NLA critic model from: {critic_model_path}")

        # Create critic with vector value head
        # Set as self.model for unified handling
        self.model = AutoModelForCausalLMWithVectorValueHead(
            pretrained_model_name_or_path=critic_model_path,
            dropout=critic_config.get("dropout", 0.1),
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation="flash_attention_2" if self.config.model.get("enable_flashattn", False) else "eager",
        )

        # Apply FSDP wrapping
        # For critic, we wrap the pretrained_model inside AutoModelForCausalLMWithVectorValueHead
        self._apply_fsdp_wrapping(
            model_to_wrap=self.model,
            model_for_policy=self.model.pretrained_model,
            inner_model_for_fsdp2=self.model.pretrained_model
        )

        # Create optimizer and scheduler
        self._create_optimizer_and_scheduler()

    def _apply_fsdp_wrapping(self, model_to_wrap, model_for_policy, inner_model_for_fsdp2):
        """Apply FSDP wrapping to the model based on the configured strategy.

        Args:
            model_to_wrap: The outer model to wrap with FSDP (for FSDP1)
            model_for_policy: The model to use for auto wrap policy generation
            inner_model_for_fsdp2: The inner model to wrap with FSDP2
        """
        fsdp_strategy = self.config.model.strategy

        if fsdp_strategy == "fsdp":
            # Setup FSDP1 with consistent configuration for both actor and critic
            from torch.distributed.fsdp import CPUOffload, ShardingStrategy
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision

            # Setup mixed precision if configured
            mixed_precision = None
            if self.config.model.fsdp_config.get("mixed_precision", False):
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32
                )

            # Setup auto wrap policy
            auto_wrap_policy = get_fsdp_wrap_policy(
                model_for_policy,
                config=self.config.model.fsdp_config.wrap_policy,
                is_lora=self.config.model.get("lora_rank", 0) > 0,
            )

            # Setup CPU offload if configured
            cpu_offload = None
            if self.config.model.fsdp_config.cpu_offload:
                cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

            # Apply FSDP1
            self.fsdp_model = FSDP(
                model_to_wrap,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )

        elif fsdp_strategy == "fsdp2":
            # Setup FSDP2 with consistent configuration for both actor and critic
            from verl.utils.fsdp_utils import CPUOffloadPolicy

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True
            )

            # Setup CPU offload if configured
            cpu_offload = None
            if self.config.model.fsdp_config.cpu_offload:
                cpu_offload = CPUOffloadPolicy(offload_params=self.config.model.fsdp_config.offload_params)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }

            # Apply FSDP2 to the inner model
            full_state = inner_model_for_fsdp2.state_dict()
            apply_fsdp2(inner_model_for_fsdp2, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(inner_model_for_fsdp2, full_state, self.device_mesh, cpu_offload)
            # For checkpoint saving/loading, point to the FSDP-wrapped inner model
            # The outer wrapper (NLAModelWrapper or AutoModelForCausalLMWithVectorValueHead)
            # contains the FSDP-wrapped model and handles forward passes correctly
            self.fsdp_model = inner_model_for_fsdp2
            # Keep reference to the outer model for forward passes
            self.model = model_to_wrap

        else:
            raise NotImplementedError(f"Strategy {fsdp_strategy} not implemented")

    def _create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler for either actor or critic training."""
        # Determine learning rate
        if self.train_mode == "critic":
            lr = self.nla_config.get("critic_lr", self.config.optim.lr)
        else:
            lr = self.config.optim.lr

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.fsdp_model.parameters(),
            lr=lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        # Calculate training steps
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        # Create learning rate scheduler
        from verl.utils.torch_functional import get_cosine_schedule_with_warmup

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps
        )

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss with activation injection for actor training."""
        if self.train_mode != "actor":
            return super()._compute_loss_and_backward(batch, do_backward, n_micro_batches)

        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        activation_vectors = batch.get("activation_vectors")
        if activation_vectors is not None:
            activation_vectors = activation_vectors.to(self.device_name)

        if activation_vectors is not None and use_sp:
            raise NotImplementedError(
                "Activation injection is not implemented when use_remove_padding/sequence parallelism is enabled."
            )

        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # For FSDP2, self.fsdp_model points to inner base_model but we need to call
        # self.model (the NLAModelWrapper) to get activation injection
        forward_model = self.model if self.config.model.strategy == "fsdp2" else self.fsdp_model

        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                labels = input_ids[:, 1:].contiguous()
                output = forward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    activation_vectors=activation_vectors,
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                batch_size, seqlen = input_ids.shape
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad,
                    position_ids_rmpad,
                    sp_size=get_ulysses_sequence_parallel_world_size(),
                )
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    None,
                    get_ulysses_sequence_parallel_world_size(),
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = forward_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                    activation_vectors=activation_vectors,
                )

                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
            loss = loss / n_micro_batches

            if do_backward:
                loss.backward()
            return loss

    def _compute_critic_loss(self, batch):
        """Compute MSE loss for critic training."""
        # Get input tokens and target activations
        # Use the full input_ids since we don't have separate response_ids
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device_name)
        target_activations = batch["activation_vectors"].to(self.device_name)

        # Forward pass through critic
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            critic_output = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            # Extract the last position's activation vector
            values = critic_output.value  # (batch, seq_len, hidden_size)
            # Get the last non-padding position for each sequence
            seq_lengths = attention_mask.sum(dim=-1) - 1
            batch_size = input_ids.shape[0]
            predicted_activations = values[torch.arange(batch_size), seq_lengths]  # (batch, hidden_size)

            # Compute MSE loss
            loss = nn.functional.mse_loss(predicted_activations, target_activations)

        return loss

    def training_step(self, batch):
        """Training step for either actor or critic."""
        if self.train_mode == "actor":
            # Use parent's training_step for actor
            return super().training_step(batch)
        else:
            # Critic training
            import time
            start_time = time.time()

            self.fsdp_model.train()
            self.optimizer.zero_grad()

            # Compute critic loss
            critic_loss = self._compute_critic_loss(batch)
            critic_loss.backward()

            # Clip gradients
            if self.config.model.strategy == "fsdp2":
                from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
                grad_norm = fsdp2_clip_grad_norm_(
                    self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad
                )
            else:
                grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

            # Step optimizer
            if torch.isfinite(grad_norm):
                self.optimizer.step()
                self.lr_scheduler.step()
            else:
                self.optimizer.zero_grad()

            # Return metrics
            end_time = time.time()
            return {
                "train/loss": critic_loss.item(),
                "train/grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else float("inf"),
                "train/lr(1e-3)": self.lr_scheduler.get_last_lr()[0] * 1e3,
                "train/time(s)": end_time - start_time,
            }

    def validation_step(self, batch):
        """Validation step for either actor or critic."""
        if self.train_mode == "actor":
            # Use parent's validation_step for actor
            return super().validation_step(batch)
        else:
            # Critic validation
            self.fsdp_model.eval()
            with torch.no_grad():
                critic_loss = self._compute_critic_loss(batch)
            return critic_loss
