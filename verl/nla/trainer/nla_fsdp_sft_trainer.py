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
    ulysses_pad_and_slice_inputs,
)
from verl.utils.device import get_device_id, is_cuda_available, is_npu_available
from verl.utils.fs import copy_to_local

# Import flash-attn padding utilities conditionally (same as base trainer)
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
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

        # Initialize metadata tracking
        self.best_val_loss = float("inf")
        self.final_train_loss = None
        self.final_val_loss = None
        self.training_start_time = None

        # Call parent constructor
        super().__init__(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    def _build_dataloader(self, train_dataset, val_dataset):
        """Override to use NLA collator that handles variable-length responses."""
        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Import NLA collator
        from verl.nla.data.nla_sft_dataset import NLASFTCollator

        # Get pad_token_id from tokenizer
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0

        # Create collator
        collate_fn = NLASFTCollator(pad_token_id=pad_token_id)

        # Build dataloader with custom collator
        from torch.utils.data import DistributedSampler

        # Use StatefulDataLoader for checkpoint support (not regular DataLoader)
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError:
            # Fallback if StatefulDataLoader not available
            from torch.utils.data import DataLoader as StatefulDataLoader

        # Calculate per-GPU batch size using micro_batch_size_per_gpu from config
        # This is what the training loop expects
        per_gpu_batch_size = self.config.data.micro_batch_size_per_gpu

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=self.device_mesh.size(),
            rank=self.device_mesh.get_rank(),
            drop_last=True,
        )

        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,  # Use NLA collator
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=self.device_mesh.size(),
            rank=self.device_mesh.get_rank(),
            drop_last=True,
        )
        self.val_dataloader = StatefulDataLoader(
            self.val_dataset,
            batch_size=per_gpu_batch_size,
            sampler=self.val_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,  # Use NLA collator
        )

        if self.device_mesh.get_rank() == 0:
            print(f"NLA FSDP SFT: Using NLASFTCollator for DataLoader")
            print(f"  Per-GPU batch size: {per_gpu_batch_size}")
            print(f"  Global batch size: {self.config.data.train_batch_size}")
            print(f"  World size: {self.device_mesh.size()}")

    def _truncate_transformer(self, model: nn.Module, num_layers_to_keep: int) -> nn.Module:
        """Truncate a HuggingFace transformer model to keep only the first N layers.

        This method:
        1. Identifies the layer attribute path (works for Llama, Gemma, Mistral, GPT-2, etc.)
        2. Slices the nn.ModuleList to keep only first K layers
        3. Updates model config to reflect new layer count

        Args:
            model: The model to truncate
            num_layers_to_keep: Number of layers to keep (from the start)

        Returns:
            The truncated model (modified in-place, but returned for convenience)
        """
        import functools

        def rgetattr(obj, attr, *args):
            """Recursive getattr for nested attributes like 'model.layers'"""

            def _getattr(obj, attr):
                return getattr(obj, attr, *args)

            return functools.reduce(_getattr, [obj] + attr.split("."))

        def rsetattr(obj, attr, val):
            """Recursive setattr for nested attributes"""
            pre, _, post = attr.rpartition(".")
            return setattr(rgetattr(obj, pre) if pre else obj, post, val)

        # Common layer paths for different model architectures
        layer_paths = [
            "model.layers",  # Llama, Gemma, Mistral, Qwen
            "transformer.h",  # GPT-2
            "transformer.blocks",  # GPT-NeoX
            "encoder.layer",  # BERT-like
            "layers",  # Some other models
        ]

        layers_attribute = None
        path_found = ""
        for path in layer_paths:
            try:
                layers_attribute = rgetattr(model, path)
                path_found = path
                break
            except AttributeError:
                continue

        if layers_attribute is None:
            print(f"Warning: Could not find transformer layers to truncate for {model.__class__.__name__}.")
            print(f"Attempted paths: {layer_paths}")
            return model

        if not isinstance(layers_attribute, nn.ModuleList):
            print(f"Warning: Found attribute at '{path_found}' but it is not an nn.ModuleList. Truncation skipped.")
            return model

        original_layers = len(layers_attribute)
        if num_layers_to_keep <= 0:
            print(f"Truncation skipped: num_layers_to_keep must be > 0, got {num_layers_to_keep}")
            return model
        if num_layers_to_keep >= original_layers:
            print(
                f"Truncation skipped: requested {num_layers_to_keep} layers but model only has "
                f"{original_layers} layers."
            )
            return model

        # Slice the ModuleList to keep only first K layers
        # This preserves layer ordering and ensures correct behavior
        rsetattr(model, path_found, layers_attribute[:num_layers_to_keep])

        # Update config to reflect new layer count
        # This is CRITICAL - many parts of the model rely on this value
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = num_layers_to_keep
        if hasattr(model.config, "n_layer"):  # For GPT-2 style models
            model.config.n_layer = num_layers_to_keep

        model_name = getattr(model.config, "name_or_path", model.__class__.__name__)
        print(f"✓ Truncated model '{model_name}' from {original_layers} to {num_layers_to_keep} layers")
        print(f"  Layer path: {path_found}")
        print(f"  Updated config.num_hidden_layers: {num_layers_to_keep}")

        return model

    def _truncate_model_config(self, config, num_layers_to_keep: int):
        """Truncate model config to specify only K layers (SAFER method for FSDP).

        This modifies the config BEFORE loading the model, which is safer for FSDP:
        - Model is instantiated with K layers from the start
        - Only weights for K layers are loaded from checkpoint
        - No post-hoc surgery on model structure
        - All ranks build identical structure deterministically

        Args:
            config: AutoConfig to modify
            num_layers_to_keep: Number of layers to keep

        Returns:
            Modified config
        """
        # Get original layer count
        original_layers = None
        if hasattr(config, "num_hidden_layers"):
            original_layers = config.num_hidden_layers
        elif hasattr(config, "n_layer"):  # GPT-2 style
            original_layers = config.n_layer
        else:
            print("Warning: Could not find layer count in config. Truncation skipped.")
            return config

        # Validate
        if num_layers_to_keep <= 0:
            print(f"Truncation skipped: num_layers_to_keep must be > 0, got {num_layers_to_keep}")
            return config
        if num_layers_to_keep >= original_layers:
            print(f"Truncation skipped: requested {num_layers_to_keep} but model has {original_layers} layers.")
            return config

        # Modify config (this is the SAFE approach for FSDP)
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_layers_to_keep
        if hasattr(config, "n_layer"):
            config.n_layer = num_layers_to_keep

        model_name = getattr(config, "name_or_path", "model")
        print(f"✓ Config truncated: {model_name} will use {num_layers_to_keep}/{original_layers} layers")
        print("  Method: Config modification (FSDP-safe)")
        print("  Model will be built with truncated structure from the start")

        return config

    def _build_model_optimizer(self):
        """Build either actor or critic model based on train_mode."""
        if self.train_mode == "actor":
            print("Building NLA-wrapped actor model and optimizer")
            # Build NLA-wrapped actor model
            self._build_nla_actor_model_optimizer()
        else:  # critic mode
            print("Building critic model and optimizer")
            # Build critic model and set it as the main model
            self._build_critic_model_optimizer()

    def _build_nla_actor_model_optimizer(self):
        """Build NLA-wrapped actor model and optimizer."""
        # Try partial_pretrain first (FSDP key), fall back to path (standard key)
        model_path = self.config.model.get("partial_pretrain", None) or self.config.model.path
        local_model_path = copy_to_local(src=model_path, verbose=True)

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

        # NOTE: Actor truncation removed - only critic supports truncation
        # Actor must remain full-size for proper generation capabilities

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

            # NOTE: Truncation already applied via config modification above
            # No need for post-hoc model surgery

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
                    f"Injection token: '{injection_config.injection_token}' "
                    f"(ID: {self.model.injection_config.injection_token_id})"
                )

        # Apply gradient checkpointing if needed
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Apply FSDP wrapping
        # For actor, we wrap the base_model inside the NLAModelWrapper
        self._apply_fsdp_wrapping(
            model_to_wrap=self.model,
            model_for_policy=self.model.base_model,
            inner_model_for_fsdp2=self.model.base_model,
        )

        # Create optimizer and scheduler
        self._create_optimizer_and_scheduler()

    def _build_critic_model_optimizer(self):
        """Build critic model that outputs activation vectors from last hidden states."""
        critic_config = self.nla_config.get("critic", {})
        # Try partial_pretrain first (FSDP key), fall back to path (standard key)
        default_model_path = self.config.model.get("partial_pretrain", None) or self.config.model.path
        critic_model_path = critic_config.get("model_path", default_model_path)

        if self.device_mesh.get_rank() == 0:
            print(f"Building NLA critic model from: {critic_model_path}")

        # Load config first for FSDP-safe truncation
        from transformers import AutoConfig

        local_model_path = copy_to_local(src=critic_model_path, verbose=True)
        critic_model_config = AutoConfig.from_pretrained(
            local_model_path, trust_remote_code=self.config.model.trust_remote_code
        )

        # TRUNCATION LOGIC FOR CRITIC (CONFIG-BASED - FSDP SAFE)
        # Modify config BEFORE loading model - safer for distributed training
        num_layers_to_keep = critic_config.get("truncate_layers")
        if num_layers_to_keep and num_layers_to_keep > 0:
            critic_model_config = self._truncate_model_config(critic_model_config, num_layers_to_keep)

        # Load the base model with modified config
        critic_base_model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=critic_model_config,  # Use modified config with truncation
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation="flash_attention_2" if self.config.model.get("enable_flashattn", False) else "eager",
        )

        # Create critic with the (potentially truncated) base model
        # Pass the pre-loaded model instead of model path
        self.model = AutoModelForCausalLMWithVectorValueHead(
            pretrained_model=critic_base_model,
            dropout=critic_config.get("dropout", 0.1),
            output_layer_index=critic_config.get("output_layer_index"),
        )

        # Apply FSDP wrapping
        # For critic, we wrap the pretrained_model inside AutoModelForCausalLMWithVectorValueHead
        self._apply_fsdp_wrapping(
            model_to_wrap=self.model,
            model_for_policy=self.model.pretrained_model,
            inner_model_for_fsdp2=self.model.pretrained_model,
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
            from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Setup mixed precision if configured
            mixed_precision = None
            if self.config.model.fsdp_config.get("mixed_precision", False):
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
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

            # Apply FSDP1 to the inner model (consistent with FSDP2)
            inner_model_fsdp = FSDP(
                inner_model_for_fsdp2,
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

            # Update the outer wrapper to point to the FSDP-wrapped inner model
            # This is crucial for FSDP1 since the wrapper stores a reference to the inner model
            if hasattr(model_to_wrap, "base_model"):
                # NLAModelWrapper case
                model_to_wrap.base_model = inner_model_fsdp
            elif hasattr(model_to_wrap, "pretrained_model"):
                # AutoModelForCausalLMWithVectorValueHead case
                model_to_wrap.pretrained_model = inner_model_fsdp

            # For checkpoint saving/loading, point to the FSDP-wrapped inner model
            # The outer wrapper (NLAModelWrapper or AutoModelForCausalLMWithVectorValueHead)
            # contains the FSDP-wrapped model and handles forward passes correctly
            self.fsdp_model = inner_model_fsdp
            # Keep reference to the outer model for forward passes
            self.model = model_to_wrap

        elif fsdp_strategy == "fsdp2":
            # Setup FSDP2 with consistent configuration for both actor and critic
            from verl.utils.fsdp_utils import CPUOffloadPolicy

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
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
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
        )

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss and backward pass for actor or critic training.

        For actor: CrossEntropy loss with activation injection
        For critic: MSE loss for activation prediction
        """
        if self.train_mode == "critic":
            # Critic training: predict activation vectors from responses
            loss = self._compute_critic_loss(batch)
            loss = loss / n_micro_batches

            if do_backward:
                loss.backward()
            return loss

        elif self.train_mode == "actor":
            # Actor training: generate with activation injection
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

            # Both FSDP1 and FSDP2 now wrap the inner model, so we always use self.model
            # (the outer wrapper) for forward passes to get activation injection
            forward_model = self.model

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
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
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

        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")

    def _compute_critic_loss(self, batch):
        """Compute MSE loss for critic training."""
        # Get input tokens and target activations
        # Use the full input_ids since we don't have separate response_ids
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device_name)
        target_activations = batch["activation_vectors"].to(self.device_name)

        # Forward pass through critic
        # Both FSDP1 and FSDP2 now wrap the inner model, so we always use self.model
        # (the wrapper) to ensure output_hidden_states=True is set
        forward_model = self.model
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            critic_output = forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            # Extract the last position's activation vector
            # values = critic_output.hidden_states  # (batch, seq_len, hidden_size)
            # # Get the last non-padding position for each sequence
            # seq_lengths = attention_mask.sum(dim=-1) - 1
            # batch_size = input_ids.shape[0]
            # predicted_activations = values[torch.arange(batch_size), seq_lengths]  # (batch, hidden_size)
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)

            if hasattr(critic_output, "value") and critic_output.value is not None:
                predicted_activations = critic_output.value
            else:
                raise ValueError("Value is not available in the critic output. this has been normed")
                if hasattr(critic_output, "last_hidden_state") and critic_output.last_hidden_state is not None:
                    last_hidden_state = critic_output.last_hidden_state  # (batch, seq_len, hidden_size)
                elif hasattr(critic_output, "hidden_states") and critic_output.hidden_states is not None:
                    last_hidden_state = critic_output.hidden_states[-1]
                else:
                    raise ValueError(
                        f"Last hidden state or hidden states are not available in the critic output: "
                        f"available keys are {critic_output.keys()} and which are none? "
                        f"{[key is None for key in critic_output.keys()]}"
                    )
                indices = torch.arange(target_activations.shape[0], device=target_activations.device)
                predicted_activations = last_hidden_state[indices, seq_lengths]  # (batch, hidden_size)

            # Compute MSE loss
            loss = nn.functional.mse_loss(predicted_activations, target_activations)

        return loss

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

    def fit(self):
        """Override fit to track training time and metrics."""
        import time

        import torch
        from omegaconf import OmegaConf
        from tensordict import TensorDict
        from tqdm import tqdm

        from verl.utils.tracking import Tracking

        self.training_start_time = time.time()
        rank = self.device_mesh.get_rank()

        tracking = None
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None

        def run_validation(step: int):
            """Execute validation loop and log metrics."""
            val_losses = []
            for val_data in self.val_dataloader:
                # Use actual batch size from dataloader, not config value
                actual_batch_size = val_data["input_ids"].shape[0]
                val_data = TensorDict(val_data, batch_size=actual_batch_size).to(self.device_name)
                val_loss = self.validation_step(val_data)
                val_losses.append(val_loss)

            metric = None
            # in run_validation() before torch.stack(val_losses)
            if not val_losses:
                if rank == 0:
                    print("Validation skipped: no batches in val_dataloader. Increase val set or reduce batch size.")
                torch.distributed.barrier()
                return None
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}
                tracking.log(data=metric, step=step)

                # Update best val loss
                self._update_metrics(val_loss=metric["val/loss"])

            torch.distributed.barrier()
            return metric

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        start_epoch = global_step // self.steps_per_epoch

        if self.config.trainer.test_freq and self.config.trainer.test_freq > 0:
            if global_step % self.config.trainer.test_freq == 0:
                metric = run_validation(global_step)
                if rank == 0 and metric is not None:
                    last_valid_metric = metric

        # Check if checkpointing is disabled and print warning
        disable_checkpointing = getattr(self.config.trainer, "disable_checkpointing", False)
        if rank == 0 and disable_checkpointing:
            print("=" * 80)
            print("WARNING: Checkpointing is DISABLED (disable_checkpointing=True)")
            print("No checkpoints will be saved during or after training")
            print("=" * 80)

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                # Use actual batch size from dataloader, not global batch size
                actual_batch_size = data["input_ids"].shape[0]
                data = TensorDict(data, batch_size=actual_batch_size).to(self.device_name)
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]

                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    # Update train loss
                    self._update_metrics(train_loss=metric["train/loss"])

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    metric = run_validation(global_step)
                    if rank == 0 and metric is not None:
                        last_valid_metric = metric

                # Check if checkpointing is disabled
                disable_checkpointing = getattr(self.config.trainer, "disable_checkpointing", False)
                if not disable_checkpointing:
                    if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                        self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return

    def save_checkpoint(self, step):
        """Override save_checkpoint to include NLA metadata."""
        import os
        import time

        from verl.utils.checkpoint.nla_metadata import create_sft_metadata_from_config, save_metadata

        # Calculate training time and create metadata BEFORE calling parent
        # so we can save it before HDFS copy
        training_time_hours = None
        if self.training_start_time is not None:
            training_time_hours = (time.time() - self.training_start_time) / 3600.0

        # Get WandB info if available
        wandb_run_id = None
        wandb_run_url = None
        try:
            import wandb

            if wandb.run is not None:
                wandb_run_id = wandb.run.id
                wandb_run_url = wandb.run.get_url()
        except Exception:
            pass  # WandB not initialized or not available

        # Prepare metadata (but don't save yet - directory may not exist)
        metadata = create_sft_metadata_from_config(
            config=self.config,
            global_step=step,
            final_train_loss=self.final_train_loss,
            final_val_loss=self.final_val_loss,
            best_val_loss=self.best_val_loss if self.best_val_loss != float("inf") else None,
            training_time_hours=training_time_hours,
            model_config=self.model_config if hasattr(self, "model_config") else None,
            wandb_run_id=wandb_run_id,
            wandb_run_url=wandb_run_url,
        )

        # Call parent's save_checkpoint (creates directory, saves model, but doesn't copy to HDFS yet)
        # We need to intercept BEFORE the HDFS copy happens
        # Strategy: Save metadata after checkpoint manager but before HDFS copy

        # We'll need to partially duplicate parent logic to insert metadata save at the right point
        from verl.utils.fs import local_mkdir_safe

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {checkpoint_dir}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save (this creates the directory and saves model weights)
        self.checkpoint_manager.save_checkpoint(
            local_path=checkpoint_dir, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state (rank 0 only)
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(checkpoint_dir)
            dataloader_local_path = os.path.join(checkpoint_dir, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            import torch

            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename

            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

            # CRITICAL: Save metadata BEFORE HDFS copy
            save_metadata(checkpoint_dir, metadata)
            print(f"Saved NLA metadata for step {step}")

        # Copy to HDFS if configured (now includes metadata!)
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            import verl.utils.hdfs_io as hdfs_io

            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=checkpoint_dir, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
            print(f"Copied checkpoint (with metadata) to HDFS: {self.config.trainer.default_hdfs_dir}")

        import torch

        torch.distributed.barrier()

    def _update_metrics(self, train_loss: float = None, val_loss: float = None):
        """Update tracked metrics."""
        if train_loss is not None:
            self.final_train_loss = train_loss

        if val_loss is not None:
            self.final_val_loss = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
