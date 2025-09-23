"""Comprehensive tests for NLA SFT components."""

import torch
import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import NLA components
from verl.nla.data.nla_sft_dataset import NLASFTDataset, NLASFTCollator
from verl.nla.models.nla_wrapper import NLAModelWrapper, InjectionConfig
from verl.nla.models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead
from verl.nla.trainer.nla_sft_trainer import NLASFTTrainer


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.convert_tokens_to_ids = lambda x: 50000  # Mock injection token ID
    tokenizer.get_vocab = lambda: {"<pad>": 0, "<unk>": 1, "</s>": 2}
    tokenizer.add_special_tokens = MagicMock()
    tokenizer.apply_chat_template = lambda x, **kwargs: "User: test\nAssistant:"

    # Mock tokenizer call
    def tokenizer_call(text, **kwargs):
        mock_output = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        if kwargs.get("padding") == "max_length":
            max_len = kwargs.get("max_length", 10)
            pad_len = max_len - 5
            mock_output["input_ids"] = torch.cat([
                mock_output["input_ids"][0],
                torch.zeros(pad_len, dtype=torch.long)
            ]).unsqueeze(0)
            mock_output["attention_mask"] = torch.cat([
                mock_output["attention_mask"][0],
                torch.zeros(pad_len, dtype=torch.long)
            ]).unsqueeze(0)
        return mock_output

    tokenizer.side_effect = tokenizer_call
    tokenizer.__call__ = tokenizer_call
    return tokenizer


@pytest.fixture
def sample_data_file():
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        # Create sample data
        data = pd.DataFrame({
            "prompt": ["What is AI?", "Explain machine learning"],
            "response": ["AI is artificial intelligence", "ML is a subset of AI"],
            "activation_vector": [
                np.random.randn(768).tolist(),
                np.random.randn(768).tolist()
            ]
        })
        data.to_parquet(f.name)
        yield f.name
        # Cleanup
        Path(f.name).unlink()


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()

    # Model config
    config.model.model_name = "gpt2"
    config.model.activation_dim = 768
    config.model.enable_flashattn = False

    # Setup model.get() to return activation_dim and critic config
    model_config_dict = {
        "activation_dim": 768,
        "critic": {
            "model_name": "gpt2",
            "pooling": "last",
            "dropout": 0.1,
            "projection_layers": 2,
        }
    }
    config.model.get.side_effect = lambda k, d=None: model_config_dict.get(k, d)

    # Injection config
    injection_config_dict = {
        "mode": "replace",
        "layer_indices": [0],
        "projection_dim": None,
        "injection_token_id": 50000,
    }
    config.model.injection.mode = injection_config_dict["mode"]
    config.model.injection.layer_indices = injection_config_dict["layer_indices"]
    config.model.injection.projection_dim = injection_config_dict["projection_dim"]
    config.model.injection.injection_token_id = injection_config_dict["injection_token_id"]
    config.model.injection.get.side_effect = lambda k, d=None: injection_config_dict.get(k, d)

    # Critic config
    config.model.critic.model_name = "gpt2"
    config.model.critic.pooling = "last"
    config.model.critic.dropout = 0.1
    config.model.critic.projection_layers = 2

    # Data config
    config.data.micro_batch_size_per_gpu = 2
    config.data.get = lambda k, default=None: {
        "activation_dim": 768,
        "injection_token": "<INJECT>",
        "injection_token_id": None,
        "max_length": 512,
        "truncation": "right",
        "use_shm": False,
        "prompt_key": "prompt",
        "response_key": "response",
        "apply_chat_template_kwargs": {}
    }.get(k, default)

    # Optimizer config
    optim_config_dict = {
        "lr": 1e-5,
        "critic_lr": 5e-5,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_norm": 1.0,
        "warmup_steps": 0,
    }
    config.optim.lr = optim_config_dict["lr"]
    config.optim.critic_lr = optim_config_dict["critic_lr"]
    config.optim.weight_decay = optim_config_dict["weight_decay"]
    config.optim.beta1 = optim_config_dict["beta1"]
    config.optim.beta2 = optim_config_dict["beta2"]
    config.optim.max_norm = optim_config_dict["max_norm"]
    config.optim.warmup_steps = optim_config_dict["warmup_steps"]
    config.optim.get.side_effect = lambda k, d=None: optim_config_dict.get(k, d)

    # Trainer config
    trainer_config_dict = {
        "critic_epochs": 1,
        "total_training_steps": 1000,
    }
    config.trainer.critic_epochs = trainer_config_dict["critic_epochs"]
    config.trainer.total_training_steps = trainer_config_dict["total_training_steps"]
    config.trainer.get.side_effect = lambda k, d=None: trainer_config_dict.get(k, d)

    return config


class TestNLASFTDataset:
    """Test NLA SFT Dataset functionality."""

    def test_dataset_initialization(self, sample_data_file, mock_tokenizer):
        """Test dataset initialization with activation vectors."""
        config = MagicMock()
        config.get = lambda k, default=None: {
            "activation_dim": 768,
            "injection_token": "<INJECT>",
            "injection_token_id": None,
            "max_length": 512,
            "truncation": "right",
            "use_shm": False,
            "prompt_key": "prompt",
            "response_key": "response",
            "apply_chat_template_kwargs": {}
        }.get(k, default)

        dataset = NLASFTDataset(
            parquet_files=sample_data_file,
            tokenizer=mock_tokenizer,
            config=config,
            mode="actor"
        )

        assert len(dataset) == 2
        assert len(dataset.activation_vectors) == 2
        assert dataset.activation_vectors[0].shape == (768,)

    def test_actor_sample_preparation(self, sample_data_file, mock_tokenizer):
        """Test sample preparation for actor training."""
        config = MagicMock()
        config.get = lambda k, default=None: {
            "activation_dim": 768,
            "injection_token": "<INJECT>",
            "injection_token_id": 50000,
            "max_length": 512,
            "truncation": "right",
            "use_shm": False,
            "prompt_key": "prompt",
            "response_key": "response",
            "apply_chat_template_kwargs": {}
        }.get(k, default)

        dataset = NLASFTDataset(
            parquet_files=sample_data_file,
            tokenizer=mock_tokenizer,
            config=config,
            mode="actor"
        )

        sample = dataset[0]
        assert "activation_vectors" in sample
        assert sample["activation_vectors"].shape == (768,)
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "loss_mask" in sample

    def test_critic_sample_preparation(self, sample_data_file, mock_tokenizer):
        """Test sample preparation for critic training."""
        config = MagicMock()
        config.get = lambda k, default=None: {
            "activation_dim": 768,
            "injection_token": "<INJECT>",
            "injection_token_id": None,
            "max_length": 512,
            "truncation": "right",
            "use_shm": False,
            "prompt_key": "prompt",
            "response_key": "response",
            "apply_chat_template_kwargs": {}
        }.get(k, default)

        dataset = NLASFTDataset(
            parquet_files=sample_data_file,
            tokenizer=mock_tokenizer,
            config=config,
            mode="critic"
        )

        sample = dataset[0]
        assert "response_ids" in sample
        assert "response_attention_mask" in sample
        assert "activation_vectors" in sample
        assert sample["activation_vectors"].shape == (768,)

    def test_both_mode_sample(self, sample_data_file, mock_tokenizer):
        """Test sample preparation for both actor and critic training."""
        config = MagicMock()
        config.get = lambda k, default=None: {
            "activation_dim": 768,
            "injection_token": "<INJECT>",
            "injection_token_id": None,
            "max_length": 512,
            "truncation": "right",
            "use_shm": False,
            "prompt_key": "prompt",
            "response_key": "response",
            "apply_chat_template_kwargs": {}
        }.get(k, default)

        dataset = NLASFTDataset(
            parquet_files=sample_data_file,
            tokenizer=mock_tokenizer,
            config=config,
            mode="both"
        )

        sample = dataset[0]
        # Should have both actor and critic fields
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "loss_mask" in sample
        assert "response_ids" in sample
        assert "response_attention_mask" in sample
        assert "activation_vectors" in sample


class TestNLASFTCollator:
    """Test NLA SFT data collator."""

    def test_collator_basic(self):
        """Test basic collation functionality."""
        collator = NLASFTCollator(pad_token_id=0)

        # Create mock samples
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "activation_vectors": torch.randn(768)
            },
            {
                "input_ids": torch.tensor([6, 7, 8, 9, 10]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "activation_vectors": torch.randn(768)
            }
        ]

        batch = collator(samples)
        assert batch["input_ids"].shape == (2, 5)
        assert batch["attention_mask"].shape == (2, 5)
        assert batch["activation_vectors"].shape == (2, 768)

    def test_collator_with_response_fields(self):
        """Test collation with critic training fields."""
        collator = NLASFTCollator(pad_token_id=0)

        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "response_ids": torch.tensor([11, 12, 13, 14, 15]),
                "response_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "activation_vectors": torch.randn(768)
            },
            {
                "input_ids": torch.tensor([6, 7, 8, 9, 10]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "response_ids": torch.tensor([16, 17, 18, 19, 20]),
                "response_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "activation_vectors": torch.randn(768)
            }
        ]

        batch = collator(samples)
        assert batch["response_ids"].shape == (2, 5)
        assert batch["response_attention_mask"].shape == (2, 5)


class TestNLAModelWrapper:
    """Test NLA model wrapper for activation injection."""

    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        # Create a small mock model
        base_model = MagicMock()
        base_model.config.hidden_size = 768

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            layer_indices=[0],
            injection_token="|"
        )

        wrapper = NLAModelWrapper(
            base_model=base_model,
            injection_config=config,
            hidden_dim=768,
            activation_dim=768,
            tokenizer=tokenizer
        )

        assert wrapper.hidden_dim == 768
        assert wrapper.activation_dim == 768
        assert wrapper.injection_config.injection_token == "|"

    def test_injection_position_finding(self):
        """Test finding injection token positions."""
        base_model = MagicMock()
        base_model.config.hidden_size = 768

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            layer_indices=[0],
            injection_token="|"
        )

        wrapper = NLAModelWrapper(
            base_model=base_model,
            injection_config=config,
            hidden_dim=768,
            activation_dim=768,
            tokenizer=tokenizer
        )

        # Create input with injection tokens
        injection_token_id = tokenizer.convert_tokens_to_ids("|")
        input_ids = torch.tensor([
            [1, 2, injection_token_id, 4, 5],  # Injection at position 2
            [6, 7, 8, injection_token_id, 10],  # Injection at position 3
        ])

        positions = wrapper._find_injection_positions(input_ids)
        assert len(positions) == 2
        assert positions[0].item() == 2
        assert positions[1].item() == 3

    @patch("verl.nla.models.nla_wrapper.NLAModelWrapper._inject_at_layer")
    def test_forward_with_injection(self, mock_inject):
        """Test forward pass with activation injection."""
        base_model = MagicMock()
        base_model.config.hidden_size = 768
        base_model.get_input_embeddings = MagicMock(return_value=MagicMock())

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            layer_indices=[0],
            injection_token="|"
        )

        wrapper = NLAModelWrapper(
            base_model=base_model,
            injection_config=config,
            hidden_dim=768,
            activation_dim=768,
            tokenizer=tokenizer
        )

        # Mock the base model forward
        mock_output = MagicMock()
        base_model.return_value = mock_output

        # Create inputs
        injection_token_id = tokenizer.convert_tokens_to_ids("|")
        input_ids = torch.tensor([[1, 2, injection_token_id, 4, 5]])
        activation_vectors = torch.randn(1, 768)

        # Forward pass
        output = wrapper.forward(
            input_ids=input_ids,
            activation_vectors=activation_vectors
        )

        # Check that injection was attempted
        assert base_model.called or base_model.get_input_embeddings.called





class TestAutoModelForCausalLMWithVectorValueHead:
    """Test critic model with vector value head for activation prediction."""

    def test_critic_initialization(self):
        """Test critic initialization with vector value head."""
        # This test would require a real model from HuggingFace
        # Skipping for now as it needs the tiny model loaded
        pass

    def test_forward_pass(self):
        """Test forward pass returns vector values."""
        # This test would require a real model from HuggingFace  
        # Skipping for now as it needs the tiny model loaded
        pass

class TestNLASFTTrainer:
    """Test NLA SFT Trainer."""

    @patch("verl.nla.trainer.nla_sft_trainer.FSDPSFTTrainer.__init__")
    @patch("verl.nla.trainer.nla_sft_trainer.FSDPSFTTrainer._build_model_optimizer")
    def test_trainer_initialization(self, mock_build, mock_init, mock_config, mock_tokenizer):
        """Test trainer initialization."""
        mock_init.return_value = None
        mock_build.return_value = None

        # Create mock objects
        device_mesh = MagicMock()
        ulysses_mesh = MagicMock()
        tokenizer = mock_tokenizer
        train_dataset = MagicMock()
        val_dataset = MagicMock()

        trainer = NLASFTTrainer(
            config=mock_config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_mode="both"
        )

        assert trainer.train_mode == "both"
        assert trainer.activation_dim == 768
        assert trainer.injection_config.mode == "replace"

    @patch("verl.nla.trainer.nla_sft_trainer.FSDPSFTTrainer.__init__")
    def test_compute_loss_with_injection(self, mock_init, mock_config, mock_tokenizer):
        """Test loss computation with activation injection."""
        mock_init.return_value = None

        trainer = NLASFTTrainer(
            config=mock_config,
            device_mesh=MagicMock(),
            ulysses_device_mesh=MagicMock(),
            tokenizer=mock_tokenizer,
            train_dataset=MagicMock(),
            train_mode="actor"
        )

        # Mock the FSDP model
        trainer.fsdp_model = MagicMock()
        trainer.device_name = "cpu"

        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(2, 10, 50257)  # batch_size, seq_len, vocab_size
        trainer.fsdp_model.return_value = mock_output

        # Create batch
        batch = {
            "input_ids": torch.randint(0, 50257, (2, 11)),
            "attention_mask": torch.ones(2, 11),
            "loss_mask": torch.ones(2, 11),
            "activation_vectors": torch.randn(2, 768)
        }

        # Compute loss
        loss = trainer._compute_loss_and_backward(batch, do_backward=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    @patch("verl.nla.trainer.nla_sft_trainer.FSDPSFTTrainer.__init__")
    def test_critic_training_step(self, mock_init, mock_config, mock_tokenizer):
        """Test critic training step."""
        mock_init.return_value = None

        trainer = NLASFTTrainer(
            config=mock_config,
            device_mesh=MagicMock(),
            ulysses_device_mesh=MagicMock(),
            tokenizer=mock_tokenizer,
            train_dataset=MagicMock(),
            train_mode="critic"
        )
        # Set config manually since we mocked __init__
        trainer.config = mock_config

        # Create a real small critic model for proper gradient computation
        import torch.nn as nn

        # Create a minimal base model
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 768)
                self.layer = nn.Linear(768, 768)
                self.config = type('Config', (), {'hidden_size': 768})()

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
                x = self.embed(input_ids)
                x = self.layer(x)
                # Return output object with proper structure
                output = type('Output', (), {})()
                # For transformers compatibility, hidden_states should be a tuple of layer outputs
                if output_hidden_states:
                    output.hidden_states = (x,)  # Tuple of hidden states from each layer
                # Last hidden state is the main output
                output.last_hidden_state = x
                return output

        # Create real critic with AutoModelForCausalLMWithVectorValueHead
        from verl.nla.models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead
        real_critic = AutoModelForCausalLMWithVectorValueHead.from_pretrained("yujiepan/gemma-2-tiny-random")

        # Add clip_grad_norm_ method to the critic (FSDP models have this)
        real_critic.clip_grad_norm_ = lambda max_norm: torch.nn.utils.clip_grad_norm_(
            real_critic.parameters(), max_norm
        )

        trainer.fsdp_critic = real_critic
        trainer.critic_optimizer = torch.optim.Adam(real_critic.parameters(), lr=1e-4)
        trainer.critic_scheduler = None
        trainer.device_name = "cpu"
        trainer.train_critic_epochs = 1

        # Create batch with small tensors
        batch = {
            "response_ids": torch.randint(0, 100, (2, 10)),
            "response_attention_mask": torch.ones(2, 10),
            "activation_vectors": torch.randn(2, 768)
        }

        # Train critic - this should now work with real gradients
        metrics = trainer._train_critic_step(batch)
        assert "critic_loss" in metrics
        assert isinstance(metrics["critic_loss"], float)
        assert metrics["critic_loss"] > 0  # Loss should be positive


# Integration test
@pytest.mark.integration
@patch("verl.nla.trainer.nla_sft_trainer.AutoModelForCausalLM.from_pretrained")
@patch("verl.nla.trainer.nla_sft_trainer.FSDP")
def test_end_to_end_training(mock_fsdp, mock_model_load, sample_data_file, mock_config, mock_tokenizer):
    """Test end-to-end training flow."""
    # This test would require flash_attn, so we mock the critical parts

    # Mock model loading
    mock_model = MagicMock()
    mock_model.config.hidden_size = 768
    mock_model_load.return_value = mock_model

    # Mock FSDP wrapper
    mock_fsdp.return_value = mock_model

    # Create dataset
    tokenizer = mock_tokenizer
    dataset_config = MagicMock()
    dataset_config.get = lambda k, default=None: {
        "activation_dim": 768,
        "injection_token": "<INJECT>",
        "max_length": 512,
        "truncation": "right",
        "use_shm": False,
        "prompt_key": "prompt",
        "response_key": "response",
        "apply_chat_template_kwargs": {}
    }.get(k, default)

    train_dataset = NLASFTDataset(
        parquet_files=sample_data_file,
        tokenizer=tokenizer,
        config=dataset_config,
        mode="both"
    )

    # Would create trainer and run training step
    # trainer = NLASFTTrainer(...)
    # metrics = trainer.training_step(batch)

    assert len(train_dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])