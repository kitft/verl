"""Tests for NLA model wrapper."""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from verl.nla.models import NLAModelWrapper, InjectionConfig
from verl.nla.utils.injection_manager import InjectionTokenManager


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, hidden_size=768, vocab_size=32000):
        super().__init__()
        self.config = type("config", (), {"hidden_size": hidden_size})()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(4)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        logits = self.lm_head(hidden_states)
        return type("output", (), {"logits": logits})()


class TestNLAModelWrapper:
    """Test suite for NLAModelWrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        model = DummyModel()
        # Use a real tokenizer - GPT2 is small and fast
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(mode="replace", layer_indices=[0])

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
            hidden_dim=768,
        )

        assert wrapper.hidden_dim == 768
        assert wrapper.activation_dim == 768
        assert wrapper.injection_config.mode == "replace"
        assert wrapper.injection_config.layer_indices == [0]

    def test_hidden_dim_inference(self):
        """Test automatic hidden dimension inference."""
        model = DummyModel(hidden_size=1024)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        wrapper = NLAModelWrapper(base_model=model, tokenizer=tokenizer)

        assert wrapper.hidden_dim == 1024

    def test_injection_position_finding(self):
        """Test finding injection positions in input."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            injection_token_id=100,  # Arbitrary token ID
        )

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
        )

        # Create input with injection tokens
        input_ids = torch.tensor([
            [1, 2, 100, 4, 5],  # Injection at position 2
            [100, 2, 3, 100, 5],  # Injection at positions 0 and 3
            [1, 2, 3, 4, 5],  # No injection
        ])

        positions = wrapper._find_injection_positions(input_ids)

        assert torch.equal(positions[0], torch.tensor([2]))
        assert torch.equal(positions[1], torch.tensor([0, 3]))
        assert positions[2] is None

    def test_forward_without_injection(self):
        """Test forward pass without activation injection."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Provide tokenizer to avoid the error
        wrapper = NLAModelWrapper(base_model=model, tokenizer=tokenizer)

        input_ids = torch.randint(0, 100, (2, 10))

        # Should work like normal model
        output = wrapper(input_ids=input_ids)

        assert hasattr(output, "logits")
        assert output.logits.shape == (2, 10, 32000)

    def test_forward_with_injection(self):
        """Test forward pass with activation injection."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            injection_token_id=100,
        )

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
            hidden_dim=768,
        )

        # Input with injection token
        input_ids = torch.tensor([[1, 2, 100, 4, 5]])
        activation_vectors = torch.randn(1, 768)

        output = wrapper(
            input_ids=input_ids,
            activation_vectors=activation_vectors,
        )

        assert hasattr(output, "logits")

    def test_projection_layer(self):
        """Test learnable projection layer."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="project",
            projection_dim=512,
            injection_token_id=999,
        )

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
            hidden_dim=768,
            activation_dim=256,  # Different from hidden dim
        )

        assert wrapper.activation_proj is not None
        assert wrapper.activation_proj.in_features == 256
        assert wrapper.activation_proj.out_features == 512

    def test_injection_at_layer(self):
        """Test injection at specific layer."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="replace",
            injection_token_id=100,
        )

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
        )

        # Setup test data
        hidden_states = torch.randn(2, 5, 768)
        activation_vector = torch.randn(2, 768)

        wrapper._current_activations = activation_vector
        wrapper._injection_positions = [torch.tensor([2]), torch.tensor([0, 4])]

        # Perform injection
        result = wrapper._inject_at_layer(hidden_states, 0)

        # Check that injection happened at correct positions
        assert torch.allclose(result[0, 2], activation_vector[0])
        assert torch.allclose(result[1, 0], activation_vector[1])
        assert torch.allclose(result[1, 4], activation_vector[1])

    def test_additive_injection(self):
        """Test additive injection mode."""
        model = DummyModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = InjectionConfig(
            mode="add",
            injection_token_id=100,
        )

        wrapper = NLAModelWrapper(
            base_model=model,
            tokenizer=tokenizer,
            injection_config=config,
        )

        hidden_states = torch.ones(1, 3, 768)
        activation_vector = torch.ones(1, 768) * 0.5

        wrapper._current_activations = activation_vector
        wrapper._injection_positions = [torch.tensor([1])]

        result = wrapper._inject_at_layer(hidden_states, 0)

        # Check additive injection
        assert torch.allclose(result[0, 1], torch.ones(768) * 1.5)
        assert torch.allclose(result[0, 0], torch.ones(768))
        assert torch.allclose(result[0, 2], torch.ones(768))


if __name__ == "__main__":
    # Run tests
    test_suite = TestNLAModelWrapper()
    test_suite.test_initialization()
    test_suite.test_hidden_dim_inference()
    test_suite.test_injection_position_finding()
    test_suite.test_forward_without_injection()
    test_suite.test_forward_with_injection()
    test_suite.test_projection_layer()
    test_suite.test_injection_at_layer()
    test_suite.test_additive_injection()
    print("All tests passed!")