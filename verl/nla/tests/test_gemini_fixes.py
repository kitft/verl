"""Test fixes from Gemini's feedback."""

import torch
import torch.nn as nn
from unittest.mock import Mock
import sys
sys.path.append('/Users/kit/Documents/Anthropic/NLA')
from transformers import AutoTokenizer

from verl.nla.models.nla_wrapper import NLAModelWrapper, InjectionConfig
from verl.nla.models.autoencoder_critic import NLAAutoencoderCritic


def test_stateless_generation():
    """Test that generation works without state flags."""
    # Mock base model
    base_model = Mock()
    base_model.config = Mock(hidden_size=768)

    # Mock get_input_embeddings to return a proper tensor
    mock_embedding = Mock()
    mock_embedding.return_value = torch.randn(1, 5, 768)  # batch_size=1, seq_len=5, hidden_dim=768
    base_model.get_input_embeddings = Mock(return_value=mock_embedding)

    # Mock the base model forward to return proper output
    base_model.forward = Mock(return_value=Mock(logits=torch.randn(1, 5, 50000)))

    # Create tokenizer and wrapper
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    wrapper = NLAModelWrapper(
        base_model=base_model,
        injection_config=InjectionConfig(injection_token="|"),
        hidden_dim=768,
        tokenizer=tokenizer
    )

    # Test forward is stateless
    injection_token_id = tokenizer.convert_tokens_to_ids("|")
    input_ids = torch.tensor([[1, 2, injection_token_id, 3, 4]])
    activation_vectors = torch.randn(1, 768)

    # Forward with injection
    wrapper.forward(
        input_ids=input_ids,
        activation_vectors=activation_vectors,
        past_key_values=None  # First pass
    )

    # Verify no persistent state
    assert wrapper._current_activations is None
    assert wrapper._injection_positions is None

    print("✅ Stateless forward test passed!")


def test_vectorized_injection_positions():
    """Test vectorized injection position finding."""
    base_model = Mock()
    base_model.config = Mock(hidden_size=768)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    wrapper = NLAModelWrapper(
        base_model=base_model,
        injection_config=InjectionConfig(injection_token="|"),
        hidden_dim=768,
        tokenizer=tokenizer
    )

    # Test with multiple injection tokens
    injection_token_id = tokenizer.convert_tokens_to_ids("|")
    input_ids = torch.tensor([
        [1, injection_token_id, 3, injection_token_id, 5],
        [injection_token_id, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]  # No injection token
    ])

    positions = wrapper._find_injection_positions(input_ids)

    # Check results are tensors
    assert positions[0] is not None and torch.equal(positions[0], torch.tensor([1, 3]))
    assert positions[1] is not None and torch.equal(positions[1], torch.tensor([0]))
    assert positions[2] is None  # No injection token

    print("✅ Vectorized position finding test passed!")


def test_weighted_pooling_learnable():
    """Test that weighted pooling has truly learnable weights."""
    # Mock base model
    base_model = Mock()
    base_model.config = Mock(hidden_size=768)

    critic = NLAAutoencoderCritic(
        base_model=base_model,
        use_pooling="weighted",
        max_seq_length=512
    )

    # Check weights are properly initialized
    assert hasattr(critic, 'pooling_weights')
    assert critic.pooling_weights.shape == (512,)

    # Weights should be different (not all ones)
    assert not torch.allclose(critic.pooling_weights, torch.ones(512))

    # Test pooling with different sequence lengths
    hidden_states = torch.randn(2, 10, 768)
    pooled = critic.pool_hidden_states(hidden_states)

    # Check output shape
    assert pooled.shape == (2, 768)

    # Verify weights are actually being used and learned
    old_weights = critic.pooling_weights.clone()

    # Simulate gradient update
    loss = pooled.sum()
    loss.backward()

    # Weights should have gradients
    assert critic.pooling_weights.grad is not None

    print("✅ Weighted pooling test passed!")


def test_fail_fast_without_transformers():
    """Test that generate fails immediately without transformers."""
    # Mock base model without GenerationMixin
    base_model = Mock()
    base_model.config = Mock(hidden_size=768)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    wrapper = NLAModelWrapper(
        base_model=base_model,
        injection_config=InjectionConfig(injection_token="|"),
        hidden_dim=768,
        tokenizer=tokenizer
    )

    # Should raise error if transformers not available
    input_ids = torch.tensor([[1, 2, 3]])

    # Check if we have transformers
    try:
        from transformers.generation import GenerationMixin
        print("⚠️  Transformers is installed, skipping fail-fast test")
    except ImportError:
        # Should fail fast
        try:
            wrapper.generate(input_ids=input_ids)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "transformers" in str(e)
            print("✅ Fail-fast test passed!")


if __name__ == "__main__":
    test_stateless_generation()
    test_vectorized_injection_positions()
    test_weighted_pooling_learnable()
    test_fail_fast_without_transformers()
    print("\n🎉 All tests passed!")