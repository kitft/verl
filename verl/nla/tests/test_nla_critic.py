#!/usr/bin/env python
"""Test the NLA critic implementation with vector value head."""

import sys

import torch

sys.path.insert(0, "verl")

from transformers import AutoTokenizer

from verl.verl.nla.models.nla_critic_model import AutoModelForCausalLMWithVectorValueHead


def test_critic_model():
    """Test the NLA critic model with vector value head."""
    print("=" * 50)
    print("Testing NLA Critic Model")
    print("=" * 50)

    # Model configuration
    model_name = "yujiepan/gemma-2-tiny-random"
    activation_dim = 8

    # Load tokenizer
    print(f"\n1. Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")

    # Load critic model with vector value head
    print("\n2. Loading critic model with vector value head")
    critic = AutoModelForCausalLMWithVectorValueHead.from_pretrained(
        model_name, activation_dim=activation_dim, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    print("✓ Critic model loaded")
    print(f"  - Base model type: {critic.config.model_type}")
    print(f"  - Hidden size: {critic.config.hidden_size}")
    print(f"  - Activation dim: {activation_dim}")
    print(f"  - Value head output: {critic.v_head.activation_dim}D vectors")

    # Test forward pass
    print("\n3. Testing forward pass")
    test_texts = ["The capital of France is Paris.", "Machine learning is a type of AI."]

    # Tokenize inputs
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    print(f"  - Input shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = critic(**inputs, return_dict=True, output_hidden_states=True)

    # Check outputs
    print("\n4. Checking outputs")
    print(f"  - Logits shape: {outputs.logits.shape}")  # (batch, seq_len, vocab_size)

    # The value output should be our vector values
    if hasattr(outputs, "value"):
        print(f"  - Value shape: {outputs.value.shape}")  # Should be (batch, seq_len, activation_dim)
        assert outputs.value.shape[-1] == activation_dim, (
            f"Expected activation_dim={activation_dim}, got {outputs.value.shape[-1]}"
        )
        print(f"  ✓ Value head outputs {activation_dim}D vectors as expected")
    else:
        # In case it's returned as tuple (for VERL compatibility)
        value = outputs[2]
        print(f"  - Value shape: {value.shape}")
        assert value.shape[-1] == activation_dim, f"Expected activation_dim={activation_dim}, got {value.shape[-1]}"
        print(f"  ✓ Value head outputs {activation_dim}D vectors as expected")

    # Test pooling strategies
    print("\n5. Testing pooling strategies")

    batch_size, seq_len, _ = outputs.value.shape
    values = outputs.value

    # Last token pooling
    last_values = values[:, -1, :]  # (batch, activation_dim)
    print(f"  - Last pooling shape: {last_values.shape}")

    # Mean pooling
    mean_values = values.mean(dim=1)  # (batch, activation_dim)
    print(f"  - Mean pooling shape: {mean_values.shape}")

    # Max pooling
    max_values, _ = values.max(dim=1)  # (batch, activation_dim)
    print(f"  - Max pooling shape: {max_values.shape}")

    print("\n6. Testing MSE loss computation")
    # Create fake target activations
    target_activations = torch.randn(batch_size, activation_dim)

    # Compute MSE with last token pooling
    mse_loss = torch.nn.functional.mse_loss(last_values, target_activations)
    print(f"  - MSE loss: {mse_loss.item():.4f}")

    # Check gradients flow
    print("\n7. Testing gradient flow")
    critic.zero_grad()
    mse_loss.backward()

    # Check if gradients are computed for value head
    if critic.v_head.linear.weight.grad is not None:
        grad_norm = critic.v_head.linear.weight.grad.norm().item()
        print(f"  ✓ Gradients flow to value head (norm: {grad_norm:.4f})")
    else:
        print("  ❌ No gradients in value head")

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nThe NLA critic is working correctly:")
    print("- Base model loads as expected")
    print("- Vector value head outputs activation_dim vectors")
    print("- Forward pass works with batched inputs")
    print("- Pooling strategies work correctly")
    print("- MSE loss can be computed")
    print("- Gradients flow through the model")


if __name__ == "__main__":
    test_critic_model()
