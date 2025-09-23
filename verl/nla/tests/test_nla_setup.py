#!/usr/bin/env python
"""Test script to verify NLA setup with fake dataset and tiny model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf
import pandas as pd
import sys
import os

# Add verl to path
sys.path.insert(0, 'verl')

from verl.verl.nla.data.nla_sft_dataset import NLASFTDataset, NLASFTCollator


def test_dataset_loading():
    """Test loading the fake NLA dataset."""
    print("=" * 50)
    print("Testing Dataset Loading...")
    print("=" * 50)

    # Load the parquet file
    df = pd.read_parquet("test_nla_dataset.parquet")
    print(f"✓ Loaded {len(df)} samples from parquet")
    print(f"✓ Columns: {df.columns.tolist()}")

    # Check activation vector
    first_activation = df.iloc[0]["activation_vector"]
    print(f"✓ Activation vector length: {len(first_activation)}")
    print(f"✓ Sample prompt: {df.iloc[0]['prompt'][:50]}...")
    print(f"✓ Sample response: {df.iloc[0]['response'][:50]}...")

    return df


def test_model_loading():
    """Test loading the tiny Gemma model."""
    print("\n" + "=" * 50)
    print("Testing Model Loading...")
    print("=" * 50)

    model_name = "yujiepan/gemma-2-tiny-random"

    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Tokenizer loaded")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - EOS token: {tokenizer.eos_token}")
    print(f"  - PAD token: {tokenizer.pad_token}")

    # Load model
    print(f"\nLoading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    print(f"✓ Model loaded")
    print(f"  - Model type: {model.config.model_type}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    print(f"  - Num heads: {model.config.num_attention_heads}")

    return tokenizer, model


def test_nla_dataset_class():
    """Test the NLA dataset class with fake data."""
    print("\n" + "=" * 50)
    print("Testing NLA Dataset Class...")
    print("=" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "yujiepan/gemma-2-tiny-random",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create config
    config = OmegaConf.create({
        "activation_dim": 128,
        "injection_token": "<INJECT>",
        "max_length": 512,
        "truncation": "right",
        "padding": "right",
    })

    # Test actor mode
    print("\nTesting Actor mode...")
    actor_dataset = NLASFTDataset(
        parquet_files="test_nla_dataset.parquet",
        tokenizer=tokenizer,
        config=config,
        mode="actor"
    )

    sample = actor_dataset[0]
    print(f"✓ Actor sample keys: {sample.keys()}")
    print(f"  - input_ids shape: {sample['input_ids'].shape}")
    print(f"  - activation_vectors shape: {sample['activation_vectors'].shape}")

    # Test critic mode
    print("\nTesting Critic mode...")
    critic_dataset = NLASFTDataset(
        parquet_files="test_nla_dataset.parquet",
        tokenizer=tokenizer,
        config=config,
        mode="critic"
    )

    sample = critic_dataset[0]
    print(f"✓ Critic sample keys: {sample.keys()}")
    print(f"  - response_ids shape: {sample['response_ids'].shape}")
    print(f"  - activation_vectors shape: {sample['activation_vectors'].shape}")

    # Test collator
    print("\nTesting Collator...")
    collator = NLASFTCollator(pad_token_id=tokenizer.pad_token_id)

    batch = collator([actor_dataset[i] for i in range(2)])
    print(f"✓ Batch keys: {batch.keys()}")
    print(f"  - Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  - Batch activation_vectors shape: {batch['activation_vectors'].shape}")


def test_generation():
    """Test a simple generation with the model."""
    print("\n" + "=" * 50)
    print("Testing Model Generation...")
    print("=" * 50)

    tokenizer, model = test_model_loading()

    # Test prompt
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"\nPrompt: {prompt}")
    print("Generating response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generated: {response}")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("NLA SETUP VERIFICATION")
    print("=" * 50)

    try:
        # Test dataset loading
        df = test_dataset_loading()

        # Test NLA dataset class
        test_nla_dataset_class()

        # Optional: test model generation (comment out if slow)
        # test_generation()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYour NLA setup is ready for training:")
        print("- Dataset: test_nla_dataset.parquet (10 samples)")
        print("- Config: verl/verl/nla/configs/test_nla_config.yaml")
        print("- Model: yujiepan/gemma-2-tiny-random")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()