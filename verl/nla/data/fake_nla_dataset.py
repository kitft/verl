"""Generate fake NLA dataset for testing."""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def generate_fake_nla_dataset(
    num_samples: int = 100,
    activation_dim: int = 768,
    output_path: str = "fake_nla_dataset.parquet"
):
    """
    Generate a fake NLA dataset with prompts, responses, and activation vectors.

    Args:
        num_samples: Number of samples to generate
        activation_dim: Dimension of activation vectors
        output_path: Path to save the parquet file
    """

    # Sample prompts and responses
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about mountains.",
        "How do neural networks learn?",
        "What causes seasons on Earth?",
        "Describe the water cycle.",
        "What is machine learning?",
        "Explain photosynthesis.",
        "What is the theory of relativity?",
        "How do computers work?",
    ]

    responses = [
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits (qubits) that can be in multiple states simultaneously, unlike classical bits.",
        "Mountains stand tall,\nSnow-capped peaks touch morning clouds,\nSilent strength endures.",
        "Neural networks learn by adjusting weights through backpropagation to minimize prediction errors.",
        "Seasons are caused by Earth's tilted axis as it orbits the Sun, changing the angle of sunlight.",
        "Water evaporates, forms clouds, falls as precipitation, and flows back to water bodies.",
        "Machine learning is a type of AI where computers learn patterns from data without explicit programming.",
        "Plants convert sunlight, water, and CO2 into glucose and oxygen through photosynthesis.",
        "Einstein's theory states that space and time are interconnected, and gravity warps spacetime.",
        "Computers process information using binary code through transistors in integrated circuits.",
    ]

    # Generate dataset
    data = []
    for i in range(num_samples):
        # Cycle through prompts and responses
        prompt_idx = i % len(prompts)

        # Generate random activation vector
        # Use different distributions for variety
        if i % 3 == 0:
            # Normal distribution
            activation = np.random.randn(activation_dim) * 0.1
        elif i % 3 == 1:
            # Uniform distribution
            activation = np.random.uniform(-0.5, 0.5, activation_dim)
        else:
            # Sparse activation (some zeros)
            activation = np.random.randn(activation_dim) * 0.1
            mask = np.random.random(activation_dim) > 0.3
            activation = activation * mask

        # Normalize activation vector
        activation = activation / (np.linalg.norm(activation) + 1e-8)

        data.append({
            "prompt": prompts[prompt_idx],
            "response": responses[prompt_idx],
            "activation_vector": activation.tolist(),  # Store as list for parquet
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)

    print(f"Generated {num_samples} samples")
    print(f"Saved to {output_path}")
    print(f"Activation dimension: {activation_dim}")

    # Print sample
    print("\nSample data:")
    print(f"Prompt: {df.iloc[0]['prompt']}")
    print(f"Response: {df.iloc[0]['response']}")
    print(f"Activation vector shape: {len(df.iloc[0]['activation_vector'])}")
    print(f"Activation vector (first 5 values): {df.iloc[0]['activation_vector'][:5]}")

    return df


def generate_small_test_dataset():
    """Generate a very small dataset for quick testing."""
    return generate_fake_nla_dataset(
        num_samples=10,
        activation_dim=128,  # Smaller dimension for faster testing
        output_path="test_nla_dataset.parquet"
    )


if __name__ == "__main__":
    # Generate main dataset
    generate_fake_nla_dataset(
        num_samples=1000,
        activation_dim=768,
        output_path="fake_nla_dataset.parquet"
    )

    # Generate small test dataset
    generate_small_test_dataset()