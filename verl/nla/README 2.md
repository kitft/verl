# Natural Language Autoencoder (NLA) with GRPO Training

This module implements Natural Language Autoencoder (NLA) training using Group Relative Policy Optimization (GRPO) within the verl framework.

## Overview

NLA is an autoencoder-style training approach where:
1. **Actor**: Generates text conditioned on activation vectors (encoder)
2. **Critic**: Reconstructs activation vectors from generated text (decoder)
3. **GRPO**: Optimizes the actor to preserve activation information using group-relative advantages

This enables controllable text generation where activations serve as steering vectors.

## Architecture

```
verl/nla/
├── models/              # Core model components
│   ├── nla_model_wrapper.py      # Activation injection wrapper
│   └── autoencoder_critic.py     # Critic that predicts activations
├── trainer/             # Training algorithms
│   ├── nla_grpo_trainer.py       # GRPO implementation for NLA
│   └── nla_ppo_trainer.py        # Standard PPO baseline
├── rewards/             # Reward computation
│   └── mse_reward.py             # MSE-based rewards
├── integration/         # verl integration
│   ├── worker_wrapper.py         # Worker integration
│   └── dataproto_adapter.py      # Data format adaptation
├── data/               # Data utilities
│   └── nla_dataset.py           # Dataset with activation vectors
├── configs/            # Configuration files
│   └── nla_grpo_config.yaml     # GRPO training config
├── examples/           # Example scripts
│   └── train_nla_grpo.py        # Training launcher
└── tests/              # Comprehensive test suite
    ├── conftest.py               # Shared fixtures
    ├── test_nla_models.py        # Model unit tests
    └── test_grpo_trainer.py      # Trainer integration tests
```

## Key Components

### 1. NLA Model Wrapper
Wraps transformer models to inject activation vectors at specified positions.

**Features:**
- **Injection Modes**: Replace, add, or project activations
- **Multi-layer Support**: Inject at multiple transformer layers
- **Token-based Injection**: Uses special tokens to mark injection points
- **Generation Compatible**: Works with both forward and generate methods

### 2. Autoencoder Critic
A critic that predicts activation vectors from generated text (instead of scalar values).

**Features:**
- **Pooling Strategies**: Last token, mean, or max pooling
- **Projection Head**: MLP to map hidden states to activation dimension
- **Supervised Training**: Trained with MSE loss to reconstruct activations

### 3. GRPO Trainer
Implements Group Relative Policy Optimization for NLA training.

**Features:**
- **N Trajectories per Prompt**: Generates multiple responses for comparison
- **Group-Relative Advantages**: Normalizes advantages within each group
- **Dual Training**: Actor (GRPO) and Critic (supervised) trained simultaneously
- **Configurable Normalization**: Supports both GRPO and Dr.GRPO modes

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/verl-nla.git
cd verl-nla

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Basic GRPO Training

```python
from verl.nla.trainer.nla_grpo_trainer import NLAGRPOTrainer, GRPOTrainerConfig
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/nla_grpo_config.yaml")

# Configure GRPO
grpo_config = GRPOTrainerConfig(
    num_trajectories_per_prompt=4,  # Generate 4 responses per prompt
    norm_adv_by_std_in_grpo=True,   # Use GRPO normalization
    activation_dim=768,              # Dimension of activation vectors
)

# Create trainer
trainer = NLAGRPOTrainer(
    config=config,
    grpo_config=grpo_config,
)

# Run training
trainer.fit()
```

### 2. Model with Activation Injection

```python
from verl.nla.models import NLAModelWrapper, InjectionConfig
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure injection
injection_config = InjectionConfig(
    mode="replace",              # Replace embeddings with activations
    layer_indices=[0],           # Inject at embedding layer
    injection_token_id=50256,    # Token ID for <INJECT>
)

# Wrap model
nla_model = NLAModelWrapper(
    base_model=model,
    injection_config=injection_config,
    hidden_dim=768,
    activation_dim=256,
)

# Generate with activation injection
outputs = nla_model.generate(
    input_ids=input_ids,
    activation_vectors=activation_vectors,
    max_new_tokens=100,
)
```

### 3. Data Preparation

```python
from verl.nla.data import NLADataset

# Prepare prompts with injection markers
prompts = [
    "The concept <INJECT> means",
    "Based on <INJECT>, we can conclude",
]

# Your activation vectors (e.g., from concept extraction)
activation_vectors = torch.randn(len(prompts), 768)

# Create dataset
dataset = NLADataset(
    prompts=prompts,
    activation_vectors=activation_vectors,
    tokenizer=tokenizer,
    injection_token="<INJECT>",
)
```

## GRPO Algorithm Details

### Advantage Calculation

GRPO computes advantages relative to other responses from the same prompt:

```python
# For each group of N responses from the same prompt:
group_mean = group_rewards.mean()
group_std = group_rewards.std()

# GRPO mode (norm_adv_by_std_in_grpo=True)
advantages = (rewards - group_mean) / group_std

# Dr.GRPO mode (norm_adv_by_std_in_grpo=False)
advantages = rewards - group_mean
```

### Training Flow

1. **Generate**: Create N trajectories per prompt with activation injection
2. **Predict**: Critic predicts activation vectors from generated text
3. **Reward**: Compute MSE between predicted and original activations
4. **Advantage**: Calculate group-relative advantages
5. **Update**: Train actor with GRPO, critic with supervised MSE

## Configuration

### Key Parameters

```yaml
# GRPO-specific
grpo:
  num_trajectories_per_prompt: 4    # N responses per prompt
  norm_adv_by_std_in_grpo: true     # true=GRPO, false=Dr.GRPO
  critic_learning_rate: 5e-5        # Critic LR (supervised)
  critic_train_epochs: 1            # Epochs per update

# Algorithm
algorithm:
  adv_estimator: "grpo"              # Use GRPO advantages
  gamma: 1.0                         # Discount factor
  use_kl_in_reward: false           # Optional KL penalty

# NLA-specific
nla:
  activation_dim: 768                # Size of activation vectors
  injection_token: "<INJECT>"        # Marker for injection
  mode: "replace"                    # Injection mode
  layer_indices: [0]                # Layers to inject at
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest verl/nla/tests/

# Run specific test categories
pytest verl/nla/tests/ -m unit          # Unit tests only
pytest verl/nla/tests/ -m integration   # Integration tests
pytest verl/nla/tests/ -m slow          # Slow tests

# Run with coverage
pytest verl/nla/tests/ --cov=verl.nla --cov-report=html
```

## Examples

### Training Script

```bash
# Train with default config
python verl/nla/examples/train_nla_grpo.py

# Train with custom config
python verl/nla/examples/train_nla_grpo.py configs/my_config.yaml

# Override parameters
python verl/nla/examples/train_nla_grpo.py \
    grpo.num_trajectories_per_prompt=8 \
    trainer.batch_size=64
```

### Distributed Training

```python
# Initialize Ray cluster
ray.init(num_gpus=8)

# Configure resource pools
resource_manager = ResourcePoolManager(
    resource_pool_spec={
        "actor_rollout_ref": [8] * num_nodes,
        "critic": [8] * num_nodes,
    }
)

# Train with distributed workers
trainer = NLAGRPOTrainer(config, grpo_config)
trainer.fit()
```

## Performance Considerations

### Memory Optimization
- **Batch Size**: Each prompt generates N responses, so effective batch = `batch_size * N`
- **Gradient Checkpointing**: Enable for large models
- **Mixed Precision**: Use fp16/bf16 for faster training

### Speed Optimization
- **Parallel Generation**: N trajectories generated in parallel
- **Distributed Training**: Use Ray for multi-GPU/multi-node
- **Critic Warmup**: Can skip early actor updates if needed

## Comparison with Standard PPO

| Feature | Standard PPO | NLA with GRPO |
|---------|-------------|---------------|
| Critic Output | Scalar value V(s) | Activation vector (D-dim) |
| Reward Source | External reward model | MSE reconstruction error |
| Advantage | GAE over trajectory | Group-relative (N responses) |
| Training | Actor + Critic (both RL) | Actor (RL) + Critic (supervised) |
| Use Case | General RLHF | Activation-conditioned generation |

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
   ```
   RuntimeError: size mismatch
   ```
   - Check `activation_dim` matches your activation vectors
   - Verify `hidden_dim` matches model dimension

2. **No Injection Happening**
   ```
   Warning: No injection token found
   ```
   - Ensure `<INJECT>` token is in prompts
   - Check tokenizer has the special token added

3. **GRPO Advantages All Zero**
   ```
   advantages.std() = 0
   ```
   - Increase `num_trajectories_per_prompt`
   - Check rewards have variance

4. **Out of Memory**
   ```
   CUDA out of memory
   ```
   - Reduce `batch_size` or `num_trajectories_per_prompt`
   - Enable gradient checkpointing
   - Use mixed precision training

## Citation

If you use this implementation, please cite:

```bibtex
@software{nla_grpo_2024,
  title={Natural Language Autoencoder with GRPO},
  author={Your Team},
  year={2024},
  url={https://github.com/your-org/verl-nla}
}
```

## References

- **GRPO**: [Group Relative Policy Optimization](https://arxiv.org/abs/XXX)
- **Dr.GRPO**: [Direct Reward GRPO](https://arxiv.org/abs/2503.20783)
- **Activation Steering**: [Representation Engineering](https://arxiv.org/abs/XXX)
- **verl**: [Versatile Reinforcement Learning](https://github.com/volcengine/verl)

## License

This project is licensed under the Apache 2.0 License - see LICENSE file for details.