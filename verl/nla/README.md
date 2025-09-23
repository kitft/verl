# Natural Language Autoencoders (NLA) Framework

A framework for training language models with activation vector injection and autoencoder-style critics using the verl reinforcement learning library.

## Overview

The NLA framework implements a novel training paradigm where:
- **Actor models** generate text conditioned on injected activation vectors
- **Critic models** reconstruct activation vectors from generated text
- Training uses both supervised fine-tuning (SFT) and reinforcement learning (RL) phases

## Architecture

### Core Components

1. **NLAModelWrapper** (`models/nla_wrapper.py`)
   - Wraps any transformer model to support activation injection
   - Injects activation vectors at specified token positions marked with `<INJECT>`
   - Supports multiple injection modes: replace, add, or project
   - Seamlessly integrates with HuggingFace generation

2. **NLAAutoencoderCritic** (`models/autoencoder_critic.py`)
   - Predicts activation vectors from generated text
   - Uses pooling strategies (last, mean, max, cls, weighted) to encode sequences
   - Outputs vectors instead of scalar values for MSE-based rewards

3. **Training Paradigms**
   - **SFT Phase**: Supervised pre-training for both actor and critic
   - **GRPO Training**: Group Relative Policy Optimization with multiple trajectories
   - **PPO Training**: Standard PPO with activation-based rewards

## Supervised Fine-Tuning (SFT) Phase

The SFT phase provides initial supervised training before RL:

### Components

1. **NLASFTDataset** (`data/nla_sft_dataset.py`)
   - Handles datasets with activation vectors
   - Supports three modes:
     - `actor`: Prompts with activation injection
     - `critic`: Responses with target activations for reconstruction
     - `both`: Combined training data

2. **NLASFTTrainer** (`trainer/nla_sft_trainer.py`)
   - Extends FSDP trainer for distributed training
   - Supports joint or separate actor/critic training
   - Actor training: Standard SFT with activation injection
   - Critic training: MSE loss for activation prediction

### Data Format

Required parquet columns:
```python
{
    "prompt": "User prompt text",
    "response": "Expected response text",
    "activation_vector": [768-dimensional vector as list]
}
```

### Configuration

See `configs/nla_sft_config.yaml` for full configuration options:

```yaml
model:
  model_name: "meta-llama/Llama-2-7b-hf"
  activation_dim: 768
  injection:
    mode: "replace"  # or "add", "project"
    layer_indices: [0]  # embedding layer
    injection_token: "<INJECT>"

trainer:
  train_mode: "both"  # or "actor", "critic"
  total_training_steps: 10000
```

### Running SFT Training

```bash
python verl/nla/scripts/run_nla_sft.py \
    --config verl/nla/configs/nla_sft_config.yaml \
    --train-mode both
```

## Reinforcement Learning Phase

### GRPO Training

Group Relative Policy Optimization generates N responses per prompt for relative comparison:

1. **Generate N trajectories** per prompt with activation injection
2. **Compute rewards** via critic's MSE between predicted and target activations
3. **Calculate advantages** within each group of N trajectories
4. **Update policies** using group-normalized advantages

See `trainer/nla_grpo_trainer.py` for implementation.

### PPO Training

Standard PPO training with activation-based rewards:
- Actor generates with activation injection
- Critic evaluates response quality via activation reconstruction
- Rewards based on reconstruction accuracy

See `trainer/nla_ppo_trainer.py` for implementation.

## Reward System

### MSE-Based Rewards
- **Fidelity**: How well the critic reconstructs the original activation
- **Reward**: -MSE(predicted_activation, target_activation)
- Optionally normalized and scaled

### Reward Components (`rewards/mse_reward.py`)
- `MSERewardComputer`: Calculates rewards from activation MSE
- `CriticSupervisedLoss`: MSE loss for critic training

## Integration with verl

### DataProto Adapter (`integration/dataproto_adapter.py`)
- Converts between NLA data format and verl's DataProto
- Handles activation vector extraction and injection
- Manages tokenization and masking

## Testing

Comprehensive test suite in `tests/`:

```bash
# Run all NLA tests (requires flash_attn)
uv run pytest verl/nla/tests/

# Run specific test file
uv run python verl/nla/tests/test_nla_sft.py
```

Test coverage includes:
- Dataset loading and processing
- Model wrapper injection mechanics
- Critic activation prediction
- Trainer initialization and training steps
- Integration with verl components

## Key Features

1. **Flexible Injection**
   - Multiple injection modes (replace, add, project)
   - Multi-layer injection support
   - Configurable injection tokens

2. **Distributed Training**
   - Full FSDP support for large models
   - Gradient accumulation
   - Mixed precision training

3. **Modular Design**
   - Separate actor and critic training
   - Pluggable reward functions
   - Easy integration with existing models

## Dependencies

- verl (reinforcement learning framework)
- transformers (model loading)
- torch (deep learning)
- flash-attn (optional, for optimized attention)

## Future Work

- [ ] Support for variable-length activation sequences
- [ ] Multi-modal activation vectors
- [ ] Hierarchical activation structures
- [ ] Online critic updates during generation
- [ ] Curriculum learning for injection difficulty