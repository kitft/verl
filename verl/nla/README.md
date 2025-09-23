# Natural Language Autoencoders (NLA) Framework

A framework for training language models with activation vector injection and autoencoder-style critics using the verl reinforcement learning library.

## Overview

The NLA framework implements a novel training paradigm where:
- **Actor models** generate text conditioned on injected activation vectors
- **Critic models** reconstruct activation vectors from generated text
- Training uses both supervised fine-tuning (SFT) and reinforcement learning (RL) phases

## Data Flow Architecture

### End-to-End Data Flow

```
1. Dataset Loading
   ├─ Parquet files with (prompt, response, activation_vector)
   ├─ NLARLDataset adds injection tokens
   └─ Packages into DataProto with metadata

2. GRPO Training Step
   ├─ Prompt Expansion (1 → N trajectories)
   │  └─ Activation vectors repeated for each trajectory
   ├─ Actor Generation
   │  ├─ NLAActorWorker extracts vectors from DataProto
   │  ├─ NLAModelWrapper injects at injection token positions
   │  └─ Generates N diverse responses per prompt
   ├─ Critic Evaluation
   │  ├─ NLAAutoencoderCritic predicts activation vectors
   │  └─ MSE computation against original vectors
   ├─ Reward & Advantage Computation
   │  ├─ Rewards = -MSE (lower error → higher reward)
   │  └─ Group-wise normalization within N responses
   └─ Model Updates
      ├─ Critic: Supervised MSE loss
      └─ Actor: PPO with GRPO advantages
```

### Activation Vector Flow

```python
# 1. Dataset provides activation vectors
dataset[i] = {
    "input_ids": [tokens],
    "activation_vectors": tensor([768]),  # Original activation
    "injection_token_id": 50000
}

# 2. DataProto stores in metadata and data
data_proto = {
    "data": {"activation_vectors": tensor},
    "metadata": {"activation_vectors": tensor, "has_nla": True}
}

# 3. Actor worker extracts and passes to model
NLAActorWorker.generate_sequences(data_proto)
→ Extracts activation_vectors
→ Passes to NLAModelWrapper.forward(activation_vectors=...)

# 4. Model injects at token positions
NLAModelWrapper.forward()
→ Finds injection token positions (auto-selected rare character)
→ Replaces embeddings with activation vectors
→ Generates conditioned text

# 5. Critic reconstructs from generated text
NLAAutoencoderCritic(response_ids)
→ Encodes text to hidden states
→ Pools to single vector
→ Projects to activation space
→ Returns predicted_activation

# 6. Reward computation
MSE(predicted_activation, original_activation)
→ Convert to reward signal
→ Use for policy optimization
```

## Architecture

### Core Components

1. **NLAModelWrapper** (`models/nla_wrapper.py`)
   - Wraps any transformer model to support activation injection
   - Injects activation vectors at specified token positions
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

### Injection Token System

The NLA framework uses a sophisticated injection token system that **does not modify the tokenizer vocabulary**:

1. **Automatic Token Selection** (`utils/injection_manager.py`)
   - Automatically finds a suitable existing rare token from the vocabulary
   - Prefers rarely-used Chinese characters (e.g., "㊗", "㊙", "㊚") that tokenize to single tokens
   - No tokenizer resizing or embedding matrix modifications needed
   - Compatible with all tokenizer backends including Rust implementations

2. **Token Requirements**
   - Must already exist in the tokenizer vocabulary
   - Must tokenize to exactly one token ID
   - Must preserve structure when wrapped in `<concept>` tags
   - The injection character appears only within `<concept>CharacterHere</concept>` tags

3. **Usage Example**
   ```python
   from verl.nla.utils.injection_manager import InjectionTokenManager

   # Auto-select a suitable token
   manager = InjectionTokenManager(tokenizer)
   print(f"Using '{manager.character}' (ID: {manager.token_id}) for injection")

   # Text with injection marker
   text = f"Explain <concept>{manager.character}</concept> in detail"
   ```

4. **Benefits**
   - No model architecture changes required
   - Maintains exact compatibility with pretrained models
   - Works with any tokenizer that has Unicode support
   - Fail-fast behavior if no suitable token found
   - Clean separation between concept markers and actual text

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
    injection_token: null  # Auto-selects suitable rare token from vocab

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

#### Key Implementation Details

**Trajectory Generation (`_generate_multiple_trajectories`)**:
- Takes batch of B prompts with activation vectors
- Expands to B×N prompts by repeating each N times
- Activation vectors are similarly expanded
- Group IDs track which responses belong to same prompt

**Reward Computation (`_compute_rewards_from_critic`)**:
- Critic predicts activation for each generated response
- MSE computed against original activation vector
- Lower MSE → Higher reward (negative MSE as reward)
- No external reward model needed

**Group Advantage Normalization (`compute_advantage_and_returns`)**:
- For each group of N responses to same prompt:
  - Compute mean and std of rewards within group
  - Normalize: `advantage = (reward - mean) / std`
- Provides relative signal: which response was better than alternatives
- More stable than global normalization

See `trainer/nla_grpo_trainer.py` for full implementation.

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

## Component Integration

### Dataset Layer (`data/`)

**NLARLDataset** - RL training dataset
- Loads prompts and activation vectors from parquet
- Automatically inserts injection tokens (auto-selected rare characters)
- Handles various activation formats (lists, numpy, tensors)
- Validates activation dimensions

**NLASFTDataset** - Supervised fine-tuning dataset
- Supports actor/critic/both training modes
- Provides prompts with injection for actor
- Provides responses with targets for critic

### Worker Layer (`workers/`)

**NLAActorWorker** - Custom actor worker
- Wraps base workers with NLAModelWrapper
- Extracts activation vectors from DataProto
- Passes vectors during generation and log prob computation
- Transparent integration with verl's distributed system

### Model Layer (`models/`)

**NLAModelWrapper** - Activation injection
- Wraps any transformer model
- Finds injection token positions (wrapped in `<concept>` tags)
- Replaces/adds/projects activation vectors at embeddings
- Maintains compatibility with HuggingFace generation

**NLAAutoencoderCritic** - Vector prediction
- Encodes generated text to hidden states
- Pools sequences (last, mean, max, cls, weighted)
- Projects to activation dimension
- Outputs vectors for MSE computation

### Trainer Layer (`trainer/`)

**NLAGRPOTrainer** - GRPO with autoencoders
- Generates N trajectories per prompt
- Computes MSE-based rewards
- Group-wise advantage normalization
- Dual actor-critic optimization

**NLASFTTrainer** - Supervised pre-training
- Joint or separate actor/critic training
- Actor: SFT with activation injection
- Critic: MSE loss for reconstruction

### Integration Layer (`integration/`)

**NLADataProtoAdapter** - Data format bridge
- Converts between NLA format and verl's DataProto
- Stores vectors in metadata and data fields
- Handles injection position tracking
- Manages batch expansion for GRPO

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

## Practical Usage

### Data Preparation

Required parquet file format:
```python
import pandas as pd
import numpy as np

# Create training data
data = pd.DataFrame({
    "prompt": [
        "Explain the concept of",
        "Write a story about",
        "Describe the process of"
    ],
    "response": [
        "machine learning in simple terms",
        "a robot learning to paint",
        "photosynthesis in plants"
    ],
    "activation_vector": [
        np.random.randn(768).tolist(),  # 768-dim vector as list
        np.random.randn(768).tolist(),
        np.random.randn(768).tolist()
    ]
})

# Save as parquet
data.to_parquet("nla_train.parquet")
```

### Training Example

```python
from verl.nla.trainer import NLAGRPOTrainer
from verl.nla.data import create_nla_rl_dataset
from verl.nla.workers import create_nla_actor_worker

# 1. Create dataset with activation vectors
dataset = create_nla_rl_dataset(
    data_files=["nla_train.parquet"],
    tokenizer=tokenizer,
    config={"activation_dim": 768, "injection_token": None}  # Auto-selects
)

# 2. Wrap actor worker for injection
nla_actor = create_nla_actor_worker(
    base_worker_cls=ActorRolloutRefWorker,
    config=config
)

# 3. Train with GRPO
trainer = NLAGRPOTrainer(
    config=config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    grpo_config=GRPOTrainerConfig(
        num_trajectories_per_prompt=4,
        activation_dim=768
    )
)
trainer.fit()
```

### Generation with Activation Injection

```python
from verl.nla.models import NLAModelWrapper, InjectionConfig
from verl.nla.utils.injection_manager import InjectionTokenManager

# Set up injection token
injection_mgr = InjectionTokenManager(tokenizer)

# Wrap model for injection
wrapped_model = NLAModelWrapper(
    base_model=your_model,
    injection_config=InjectionConfig(
        mode="replace",
        injection_token_id=injection_mgr.token_id,
        injection_character=injection_mgr.character
    )
)

# Generate with activation
prompt = f"Explain <concept>{injection_mgr.character}</concept> in detail"
inputs = tokenizer(prompt, return_tensors="pt")
activation_vector = torch.randn(1, 768)  # Your activation

output = wrapped_model.generate(
    input_ids=inputs["input_ids"],
    activation_vectors=activation_vector,
    max_length=100
)
```

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

4. **GRPO Optimization**
   - Generate multiple responses per prompt
   - Group-wise advantage normalization
   - Self-contained reward signal from reconstruction

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