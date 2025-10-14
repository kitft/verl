# Qwen 30B MoE Training Guide

Complete pipeline for training NLA with **Qwen/Qwen3-30B-A3B** (~30B total parameters, ~3B active per token) on a single 8-GPU node.

## Model Information

- **Model**: Qwen/Qwen3-30B-A3B
- **Architecture**: Mixture of Experts (MoE)
- **Total Parameters**: ~30B
- **Active Parameters**: ~3B per token
- **Target Layer**: Layer 32 (adjust if needed for MoE architecture)

## Prerequisites

```bash
# Navigate to project root
cd /workspace/kitf/nla

# Activate virtual environment
source .venv/bin/activate

# Verify 8 GPUs available
nvidia-smi
```

## Complete Training Pipeline

### Step 1: Generate SFT Training Data

Generate supervised fine-tuning datasets from FineWeb using **parallel generation across 8 GPUs**:

```bash
# Generate SFT training dataset (50K samples) - FAST with 8 GPUs in parallel
./scripts/dataset_generation/run_parallel_generation.sh +runs/qwen_30b=sft_train

# Generate SFT validation dataset (500 samples) - FAST with 8 GPUs in parallel
./scripts/dataset_generation/run_parallel_generation.sh +runs/qwen_30b=sft_val
```

**Output files:**
- `data/q30b/sft/train_nla_sft_dataset_fineweb_layer32.parquet`
- `data/q30b/sft/val_nla_sft_dataset_fineweb_layer32.parquet`

**Notes:**
- Uses HuggingFaceFW/fineweb (sample-10BT) dataset
- Extracts activations from layer 32
- **Parallel execution**: Each GPU processes 1/8 of the data independently
- **Deterministic seed handling**: Per-example RNG ensures identical results vs single-run (see docs/PARALLEL_GENERATION_SEED_FIX.md)
- **Expected time**:
  - Training (50K): ~30-60 minutes (vs 4-8 hours single GPU)
  - Validation (500): ~2-5 minutes
- Batch size 32 per GPU (optimized for 60GB model on H100 80GB)

**Optional - Reuse summaries from existing model:**

If you already have summarized datasets from a smaller model (e.g., Qwen 7B), you can reuse those expensive summaries instead of regenerating them:

```bash
# After generating the base datasets above, merge in existing summaries:

# Merge training summaries (with automatic validation)
python scripts/merge_summaries.py \
    data/q30b/sft/train_nla_sft_dataset_fineweb_layer32.parquet \
    data/q7b/sft/train_nla_sft_dataset_fineweb_layer20_summarized.parquet \
    --output data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_internal.parquet

# Merge validation summaries (with automatic validation)
python scripts/merge_summaries.py \
    data/q30b/sft/val_nla_sft_dataset_fineweb_layer32.parquet \
    data/q7b/sft/val_nla_sft_dataset_fineweb_layer20_summarized.parquet \
    --output data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized_internal.parquet
```

This creates `*_summarized_internal.parquet` files with Qwen 30B activation vectors and Qwen 7B summaries.

**Automatic Validation:**

The merge script automatically validates consistency between datasets:
- ✓ **Same datapoints** (source_index values match)
- ✓ **Same source text** (identical input documents)
- ✓ **Same token positions** (activation_token_id match)
- ✓ **Same context** (immediate_prefix match)

**Handling Missing Rows:**

If the source dataset has fewer rows (e.g., due to filtered empty responses), the script will:
1. Detect the missing datapoints
2. Prompt you to either:
   - **Drop them from target** (RECOMMENDED) - ensures clean, consistent dataset
   - Keep them with empty responses (not recommended)
   - Abort and investigate

Example output:
```
The source dataset is missing 29 datapoints that exist in the target.
This typically happens when the source had empty responses that were filtered out.

Options:
  1. Drop these 29 rows from the target dataset (RECOMMENDED)
  2. Keep them with empty responses (not recommended)
  3. Abort and investigate

Drop 29 missing rows from target? (y/n/abort):
```

**Other Options:**
- `--strict` - Abort immediately on any validation error
- `--no-validate-consistency` - Skip validation entirely

**Why reuse summaries?**
- Saves hours of Claude API calls and costs
- Summaries describe text content, not model-specific activations
- Works well across different model sizes from the same family
- The activation vectors are model-specific and freshly extracted

**Alternative - Generate fresh summaries:**

If you prefer to generate new summaries for Qwen 30B:

```bash
# Summarize the newly generated datasets
python scripts/sft_training/summarize_sft_dataset.py \
    data/q30b/sft/train_nla_sft_dataset_fineweb_layer32.parquet \
    --output data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    --model claude-3-haiku-20240307 \
    --use-regular-api \
    --num-threads 50

python scripts/sft_training/summarize_sft_dataset.py \
    data/q30b/sft/val_nla_sft_dataset_fineweb_layer32.parquet \
    --output data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    --model claude-3-haiku-20240307 \
    --use-regular-api \
    --num-threads 50
```

Note: This takes several hours and incurs API costs (~$50-100 for 50K samples).

**IMPORTANT: Dataset Consistency Across Models**

The dataset generation uses **deterministic random seeds** (seed: 42 for training, 43 for validation) to ensure:
- ✅ **Same datapoints selected** for Qwen 7B and Qwen 30B (identical slice_start/slice_length)
- ✅ **Same tokens selected** within each datapoint (seeded RNG with consistent evolution)
- ✅ Only difference is the **activation vectors** (model-specific)

This guarantees that experiments are directly comparable across model sizes!

### Step 2: Generate RL Training Data

Generate reinforcement learning datasets with activation vectors using **parallel generation across 8 GPUs**:

```bash
# Generate RL training dataset (500K samples) - FAST with 8 GPUs in parallel
./scripts/dataset_generation/run_parallel_generation.sh +runs/qwen_30b=rl_train

# Generate RL validation dataset (2048 samples) - FAST with 8 GPUs in parallel
./scripts/dataset_generation/run_parallel_generation.sh +runs/qwen_30b=rl_val
```

**Output files:**
- `data/generated/rl_train_fineweb10bt_qwen_30b_layer32.parquet`
- `data/generated/rl_train_fineweb10bt_qwen_30b_layer32_random.parquet` (shuffled baseline)
- `data/generated/rl_val_fineweb10bt_qwen_30b_layer32.parquet`

**Notes:**
- Longer sequences (1024 tokens) for RL training
- Disjoint slices from SFT data (no overlap)
- Training set includes random variant for baseline
- **Parallel execution**: Each GPU processes 1/8 of the data independently
- **Deterministic seed handling**: Per-example RNG ensures identical results vs single-run (see docs/PARALLEL_GENERATION_SEED_FIX.md)
- **Expected time**:
  - Training (500K): ~2-4 hours (vs 20-40 hours single GPU)
  - Validation (2K): ~5-10 minutes
- Batch size 16 per GPU (optimized for 1024 token context)

### Step 3: Generate NLA Evaluation Data

Generate evaluation datasets for monitoring training progress:

```bash
# Generate evaluation activations (QA, 2hop, etc.)
python scripts/nla_eval/generate_eval_activations.py \
    model.path=Qwen/Qwen3-30B-A3B \
    extraction.layer_index=32 \
    output.path=data/eval/nla_eval_activations_qwen_30b_layer32.parquet

# Or generate specific evaluation datasets
python scripts/nla_eval/generate_eval_activations.py \
    model.path=Qwen/Qwen3-30B-A3B \
    extraction.layer_index=32 \
    output.base_path=data/eval \
    output.model_suffix=qwen_30b
```

**Output files:**
- `data/eval/nla_eval_activations_QA_qwen_30b_layer32.parquet`
- `data/eval/nla_eval_activations_2hop_qwen_30b_layer32.parquet`
- Additional evaluation datasets as configured

**CRITICAL:** Evaluation activations MUST be generated with the same model and layer as training data!

### Step 4: Generate Random Baseline Datasets (Optional)

Create randomized versions for baseline comparison:

```bash
# If using merged summaries (_internal suffix):
python scripts/permute_activation_vectors.py \
    data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_internal.parquet

python scripts/permute_activation_vectors.py \
    data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized_internal.parquet

# OR if using freshly generated summaries (no _internal suffix):
python scripts/permute_activation_vectors.py \
    data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet

python scripts/permute_activation_vectors.py \
    data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet
```

**Output files (depending on which you generated):**
- `data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_internal_random.parquet`
- `data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized_internal_random.parquet`

OR

- `data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_random.parquet`
- `data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized_random.parquet`

This automatically appends `_random` suffix and uses a fixed seed (42) for reproducibility.

### Step 5: SFT Training - Actor (Decoder)

Train the actor model with activation injection on 8 GPUs.

#### Option A: FSDP Backend (Recommended for ease of use)

```bash
# Full-scale actor training using the Qwen 30B config with FSDP
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=actor
```

**Using freshly generated summaries (no _internal suffix):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=actor \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    data.val_files=data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet
```

**Configuration highlights (nla_sft_30b_8gpu - FSDP):**
- **Model**: Qwen/Qwen3-30B-A3B (30B MoE, 3B active)
- **Backend**: FSDP2 (PyTorch FSDP)
- **Batch size**: 256 total (16 per GPU × 8 GPUs × 2 gradient accumulation)
- **Sequence length**: 512 tokens (same as 7B)
- **Learning rate**: 3e-5 (reduced from 5e-5 for stability)
- **Gradient checkpointing**: Enabled for memory efficiency
- **FSDP**: Auto-enabled for 8-GPU training
- **Inherits from**: `nla_sft_7b_8gpu.yaml`

**Expected checkpoint location:**
- `checkpoints/nla_sft_qwen_30b_8gpu/global_step_XXXX/huggingface/`

#### Option B: Megatron Backend (Recommended for large-scale training)

For better performance with MoE models, use the Megatron backend with Expert Parallelism:

```bash
# Actor training with Megatron backend (must use torchrun)
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=actor
```

**Using freshly generated summaries (no _internal suffix):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=actor \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    data.val_files=data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet
```

**Configuration highlights (nla_sft_30b_8gpu_megatron):**
- **Model**: Qwen/Qwen3-30B-A3B (30B MoE, 3B active)
- **Backend**: Megatron-LM with NLA injection hooks
- **Parallelism**: TP=2 (Tensor Parallel) × EP=4 (Expert Parallel) = 8 GPUs
  - **TP=2**: Model layers sharded across 2 GPUs
  - **EP=4**: MoE experts distributed across 4 groups
  - **Sequence Parallel**: Enabled (requires TP > 1)
- **Batch size**: 256 total (8 micro-batch per GPU with dynamic batching)
- **Learning rate**: 3e-5 with cosine schedule
- **Memory optimizations**: Selective recomputation, distributed optimizer
- **Activation injection**: Embedding-layer hooks with micro-batch tracking

**Expected checkpoint location:**
- `checkpoints/nla_sft_qwen_30b_8gpu_megatron/global_step_XXXX/huggingface/`

**Megatron advantages for MoE:**
- Expert Parallelism efficiently distributes MoE experts
- Better memory efficiency with distributed optimizer
- Faster training with optimized kernels
- Full support for PP/TP/EP combinations

**Optional - Random baseline (FSDP):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=actor \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_internal_random.parquet \
    trainer.default_local_dir=checkpoints/nla_actor_q30b_random \
    trainer.experiment_name=qwen30b-actor-random
```

**Optional - Random baseline (Megatron):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=actor \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized_internal_random.parquet \
    trainer.default_local_dir=checkpoints/nla_actor_q30b_megatron_random \
    trainer.experiment_name=qwen30b-actor-megatron-random
```

### Step 6: SFT Training - Critic (Encoder)

Train the critic model to extract activation vectors from text.

#### Option A: FSDP Backend

```bash
# Full-scale critic training using the Qwen 30B config with FSDP
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=critic
```

**Using freshly generated summaries (no _internal suffix):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=critic \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    data.val_files=data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet
```

**Optional - Truncate critic for faster training:**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_30b_8gpu \
    nla.train_mode=critic \
    nla.critic.truncate_layers=15
```

#### Option B: Megatron Backend

```bash
# Critic training with Megatron backend (must use torchrun)
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=critic
```

**Using freshly generated summaries (no _internal suffix):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=critic \
    data.train_files=data/q30b/sft/train_nla_sft_dataset_fineweb_layer32_summarized.parquet \
    data.val_files=data/q30b/sft/val_nla_sft_dataset_fineweb_layer32_summarized.parquet
```

**Optional - Truncate critic for faster training (Megatron):**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    verl/verl/trainer/nla_megatron_sft_trainer_entry.py \
    --config-name=nla_sft_30b_8gpu_megatron \
    nla.train_mode=critic \
    nla.critic.truncate_layers=15
```

**Expected checkpoint locations:**
- **FSDP**: `checkpoints/nla_sft_qwen_30b_8gpu/global_step_XXXX/huggingface/`
- **Megatron**: `checkpoints/nla_sft_qwen_30b_8gpu_megatron/global_step_XXXX/huggingface/`

**Notes:**
- Critic outputs activation vectors (not scalars)
- Processes response text only (not user_prompt + response)
- Uses same training data as actor
- `truncate_layers` can reduce critic to first K layers for speed (set to 10-15 for faster training)
- Config inherits appropriate batch sizes for 30B MoE
- **Megatron backend**: NLA implementation is actor-only currently (critic support coming soon)

### Step 7: RL Training (GRPO)

Train with reinforcement learning using both SFT-trained models:

```bash
# Replace XXXX with your actual checkpoint step numbers
export ACTOR_CHECKPOINT="checkpoints/nla_actor_q30b/global_step_XXXX/huggingface"
export CRITIC_CHECKPOINT="checkpoints/nla_critic_q30b/global_step_XXXX/huggingface"

# Run RL training with GRPO
python -m verl.trainer.nla_main \
    --config-name runs/qwen_30b/rl_grpo_summ \
    actor_rollout_ref.model.path=${ACTOR_CHECKPOINT} \
    critic.model.path=${CRITIC_CHECKPOINT}
```

**Using base models (without SFT):**
```bash
# Train directly from base Qwen 30B MoE
python -m verl.trainer.nla_main \
    --config-name runs/qwen_30b/rl_grpo_summ
```

**Configuration highlights:**
- **GPUs**: 8 GPUs, sequential usage (actor rollout → critic)
- **Rollout**: TP=1, 8 responses per prompt (GRPO)
- **Batch sizes**: Micro batch 4/GPU, mini batch 512
- **Memory**: GPU utilization 0.45, FSDP offloading enabled
- **Training**: 10,000 steps, eval every 100 steps
- **WandB**: Logged to `nla-rl-q30b/qwen-30b-moe-multigpu`

**With custom critic prompt:**
```bash
python -m verl.trainer.nla_main \
    --config-name runs/qwen_30b/rl_grpo_summ \
    actor_rollout_ref.model.path=${ACTOR_CHECKPOINT} \
    critic.model.path=${CRITIC_CHECKPOINT} \
    critic.critic_prompt="summary of the following text: "
```

**Training with random baseline:**
```bash
python -m verl.trainer.nla_main \
    --config-name runs/qwen_30b/rl_grpo_summ \
    data.train_files=/workspace/kitf/nla/data/generated/rl_train_fineweb10bt_qwen_30b_layer32_random.parquet
```

### Step 8: Test Autoencoder (Optional)

Verify activation injection is working:

```bash
# Test with SFT checkpoints
python scripts/nla_inference/run_nla_ray_autoencoder.py \
    --actor-model ${ACTOR_CHECKPOINT} \
    --critic-model ${CRITIC_CHECKPOINT} \
    --source-prompts "Quantum computing uses qubits to perform calculations." \
    --token-indices 3 \
    --target-prompts "Explain the following text <concept><INJECT></concept>" \
    --layer-index 32 \
    --max-new-tokens 64 \
    --num-completions 3 \
    --num-gpus 1 \
    --num-cpus 4

# Test with amplified activations (should see dramatic changes)
python scripts/nla_inference/run_nla_ray_autoencoder.py \
    --actor-model ${ACTOR_CHECKPOINT} \
    --critic-model ${CRITIC_CHECKPOINT} \
    --source-prompts "Quantum computing uses qubits." \
    --token-indices 3 \
    --target-prompts "Explain the following text <concept><INJECT></concept>" \
    --num-completions 3 \
    --test-scale 50.0 \
    --num-gpus 1
```

## Configuration Files

All configs located in `verl/verl/trainer/config/runs/qwen_30b/`:

- **rl_grpo_summ.yaml** - Main RL training config (8 GPU, GRPO)
- **sft_train.yaml** - SFT training data generation
- **sft_val.yaml** - SFT validation data generation
- **rl_train.yaml** - RL training data generation
- **rl_val.yaml** - RL validation data generation
- **rl_grpo.yaml** - Alternative RL config (if needed)
- **rl_grpo_split.yaml** - Split resource pool config (advanced)

## Key Parameters

### Dataset Generation
```yaml
model:
  name: Qwen/Qwen3-30B-A3B
  layer_index: 32  # Adjust if needed for MoE
  model_dtype: bf16

sampling:
  parallel_batch_size: null  # Auto-detect based on 8 GPUs
  max_length: 512  # SFT: 512, RL: 1024
```

### RL Training (rl_grpo_summ.yaml)
```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-30B-A3B
  actor:
    ppo_micro_batch_size_per_gpu: 4  # Reduced for MoE
  rollout:
    gpu_memory_utilization: 0.45  # Slightly reduced for MoE
    tensor_model_parallel_size: 1
    n: 8  # GRPO: 8 responses per prompt

critic:
  output_layer_index: 32  # Match extraction layer
  ppo_micro_batch_size_per_gpu: 4
  model:
    path: Qwen/Qwen3-30B-A3B

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  total_training_steps: 10000
  test_freq: 100
  project_name: nla-rl-q30b
  experiment_name: qwen-30b-moe-multigpu

nla_eval:
  enabled: true
  eval_files:
    QA: data/eval/nla_eval_activations_QA_qwen_30b_layer32.parquet
    2hop: data/eval/nla_eval_activations_2hop_qwen_30b_layer32.parquet
  freq: 100
```

## Resource Requirements

### Dataset Generation
- **GPUs**: 8 (parallel execution with `run_parallel_generation.sh`)
- **Memory**: ~60GB GPU memory per GPU (model weights) + ~10-20GB activations
- **Batch size**: 32 per GPU (SFT, 512 tokens), 16 per GPU (RL, 1024 tokens)
- **Time** (with parallel script):
  - SFT train (50K): ~30-60 minutes (8 GPUs in parallel)
  - SFT val (500): ~2-5 minutes (8 GPUs in parallel)
  - RL train (500K): ~2-4 hours (8 GPUs in parallel)
  - RL val (2K): ~5-10 minutes (8 GPUs in parallel)

### SFT Training
- **GPUs**: 8 (single node)
- **Memory**: ~40-60GB per GPU with gradient checkpointing and FSDP offloading
- **Time**: ~5-10 hours per epoch (50K samples)
- **Checkpoints**: ~60GB per checkpoint (sharded FSDP + HuggingFace)

### RL Training
- **GPUs**: 8 (sequential usage: rollout → critic)
- **Memory**: ~40-60GB per GPU with offloading
- **Time**: ~50-100 hours for 10K steps
- **Ray resources**: 32 CPUs, 8 GPUs

## Memory Optimization Tips

For OOM issues:

```bash
# Reduce micro batch size
data.micro_batch_size_per_gpu=1

# Enable all memory optimizations
model.enable_gradient_checkpointing=true
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
critic.model.fsdp_config.param_offload=true
critic.model.fsdp_config.optimizer_offload=true

# Reduce GPU memory utilization for rollout
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# Truncate critic (faster, less memory)
nla.critic.truncate_layers=15

# Reduce context length
data.max_response_length=64
actor_rollout_ref.rollout.response_length=64
```

## Monitoring

### WandB Projects
- **SFT Training**: `nla-sft-q30b`
- **RL Training**: `nla-rl-q30b`

### Key Metrics
- **SFT**: `train/loss`, `train/actor_loss`, `train/critic_loss`
- **RL**: `reward/mean`, `nla_eval/accuracy`, `nla_eval/QA/accuracy`
- **Eval**: `nla_eval/string_match_accuracy`, per-category accuracies

### Checkpoints
- **Location**: `checkpoints/nla_{actor,critic}_q30b/`
- **Format**: FSDP sharded + HuggingFace format in `huggingface/` subdirectory
- **Frequency**: End of training by default (`save_freq: -1`)

## Troubleshooting

### OOM During Dataset Generation
The parallel script already uses optimized batch sizes (32 for SFT, 16 for RL). If you still encounter OOM:

```bash
# Manually edit the config files to reduce batch size:
# verl/verl/trainer/config/runs/qwen_30b/sft_train.yaml
# sampling.parallel_batch_size: 32 → 16

# verl/verl/trainer/config/runs/qwen_30b/rl_train.yaml
# sampling.parallel_batch_size: 16 → 8

# Then re-run the parallel script
./scripts/dataset_generation/run_parallel_generation.sh +runs/qwen_30b=sft_train
```

### OOM During SFT Training
- Reduce `data.micro_batch_size_per_gpu` to 1
- Enable gradient checkpointing
- Use mixed precision (bf16 already default)
- Consider truncating critic layers

### OOM During RL Training
- Reduce `ppo_micro_batch_size_per_gpu` to 2
- Lower `gpu_memory_utilization` to 0.35-0.4
- Enable all FSDP offloading options
- Reduce response length

### Eval Activations Not Working
- **CRITICAL**: Regenerate eval data with exact same model and layer:
  ```bash
  python scripts/nla_eval/generate_eval_activations.py \
      model.path=Qwen/Qwen3-30B-A3B \
      extraction.layer_index=32
  ```
- Check `nla_eval.enabled: true` in config
- Verify layer index matches: `critic.output_layer_index` == `extraction.layer_index`

### Slow Training
- Verify 8 GPUs detected: check logs for `n_gpus_per_node: 8`
- Ensure FSDP enabled (automatic for multi-GPU)
- Check Ray resource allocation in logs
- Consider enabling fused kernels (experimental):
  ```bash
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
  ```

## Comparison with Qwen 7B

| Aspect | Qwen 7B | Qwen 30B MoE |
|--------|---------|--------------|
| Total params | 7B | ~30B |
| Active params | 7B | ~3B |
| Micro batch/GPU | 8 | 4 |
| GPU memory util | 0.5 | 0.45 |
| SFT time (est.) | 2-4 hours | 5-10 hours |
| RL time (est.) | 20-40 hours | 50-100 hours |
| Checkpoint size | ~28GB | ~60GB |

## Advanced Usage

### Custom Layer Index
```bash
# Use different layer for activations
python scripts/sft_training/generate_sft_nla_dataset.py \
    --config-name runs/qwen_30b/sft_train \
    model.layer_index=25
```

### Custom Dataset Size
```bash
# Generate smaller dataset for testing
python scripts/sft_training/generate_sft_nla_dataset.py \
    --config-name runs/qwen_30b/sft_train \
    dataset.slice_length=5000 \
    sampling.total_activations=5000
```

### Resuming Training
```bash
# Resume SFT from checkpoint
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/sft_training/run_sft_nla_training.py \
    --config-name=nla_sft_full \
    trainer.resume_mode=auto \
    trainer.resume_from_path=checkpoints/nla_actor_q30b/global_step_500

# Resume RL training (automatic via Ray)
python -m verl.trainer.nla_main \
    --config-name runs/qwen_30b/rl_grpo_summ \
    trainer.resume_mode=auto
```

## Quick Start Summary

**Recommended workflow for reusing existing summaries:**

1. **Generate base datasets** with Qwen 30B activations (Step 1)
2. **Merge existing summaries** from Qwen 7B (Step 1 - optional section)
3. **Create random baselines** for comparison (Step 4)
4. **Generate RL and eval data** (Steps 2-3)
5. **Train actor and critic** with SFT (Steps 5-6)
6. **Run RL training** with trained models (Step 7)
7. **Monitor and test** convergence (Step 8)

**Time savings with summary reuse:**
- Skip ~4-6 hours of Claude API processing
- Save ~$50-100 in API costs
- Get identical training quality (summaries are model-agnostic)

## Next Steps

1. Complete all dataset generation (Steps 1-3)
2. Optionally merge existing summaries to save time/cost (Step 1)
3. Generate random baselines (Step 4)
4. Train actor and critic with SFT (Steps 5-6)
5. Run RL training with trained models (Step 7)
6. Monitor WandB for convergence and eval metrics
7. Test autoencoder reconstruction quality (Step 8)
8. Scale to production with optimized hyperparameters

## Files Overview

```
qwen_30b/
├── QWEN30B_README.md           # This file
├── rl_grpo_summ.yaml           # Main RL config (RECOMMENDED)
├── sft_train.yaml              # SFT training data generation
├── sft_val.yaml                # SFT validation data generation
├── rl_train.yaml               # RL training data generation
├── rl_val.yaml                 # RL validation data generation
└── rl_grpo.yaml                # Alternative RL config

Generated data:
data/q30b/sft/                  # SFT datasets
data/generated/                 # RL datasets (qwen_30b prefix)
data/eval/                      # Evaluation datasets (qwen_30b suffix)

Checkpoints:
checkpoints/nla_actor_q30b/     # Actor SFT checkpoints
checkpoints/nla_critic_q30b/    # Critic SFT checkpoints
```

## Support

For issues or questions:
- Check the main README: `/workspace/kitf/nla/README_NLA_TRAINING.md`
- Review architecture docs: `/workspace/kitf/nla/docs/`
- Verify environment setup in `/workspace/kitf/nla/CLAUDE.md`
