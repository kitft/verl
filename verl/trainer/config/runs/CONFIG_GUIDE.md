# NLA GRPO Configuration Guide

This directory contains optimized multi-GPU configurations for various Qwen model sizes.

## Two Configuration Variants

Each model size has **two versions**:

### 1. `rl_grpo.yaml` - Standard (Recommended for Most Users)

**Architecture:**
- All workers use all available GPUs sequentially
- No resource pools - follows verl's default behavior
- Simpler configuration with less overhead

**GPU Usage Pattern:**
```
Step 1: Generation (SGLang)     → All GPUs available
Step 2: Actor Training (FSDP)   → All GPUs (sequential)
Step 3: Critic Training (FSDP)  → All GPUs (sequential)
```

**Best for:**
- Small to medium models (0.5B - 3B)
- Single-node training
- Users who want simple, reliable configs
- When communication overhead > parallelism benefit

**Example:**
```bash
python verl/verl/trainer/nla_main.py --config-name=runs/qwen_1p5b/rl_grpo
```

---

### 2. `rl_grpo_split.yaml` - Split Placement (Advanced)

**Architecture:**
- Actor/Rollout and Critic use separate GPU pools
- Can run actor/critic training in parallel (with code modifications)
- Higher tensor parallelism for inference

**GPU Usage Pattern:**
```
Step 1: Generation (SGLang TP=2/4) → Pool 1 (GPUs 0-3)
Step 2: Actor Training              → Pool 1 (GPUs 0-3)
Step 3: Critic Training             → Pool 2 (GPUs 4-7) [Can run parallel with actor*]
```
*Requires `blocking=False` in worker code (see note below)

**Best for:**
- Large models (7B+)
- High memory pressure during inference
- Advanced users comfortable with verl internals
- When model size justifies communication overhead

**Example:**
```bash
python verl/verl/trainer/nla_main.py --config-name=runs/qwen_7b/rl_grpo_split
```

---

## Configuration Matrix

| Model | Standard Config | Split Config | TP (Standard) | TP (Split) | Recommendation |
|-------|----------------|--------------|---------------|------------|----------------|
| **0.5B (4 GPU)** | ✅ Default | 🔧 Advanced | 1 | 2 | Use standard |
| **1.5B** | ✅ Default | 🔧 Advanced | 1 | 2 | Use standard |
| **3B** | ✅ Default | 🔧 Advanced | 1 | 2 | Either works |
| **7B** | ⚡ Good | ✅ Better | 2 | 2 | **Use split** |
| **14B** | ⚠️ Slow | ✅ Recommended | 4 | 4 | **Use split** |
| **32B** | ❌ Too slow | ✅ Required | 8 | 4 | **Use split** |

---

## Key Differences

### Tensor Parallelism (TP)

**Critical Rule:** `tensor_model_parallel_size` must match GPU availability during inference!

**Standard configs:**
- SGLang can use all GPUs (no pool restrictions)
- TP=1 for small models (sufficient memory)
- TP increases with model size (memory pressure)

**Split configs:**
- SGLang limited to actor_rollout pool size
- TP must equal pool size to avoid idle GPUs
- Example: 4 GPU pool → TP=4 (or TP=2 for efficiency)

### Memory Utilization

**Standard configs:**
- `gpu_memory_utilization: 0.5-0.8` (higher is fine)
- No memory contention between pools

**Split configs:**
- `gpu_memory_utilization: 0.5` (more conservative)
- TP overhead requires more memory coordination

---

## Enabling Parallel Actor/Critic Training

Split placement configs **support** parallel training, but require code modifications:

**Modify:** `verl/workers/fsdp_workers.py`

```python
# Change from:
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    ...

# To:
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_actor(self, data: DataProto):
    ...

# Do the same for update_critic
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_critic(self, data: DataProto):
    ...
```

**Without this modification**, split configs still work but actor/critic run sequentially (still beneficial for large models due to better memory management).

---

## GRPO Batch Size Formula

**All configs follow this rule:**
```
ppo_mini_batch_size = data.batch_size × rollout.n
```

**Why?** With `rollout.n=4`, each prompt generates 4 responses:
- 32 prompts × 4 = 128 total experiences
- For ONE gradient update: `ppo_mini_batch_size = 128`

**If you change `rollout.n`, you MUST update `ppo_mini_batch_size`!**

---

## Common Issues

### Issue: Low throughput with standard config
**Solution:** Try split config with higher TP

### Issue: "Out of memory" during generation
**Solution:**
1. Lower `gpu_memory_utilization`
2. Increase `tensor_model_parallel_size`
3. Switch to split config

### Issue: Only 1 GPU active during generation
**Cause:** `tensor_model_parallel_size=1` with multi-GPU setup
**Solution:** Increase TP or use split config

### Issue: Split config shows no speedup
**Cause:** Actor/critic not running in parallel (need `blocking=False`)
**Solution:** Add code modifications or use standard config

### Issue: "AssertionError: only support equal chunk" during validation
**Error:** `Got size of DataProto N and chunk M` (where N < M or N % M != 0)
**Cause:** Validation batch size not divisible by number of critic workers
**Solution:** Ensure `trainer.log_val_generations` is divisible by:
- **Standard config:** `n_gpus_per_node` (e.g., 8 GPUs → use 96, 104, 128, etc.)
- **Split config:** `ray_kwargs.resource_pool.critic.num_gpus` (e.g., 4 GPUs → use 96, 100, 104, etc.)

All provided configs use `log_val_generations: 96` which works for both 4 and 8 GPU setups.

---

## Examples

### Standard Config (Simple)
```bash
# Qwen 1.5B on 8 GPUs - all GPUs used sequentially
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python verl/verl/trainer/nla_main.py \
  --config-name=runs/qwen_1p5b/rl_grpo \
  trainer.experiment_name=my-experiment
```

### Split Config (Advanced)
```bash
# Qwen 7B on 8 GPUs - split placement with TP=2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python verl/verl/trainer/nla_main.py \
  --config-name=runs/qwen_7b/rl_grpo_split \
  trainer.experiment_name=my-experiment-split
```

---

## Quick Start Recommendations

1. **Start with standard config** - simpler and works for most cases
2. **Monitor GPU utilization** with `nvidia-smi`
3. **Switch to split config if:**
   - Model size ≥ 7B
   - OOM errors during generation
   - You need maximum throughput and can modify code
4. **Always verify all GPUs are active** during training

---

## References

- [verl SGLang Documentation](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)
- [Split Placement Example](../../examples/split_placement/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
