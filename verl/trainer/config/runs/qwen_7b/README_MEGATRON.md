# Qwen 7B NLA GRPO Configurations

This directory contains configurations for training Qwen2.5-7B-Instruct with NLA GRPO using different backends.

## Available Configurations

### 1. `rl_grpo.yaml` - FSDP (Original)
**Backend**: FSDP2
**GPUs**: 8 (all GPUs used sequentially for actor, then critic)
**Memory**: Uses parameter and optimizer offloading

```bash
python verl/trainer/nla_main.py --config-name=runs/qwen_7b/rl_grpo
```

**Characteristics**:
- ✅ Simple configuration
- ✅ Good for single-node training
- ✅ Parameter offloading for memory efficiency
- ⚠️ Sequential GPU usage (actor → critic)

---

### 2. `rl_grpo_megatron.yaml` - Megatron TP=4
**Backend**: Megatron-LM
**GPUs**: 8 (4 for actor TP=4, 4 for critic TP=4)
**Memory**: Tensor parallelism sharding, activation checkpointing

```bash
python verl/trainer/nla_main.py --config-name=runs/qwen_7b/rl_grpo_megatron
```

**Characteristics**:
- ✅ Tensor parallelism (TP=4) for faster training
- ✅ Parallel actor and critic training
- ✅ Suitable for multi-node scaling
- ⚠️ Requires PP=1 (NLA limitation)
- ⚠️ More complex setup

**Resource Allocation**:
```
GPUs 0-3: Actor with TP=4
GPUs 4-7: Critic with TP=4
```

---

### 3. `rl_grpo_megatron_tp2.yaml` - Megatron TP=2
**Backend**: Megatron-LM
**GPUs**: 2 (both for actor and critic with TP=2)
**Memory**: Tensor parallelism sharding

```bash
python verl/trainer/nla_main.py --config-name=runs/qwen_7b/rl_grpo_megatron_tp2
```

**Characteristics**:
- ✅ Smaller footprint for testing
- ✅ Good for development/debugging
- ✅ Faster startup than TP=4
- ⚠️ Sequential actor/critic (shares 2 GPUs)

**Use Cases**:
- Testing Megatron support on smaller hardware
- Debugging TP behavior with fewer ranks
- Development iteration

---

## Key Configuration Differences

| Setting | FSDP | Megatron TP=4 | Megatron TP=2 |
|---------|------|---------------|---------------|
| **Strategy** | `fsdp2` | `megatron` | `megatron` |
| **GPUs** | 8 | 8 | 2 |
| **TP Size** | N/A | 4 | 2 |
| **PP Size** | N/A | 1 (required) | 1 (required) |
| **Actor GPUs** | 8 (sequential) | 4 (dedicated) | 2 (shared) |
| **Critic GPUs** | 8 (sequential) | 4 (dedicated) | 2 (shared) |
| **Offloading** | Yes | No | No |
| **Micro Batch/GPU** | 8 | 4 | 8 |
| **Mini Batch** | 512 | 512 | 512 |

---

## Important Megatron Limitations

### Pipeline Parallelism (PP) Must Be 1

**Why?** NLA critic extracts activation vectors from an intermediate layer (layer 20 for Qwen 7B). With PP>1:
- Different layers are on different pipeline stages
- Only final stage has complete `hidden_states`
- Cannot extract intermediate layer without custom pipeline hooks

**Configuration**:
```yaml
critic:
  megatron:
    pipeline_model_parallel_size: 1  # MUST be 1
  output_layer_index: 20  # Layer to extract from
```

### Tensor Parallelism (TP) Fully Supported

✅ TP works perfectly for NLA:
- Actor embedding extraction handles TP automatically
- Critic hidden state extraction works across TP ranks
- Can scale to large TP values (TP=8, TP=16, etc.)

---

## Batch Size Calculations

All configs use the same effective batch size:

```
Prompts per step: 64
Responses per prompt (rollout.n): 8
Total experiences: 64 × 8 = 512
Mini batch size: 512 (one gradient update)
```

**Micro batch per GPU varies**:
- FSDP: 8 per GPU × 8 GPUs = 64 total capacity
- Megatron TP=4: 4 per GPU × 8 GPUs (4 TP groups of 2) = 32 total capacity
- Megatron TP=2: 8 per GPU × 2 GPUs (1 TP group) = 16 total capacity

---

## Resource Pool Configuration

### FSDP (No Resource Pools)
```yaml
# No resource_pool specified
# verl uses all 8 GPUs sequentially:
# Step 1: Actor rollout on all 8 GPUs
# Step 2: Critic training on all 8 GPUs
```

### Megatron TP=4 (Split Pools)
```yaml
resource_pool:
  actor_rollout:
    num_gpus: 4  # GPUs 0-3
  critic:
    num_gpus: 4  # GPUs 4-7
# Actor and critic can run in parallel
```

### Megatron TP=2 (Shared Pool)
```yaml
resource_pool:
  actor_rollout:
    num_gpus: 2  # GPUs 0-1
  critic:
    num_gpus: 2  # GPUs 0-1 (same as actor)
# Actor and critic share GPUs, run sequentially
```

---

## Performance Considerations

### FSDP
- **Pros**: Simple, good single-node performance, memory offloading
- **Cons**: Sequential GPU usage, slower than TP for large models
- **Best for**: Single-node training, memory-constrained setups

### Megatron TP=4
- **Pros**: Faster with TP, parallel actor/critic, multi-node ready
- **Cons**: More complex, requires more GPUs
- **Best for**: Production training, multi-node scaling, large models

### Megatron TP=2
- **Pros**: Easier testing, faster iteration, smaller footprint
- **Cons**: Sequential actor/critic, limited by 2 GPUs
- **Best for**: Development, debugging, small-scale experiments

---

## Troubleshooting

### Error: "NLA Megatron critic does not support pipeline parallelism (PP=X)"
**Solution**: Set `pipeline_model_parallel_size: 1` in both actor and critic Megatron configs.

### Error: "Embedding layer not initialized"
**Cause**: Actor rollout trying to prepare embeddings on non-first pipeline stage.
**Solution**: Ensure PP=1, or check that embedding extraction is guarded by `mpu.is_pipeline_first_stage()`.

### Out of Memory
**Solutions**:
1. Enable activation checkpointing: `recompute_num_layers: 1`
2. Reduce micro batch size: `ppo_micro_batch_size_per_gpu: 2`
3. Use larger TP: `tensor_model_parallel_size: 8`
4. Enable sequence parallel: `sequence_parallel: true`

### Slow Training
**Solutions**:
1. Increase TP size for better parallelism
2. Use split resource pools to enable parallel actor/critic
3. Disable unnecessary recomputation: `recompute_num_layers: 0`
4. Tune `gpu_memory_utilization` for SGLang rollout

---

## Migration Guide: FSDP → Megatron

To convert an FSDP config to Megatron:

1. **Change strategy**:
   ```yaml
   actor_rollout_ref:
     actor:
       strategy: megatron  # was: fsdp2
   ```

2. **Add Megatron config**:
   ```yaml
   actor:
     megatron:
       tensor_model_parallel_size: 4
       pipeline_model_parallel_size: 1
       override_transformer_config:
         bf16: true
         params_dtype: bfloat16
   ```

3. **Remove FSDP config**:
   ```yaml
   actor:
     fsdp_config: null  # Remove FSDP settings
   ```

4. **Adjust batch sizes** (Megatron typically uses smaller micro batches)

5. **Configure resource pools** (for split placement)

6. **Repeat for critic**

---

## References

- Main docs: `docs/MEGATRON_SUPPORT.md`
- Verification: `docs/MEGATRON_SGLANG_VERIFICATION.md`
- Base config: `config/nla_grpo_full.yaml`
