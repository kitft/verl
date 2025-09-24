# SGLang Embedding Integration for NLA

## Overview
This document describes how NLA (Natural Language Autoencoder) integrates with SGLang using custom input embeddings for activation injection.

## Data Flow

1. **NLAActorRolloutRefWorker** (`nla_actor_worker.py`)
   - Extracts activation vectors from DataProto
   - Finds injection positions (location of special tokens)
   - Passes both via `meta_info` to the rollout

2. **NLASGLangRollout** (`nla_sglang_rollout.py`)
   - Receives activation vectors and positions from meta_info
   - Calls `_prepare_input_embeddings()` to create modified embeddings
   - **IMPORTANT**: Converts embeddings to CPU for network transmission
   - Stores `input_embeds` in `prompts.batch` (not meta_info)

3. **SGLangRollout** (parent class)
   - Receives the batch with `input_embeds`
   - Passes to SGLang engine's `async_generate()` method

## Key Implementation Details

### Input Embeddings Format
```python
# Shape: (batch_size, seq_len, hidden_dim)
prompts.batch['input_embeds'] = input_embeds  # CPU tensor
```

### Network Transmission Warning
⚠️ **Always convert embeddings to CPU before passing to SGLang**

SGLang transmits data over the network to its inference engine. GPU tensors would need to be copied anyway, so we do it explicitly:

```python
if input_embeds.is_cuda:
    print("WARNING: Converting input_embeds to CPU for network transmission")
    input_embeds = input_embeds.cpu()
```

### Embedding Preparation Process

1. **Get base embeddings**: Use the model's embedding layer to convert token IDs to embeddings
2. **Project if needed**: If activation vectors have different dimensions, project to hidden_dim
3. **Inject activations**: Replace embeddings at injection_positions with activation vectors
4. **Convert to CPU**: Ensure embeddings are on CPU for efficient network transmission

## Configuration

In your training config (e.g., `nla_grpo_tiny.yaml`):

```yaml
actor_rollout_ref:
  rollout:
    name: nla_sglang  # Use NLA-specific SGLang rollout
    mode: sync        # Sync mode for simpler setup
```

## Fallback Handling

If SGLang doesn't expose the model's embedding layer, the system will:
1. Try multiple access patterns (`.model`, `.embed_tokens`, `.model.embed_tokens`)
2. Fall back to creating a simple embedding layer if needed
3. Log warnings about suboptimal results

## Testing

To verify the integration works:
1. Check logs for "NLA SGLang: Prepared input_embeds with shape: ..."
2. Verify CPU conversion warning appears
3. Confirm injection positions are found correctly

## Future Improvements

1. **Learned Projection**: Instead of simple linear projection, use a trained projection layer
2. **Caching**: Cache base embeddings for repeated tokens
3. **Batch Optimization**: Optimize the injection loop for better performance
4. **Direct Model Access**: Work with SGLang team to expose embedding layer more reliably