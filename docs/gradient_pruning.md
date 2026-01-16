# Saliency-Based Gradient Pruning

This implementation provides gradient-based pruning for Lottery Ticket Hypothesis experiments, as an alternative to magnitude-based pruning.

## Overview

**Magnitude Pruning** (original): Removes weights with smallest `|w|`

**Gradient Saliency Pruning** (new): Removes weights with smallest importance score based on gradients

### Available Methods

1. **Taylor Expansion** (`method="taylor"`): 
   - Importance: `|w * ∂L/∂w|`
   - First-order approximation of loss change when removing weight
   - **Recommended** for best performance
   
2. **Pure Gradient** (`method="gradient"`):
   - Importance: `|∂L/∂w|`
   - Weights with largest gradients are kept
   
3. **Magnitude** (`method="magnitude"`):
   - Importance: `|w|`
   - Same as original implementation (for comparison)

## Usage

### Option 1: Use the New Script (Easiest)

```bash
# Create gradient-based masks
python scripts/_02b_create_mask_gradient.py
```

This will:
1. Load your trained dense models
2. Sample batches from the replay buffer
3. Compute accumulated gradients
4. Create masks using Taylor saliency
5. Save masks as `mask_gradient.pkl`

### Option 2: Modify Existing Script

Update `scripts/_02_create_mask.py` to use gradient pruning:

```python
from src.lth import (
    prune_kernels_by_gradient_saliency,
    accumulate_gradient_statistics,
    compute_sparsity
)

# ... after loading agent and replay buffer ...

# Compute gradients
actor_grads, critic_grads = accumulate_gradient_statistics(
    agent=agent,
    replay_buffer=replay_buffer,
    num_batches=100,  # More batches = more stable estimates
    batch_size=256,
)

# Prune with gradient saliency
actor_mask = prune_kernels_by_gradient_saliency(
    params=agent.state.actor_params,
    gradients=actor_grads,
    target_sparsity=0.8,
    method="taylor"  # or "gradient" or "magnitude"
)
```

### Option 3: Direct API Usage

```python
from src.lth.gradient_pruning import (
    prune_kernels_by_gradient_saliency,
    compute_gradients_from_batch,
)

# Compute gradients from a single batch
batch = replay_buffer.sample(256)
actor_grads, critic_grads = compute_gradients_from_batch(
    agent, batch, normalize_obs=True
)

# Create mask
mask = prune_kernels_by_gradient_saliency(
    params=agent.state.actor_params,
    gradients=actor_grads,
    target_sparsity=0.8,
    method="taylor"
)
```

## Configuration

Add to your `config.yaml`:

```yaml
pruning:
  sparsity: 0.8  # 80% of weights pruned
  gradient_method: "taylor"  # "taylor", "gradient", or "magnitude"
  num_gradient_batches: 100  # More = better estimates
  gradient_batch_size: 256
```

## Theory

### Why Gradient Saliency?

Magnitude pruning assumes `|w|` indicates importance, but a large weight with small gradient contributes less to the loss than a smaller weight with large gradient.

The **Taylor expansion** approximates the change in loss when removing a weight:
```
ΔL ≈ w * ∂L/∂w  (first-order)
```

Therefore, `|w * ∂L/∂w|` is a better estimate of weight importance.

### References

- Molchanov et al. (2017): "Pruning Convolutional Neural Networks for Resource Efficient Inference"
- Lee et al. (2018): "SNIP: Single-shot Network Pruning based on Connection Sensitivity"

## Comparison with Magnitude Pruning

Run experiments with both methods:

```bash
# Magnitude pruning (original)
python scripts/_02_create_mask.py

# Gradient saliency pruning
python scripts/_02b_create_mask_gradient.py

# Train tickets with both
python scripts/_03_train_ticket.py  # uses mask.pkl
# Or modify to use mask_gradient.pkl
```

Expected results:
- Gradient saliency should find better tickets (higher final performance)
- Especially effective at higher sparsities (90%+)
- More computationally expensive (requires gradient computation)

## Implementation Notes

### Gradient Accumulation

We accumulate gradients over multiple batches for stability:
```python
grad_avg = Σ |grad_i| / N
```

This reduces noise from individual batch statistics.

### Computational Cost

- **Magnitude pruning**: O(1) - just threshold parameters
- **Gradient pruning**: O(N) - compute N forward/backward passes

For typical setups with 100 batches, this adds ~30 seconds per task/seed.

### Memory Requirements

Gradients have the same shape as parameters, so memory usage is 2x (params + grads). This is not an issue for standard SAC networks.

## Troubleshooting

### Error: "No replay buffer found"

Make sure you saved the replay buffer during dense training. Update `scripts/_01_train_dense.py`:

```python
# Save replay buffer for gradient computation
buffer_data = agent.replay_buffer.save()
with open(output_dir / "replay_buffer.pkl", "wb") as f:
    pickle.dump(buffer_data, f)
```

### Different sparsity than expected

The actual sparsity might differ slightly from target due to:
- Ties in saliency scores at the threshold
- Integer rounding when computing k

This is normal and typically within 0.1% of target.

### NaN gradients

If you see NaN gradients:
1. Check your dense model checkpoint is valid
2. Verify replay buffer has valid data
3. Try reducing `num_gradient_batches`
