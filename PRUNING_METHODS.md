# Pruning Methods Guide

Your pipeline now supports both magnitude-based and gradient-based pruning!

## Quick Start

Edit [run_pipeline.py](run_pipeline.py#L49) to choose your pruning method:

```python
if __name__ == "__main__":
    main(
        train_agent=True,
        create_ticket=True,
        run_ticket=True,
        pruning_method="magnitude",  # Change to "gradient" for better performance
    )
```

Then run:
```bash
uv run python run_pipeline.py
```

## Pruning Methods

### 1. Magnitude-Based Pruning (Default)

**Method**: `pruning_method="magnitude"`

- **How it works**: Removes weights with smallest absolute values `|w|`
- **Speed**: Very fast (no gradient computation needed)
- **Performance**: Good baseline
- **Requirements**: Only needs trained model weights
- **Use when**: You want quick results or baseline comparisons

**Mask filename**: `data/masks/{task}_seed_{seed}.pkl`

### 2. Gradient-Based Pruning (Recommended)

**Method**: `pruning_method="gradient"`

- **How it works**: Removes weights with smallest saliency `|w * ∂L/∂w|`
- **Speed**: Slower (requires gradient computation over replay buffer)
- **Performance**: Better lottery tickets, especially at high sparsities
- **Requirements**: Replay buffer saved during training (automatically done)
- **Use when**: You want best performance and can afford extra computation

**Mask filename**: `data/masks/{task}_seed_{seed}_gradient.pkl`

## Configuration

Edit [config.yaml](config.yaml):

```yaml
pruning:
  sparsity: 0.8  # 80% of weights pruned
  
  # Gradient pruning settings (only used with pruning_method="gradient")
  gradient_method: "taylor"  # Options: "taylor", "gradient", "magnitude"
  num_gradient_batches: 100  # More batches = more stable estimates
  gradient_batch_size: 256
```

### Gradient Method Options

- **`taylor`** (recommended): Importance = `|w * ∂L/∂w|` (first-order Taylor expansion)
- **`gradient`**: Importance = `|∂L/∂w|` only
- **`magnitude`**: Importance = `|w|` only (same as magnitude pruning, for testing)

## Comparison

| Aspect | Magnitude | Gradient (Taylor) |
|--------|-----------|-------------------|
| Speed | ✅ Very Fast | ⚠️ ~30s extra per task |
| Performance | Good | Better |
| Setup | None | Replay buffer needed |
| Best for | Quick experiments | Final results |

## Running Experiments

### Compare Both Methods

```python
# First run with magnitude
main(pruning_method="magnitude")

# Then run with gradient
main(pruning_method="gradient")
```

This creates two sets of masks for comparison.

### Full Experiment Pipeline

```python
main(
    train_agent=True,      # Step 1: Train dense networks
    create_ticket=True,    # Step 2: Create pruning masks
    run_ticket=True,       # Step 3: Train lottery tickets
    pruning_method="gradient",
)
```

## Troubleshooting

### Error: "Replay buffer not found"

If you see this error with gradient pruning, your dense training didn't save the replay buffer.

**Solution**: Re-run dense training:
```python
main(train_agent=True, create_ticket=False, run_ticket=False)
```

The updated training script now automatically saves the replay buffer.

### Memory Issues

If gradient pruning uses too much memory:
- Reduce `num_gradient_batches` in config.yaml (try 50)
- Reduce `gradient_batch_size` (try 128)

### Which method should I use?

- **For quick testing**: Use magnitude
- **For paper results**: Use gradient with `taylor` method
- **For comparison**: Run both and compare performance
