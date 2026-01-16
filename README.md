# Reinforcement Learning with Lottery Ticket Hypothesis

A JAX-based implementation of Soft Actor-Critic (SAC) for Meta-World tasks with support for the Lottery Ticket Hypothesis and **parallelized environment execution**.

## 🚀 Quick Start

### Training with Parallelization (Recommended)

For **4-16x faster** training, use parallelized environments:

```bash
# Edit config.yaml to enable parallelization
# Set environments.parallel.enabled = true

python scripts/train_parallel.py
```

### Standard Training

```bash
python scripts/_01_train_dense.py
```

## ⚡ Parallelization Features

This project now supports **vectorized environment execution** for dramatically faster data collection:

- **Sync Mode**: 8-16x speedup for Meta-World tasks
- **Async Mode**: Better for variable-duration environments
- **Easy Configuration**: Just edit `config.yaml`

See **[PARALLELIZATION.md](PARALLELIZATION.md)** for the complete guide.

### Quick Config

```yaml
environments:
  parallel:
    enabled: true      # Enable parallelization
    num_envs: 8        # Number of parallel environments
    strategy: "sync"   # 'sync' or 'async'
```

## 📊 Benchmark Performance

Run the benchmark to test speedup on your hardware:

```bash
python scripts/benchmark_parallel.py
```

Expected output:
```
Single Environment: 120 steps/sec (baseline)

Vectorized Environments:
Config               Rate (steps/sec)     Speedup
--------------------------------------------------------------------------------
8 envs (sync)                   960.0        8.00x
16 envs (sync)                 1680.0       14.00x
```

## 🏗️ Project Structure

```
src/
├── agents/          # SAC agent implementation
├── envs/            # Environment wrappers
│   ├── factory.py          # Single environment creation
│   └── vectorized.py       # Parallel environment execution ⚡NEW
├── training/        # Training loops
│   ├── loops.py            # Single-env training loop
│   └── loops_vectorized.py # Vectorized training loop ⚡NEW
├── lth/             # Lottery Ticket Hypothesis
├── networks/        # Neural network architectures
└── utils/           # Utilities

scripts/
├── train_parallel.py       # Parallelized training ⚡NEW
├── benchmark_parallel.py   # Performance benchmark ⚡NEW
├── _01_train_dense.py      # Standard dense training
├── _02_create_mask.py      # Pruning
└── _03_train_ticket.py     # Lottery ticket training
```

## 📖 Documentation

- **[PARALLELIZATION.md](PARALLELIZATION.md)**: Complete parallelization guide
- **[config.yaml](config.yaml)**: Configuration reference

## 🎯 Features

- ✅ JAX-based SAC implementation
- ✅ Meta-World task support
- ✅ Lottery Ticket Hypothesis experiments
- ✅ **Parallelized environment execution** (NEW)
- ✅ **Sync/Async vectorization strategies** (NEW)
- ✅ Automatic checkpointing
- ✅ Comprehensive logging

## 🔧 Requirements

```bash
pip install jax gymnasium metaworld pyyaml numpy
```

## 💡 Tips

1. **Start with 8 parallel environments** for best balance of speed and stability
2. **Use sync strategy** for Meta-World tasks (faster than async)
3. **Monitor CPU usage** to find optimal `num_envs` for your hardware
4. **Increase batch_size** when using more parallel environments

## 📚 Learn More

- Vectorized environments use Gymnasium's `VectorEnv` API
- The vectorized training loop maintains the same learning dynamics as single-env
- Parallelization only speeds up data collection, not gradient computation
