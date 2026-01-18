# Optuna Hyperparameter Tuning for SAC

## Quick Start

```bash
# Run tuning (50 trials, 100k steps each, 8 parallel envs)
python scripts/optuna_tuning.py --task push-v3 --n-trials 50

# Launch dashboard in separate terminal (use your study name)
optuna-dashboard sqlite:///results/optuna/push-v3/<study-name>.db
# Open http://127.0.0.1:8080
```

## Search Space

| Parameter | Range | Notes |
|-----------|-------|-------|
| `actor_lr` | 1e-5 → 1e-3 | Actor learning rate (log scale) |
| `critic_lr` | 1e-5 → 1e-3 | Critic learning rate (log scale) |
| `alpha_lr` | 1e-6 → 1e-4 | Entropy temperature LR |
| `tau` | 0.001 → 0.02 | Target network update rate |
| `init_alpha` | 0.05 → 1.0 | Initial entropy temperature |
| `batch_size` | 256, 512, 1024, 2048, 4096 | Training batch size |
| `actor_hidden` | (128,128), (256,256), etc. | Actor network architecture |
| `critic_hidden` | (128,128), (256,256), etc. | Critic network architecture |

**Design choice**: Actor and critic have separate hyperparameters since they have different learning dynamics. Both critics (Q1/Q2) share the same architecture.

## Options

```bash
python scripts/optuna_tuning.py \
    --task push-v3 \      # MetaWorld task name
    --n-trials 50 \       # Number of trials
    --tuning-steps 100000 \ # Steps per trial
    --num-envs 8 \        # Parallel envs (default: 8 for M1 Max)
    --study-name study    # Study name (default: "study")
```

## Features

- **Vectorized envs**: 8 parallel environments for ~8x faster data collection
- **MedianPruner**: Kills underperforming trials early (5 checkpoints per trial)
- **SQLite storage**: Auto-resumes interrupted studies
- **Progress logging**: Shows step progress and eval results
