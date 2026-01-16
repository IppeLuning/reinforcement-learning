import json
import os
from typing import Any, Dict

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env
from src.training import run_training_loop
from src.utils import Checkpointer, set_seed


def train_dense(cfg: Dict[str, Any], task_name: str, seed: int, save_dir: str) -> None:
    """
    Executes Step 1 of the LTH pipeline:
    1. Initializes network.
    2. Saves 'checkpoint_init.pkl' (The Ticket / W0).
    3. Trains to completion.
    4. Saves 'checkpoint_final.pkl' (For Pruning).
    """

    # Check if already done to save time (optional safety)
    if os.path.exists(os.path.join(save_dir, "checkpoint_final.pkl")):
        print(f"  [Skip] Dense training already completed for {task_name} seed {seed}")
        return

    print(f"  > Starting Dense Training...")

    # 1. Setup Configuration
    # We assume cfg['hyperparameters'] structure based on your previous config.yaml
    hp = cfg["hyperparameters"]
    defaults = hp["defaults"]
    task_overrides = hp.get("tasks", {}).get(task_name, {})

    # Merge: Overrides > Defaults
    params = {**defaults, **task_overrides}

    # Extract specific params
    hidden_dims = tuple(hp["hidden_dims"])
    total_steps = hp["total_steps"]
    start_steps = hp["start_steps"]

    # 2. Initialize Environment & Seeding
    set_seed(seed)
    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name, hp["max_episode_steps"], seed
    )

    # 3. Initialize Agent
    # For Dense training, use_masking is ALWAYS False
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),  # Enable automatic alpha tuning
        init_alpha=params.get("init_alpha", 0.2),
        hidden_dims=hidden_dims,
        use_masking=False,
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )

    # 4. Infrastructure
    buffer = ReplayBuffer(
        obs_dim, act_dim, max_size=hp.get("replay_buffer_size", 1_000_000)
    )
    checkpointer = Checkpointer(save_dir)

    # 5. [CRITICAL LTH STEP] Save Initial Weights (W0)
    # This must happen before any training updates
    print(f"  > Saving W0 (checkpoint_init.pkl)...")
    checkpointer.save(agent.state, filename="checkpoint_init")

    # 6. Run Training
    stats = run_training_loop(
        env=env,
        agent=agent,
        replay_buffer=buffer,
        total_steps=total_steps,
        start_steps=start_steps,
        batch_size=hp["batch_size"],
        eval_interval=hp["eval_interval"],
        save_dir=save_dir,
        seed=seed,
        task_name=task_name,
        target_mean_success=params.get("target_mean_success", None),
        patience=params.get("patience", 20),
        updates_per_step=params.get("updates_per_step", 1),
        eval_episodes=hp.get("eval_episodes", 5),
        checkpointer=checkpointer,
    )

    # 7. Cleanup
    env.close()

    # Save simple JSON log
    with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
        # Filter out heavy arrays just in case
        serializable_stats = {k: v for k, v in stats.items() if k != "metrics_history"}
        json.dump(serializable_stats, f, indent=2)

    print(f"  > Dense training complete. Best success: {stats['best_success']:.2%}")
