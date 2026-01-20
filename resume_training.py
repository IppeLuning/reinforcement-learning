"""
Resume training from a checkpoint.
Usage: uv run python resume_training.py
"""

import os
import pickle
import yaml

from scripts._03_train_ticket import train_mask_with_buffer


def main():
    """Resume Round 3 training from the last checkpoint."""
    
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Configuration for the interrupted run
    task = "reach-v3"
    seed = 3000
    round_num = 3
    
    # Paths
    base_exp_dir = f"data/experiments/{task}/seed_{seed}"
    current_round_dir = os.path.join(base_exp_dir, f"round_{round_num}")
    prev_round_dir = os.path.join(base_exp_dir, f"round_{round_num - 1}")
    mask_path = os.path.join(current_round_dir, "mask.pkl")
    rewind_ckpt_path = os.path.join(base_exp_dir, "round_0/checkpoint_rewind.pkl")
    
    # Find the latest checkpoint to resume from
    checkpoints = []
    for f in os.listdir(current_round_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pkl"):
            step = int(f.replace("checkpoint_step_", "").replace(".pkl", ""))
            checkpoints.append((step, f))
    
    if not checkpoints:
        print("No checkpoints found to resume from!")
        return
    
    # Sort by step and get the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_ckpt = checkpoints[0]
    resume_checkpoint = os.path.join(current_round_dir, latest_ckpt)
    
    # Find replay buffer from previous round to pre-fill
    replay_buffer_path = os.path.join(prev_round_dir, "replay_buffer.pkl")
    
    print(f"\n{'='*60}")
    print(f"RESUMING TRAINING")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Seed: {seed}")
    print(f"Round: {round_num}")
    print(f"Resume from: {resume_checkpoint} (step {latest_step})")
    print(f"Target: {cfg['hyperparameters']['total_steps']} steps")
    if os.path.exists(replay_buffer_path):
        print(f"Pre-filling buffer from: {replay_buffer_path}")
    print(f"{'='*60}\n")
    
    # Run training with resume and pre-filled buffer
    train_mask_with_buffer(
        cfg=cfg,
        task_name=task,
        seed=seed,
        mask_path=mask_path,
        save_dir=current_round_dir,
        rewind_ckpt_path=rewind_ckpt_path,
        resume_checkpoint=resume_checkpoint,
        replay_buffer_path=replay_buffer_path if os.path.exists(replay_buffer_path) else None,
    )
    
    print("\n>> Training resumed and completed!")


if __name__ == "__main__":
    main()
