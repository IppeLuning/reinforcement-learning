from typing import Any, Dict

from src.training.loops import train_multitask_session, train_single_task_session


class Pipeline:
    """
    Orchestrates the training of single-task and multi-task SAC agents
    based on the loaded configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.single_task_cfg = config.get("single_task", {})
        self.multi_task_cfg = config.get("multi_task", {})
        self.env_cfg = config.get("environments", {})
        self.default_cfg = config.get("defaults", {})

    def _train_single_task_mode(self):
        """Trains all specified tasks, each across all configured seeds."""

        if not self.single_task_cfg.get("enabled", False):
            print("Single-Task training is disabled in the configuration.")
            return

        print("\n=== STARTING SINGLE-TASK TRAINING PHASE ===")

        tasks = self.env_cfg["tasks"]
        seeds = self.single_task_cfg["seeds"]

        for task in tasks:
            for seed in seeds:
                try:
                    train_single_task_session(
                        cfg=self.config, task_name=task, seed=seed
                    )
                except Exception as e:
                    print(f"Error training single task {task} with seed {seed}: {e}")

    def _train_multi_task_mode(self):
        """Trains the multi-task agent across all configured seeds."""

        if not self.multi_task_cfg.get("enabled", False):
            print("Multi-Task training is disabled in the configuration.")
            return

        print("\n=== STARTING MULTI-TASK TRAINING PHASE ===")

        seeds = self.multi_task_cfg["seeds"]

        for seed in seeds:
            try:
                train_multitask_session(cfg=self.config, seed=seed)
            except Exception as e:
                print(f"Error training multi-task agent with seed {seed}: {e}")

    def execute(self):
        """
        Orchestrates the training pipeline.
        """
        print(f"--- Pipeline Execution Started ---")

        self._train_single_task_mode()
        self._train_multi_task_mode()
        # eval and metrics
        # create videos?
        # create plots?

        # todo: winning tickets, comparisons, etc.

        print("--- Pipeline Execution Finished ---")
