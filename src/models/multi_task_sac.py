from src.models.sac import SACAgent, SACConfig


class MTSACAgent(SACAgent):
    """
    Multi-Task SAC Agent.
    Extends the standard SAC agent to handle state + task_id inputs.
    """

    def __init__(
        self, obs_dim, act_dim, num_tasks, act_low, act_high, device, config: SACConfig
    ):
        # Calculate the actual input dimension the network will see
        effective_obs_dim = obs_dim + num_tasks

        super().__init__(
            obs_dim=effective_obs_dim,
            act_dim=act_dim,
            act_low=act_low,
            act_high=act_high,
            device=device,
            config=config,
        )
        self.num_tasks = num_tasks
