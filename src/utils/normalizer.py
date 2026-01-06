import torch


class RunningNormalizer:
    """
    Tracks running mean / var and normalizes tensors.
    Lazily infers obs_dim from the first batch.
    """

    def __init__(self, eps: float = 1e-8, device: str = "cpu"):
        self.device = device
        self.eps = eps

        self.count = 0.0
        self.mean = None  # torch.Tensor | None
        self.var = None  # torch.Tensor | None

    @torch.no_grad()
    def _ensure_stats(self, batch_obs: torch.Tensor):
        if self.mean is None or self.var is None:
            obs_dim = batch_obs.shape[-1]
            self.mean = torch.zeros(obs_dim, device=self.device)
            self.var = torch.ones(obs_dim, device=self.device)

    @torch.no_grad()
    def update(self, batch_obs: torch.Tensor):
        if batch_obs is None:
            return

        if batch_obs.ndim == 1:
            batch_obs = batch_obs.unsqueeze(0)

        if batch_obs.numel() == 0:
            return

        batch_obs = batch_obs.to(self.device)
        self._ensure_stats(batch_obs)

        batch_mean = batch_obs.mean(dim=0)
        batch_var = batch_obs.var(dim=0, unbiased=False)
        batch_count = float(batch_obs.shape[0])

        if self.count == 0.0:
            self.mean = batch_mean
            self.var = batch_var + self.eps
            self.count = batch_count
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = (m2 / total_count) + self.eps

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.var is None:
            return obs.to(self.device)
        return (obs.to(self.device) - self.mean) / torch.sqrt(self.var)
