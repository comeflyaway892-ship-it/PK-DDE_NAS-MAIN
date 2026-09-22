import math

import torch


class PresetPriorMatrixScheduler:
    """
    D3PM scheduler with per-position architecture-prior kernels.

    Q_t,d = (1 - beta_t) I + beta_t M_d

    For a marginal prior, M_d = 1 pi_d^T and Qbar has the familiar closed
    form. For structural kernels, Qbar is computed by multiplying transition
    matrices, so the same scheduler supports both cases.
    """

    def __init__(
        self,
        num_steps,
        vocab_size,
        prior_kernel,
        eps=1e-5,
        schedule="cosine",
        cosine_s=0.008,
        token_mask=None,
    ):
        self.T = int(num_steps)
        self.K = int(vocab_size)
        self.eps = float(eps)
        self.schedule = schedule
        self.token_mask = None if token_mask is None else token_mask.bool()
        self.prior_kernel = self._prepare_prior_kernel(prior_kernel)

        self.alpha_bar = self._build_alpha_bar(self.T, self.eps, schedule, cosine_s)
        self.beta = self._alpha_bar_to_beta(self.alpha_bar)
        self.Q = self._build_Q_matrices(self.beta, self.K, self.prior_kernel, self.token_mask)
        self.Qbar = self._build_Qbar_matrices(self.Q)

    def _prepare_prior_kernel(self, prior_kernel):
        prior_kernel = torch.as_tensor(prior_kernel, dtype=torch.float32)
        if prior_kernel.dim() != 3 or prior_kernel.shape[-2:] != (self.K, self.K):
            raise ValueError(
                "prior_kernel must have shape [D, K, K], "
                f"got {tuple(prior_kernel.shape)} with K={self.K}"
            )
        if torch.any(prior_kernel < 0):
            raise ValueError("prior_kernel cannot contain negative entries")
        return prior_kernel / prior_kernel.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    def _build_alpha_bar(self, T, eps, schedule, cosine_s):
        schedule = schedule.lower()
        if schedule in ("linear", "uniform"):
            return self._build_alpha_bar_linear(T, eps)
        if schedule in ("cosine", "cos"):
            return self._build_alpha_bar_cosine(T, eps, cosine_s)
        raise ValueError(f"Unknown schedule={schedule!r}. Use 'linear' or 'cosine'.")

    def _build_alpha_bar_linear(self, T, eps):
        t = torch.arange(0, T + 1, dtype=torch.float32)
        alpha_bar = 1.0 - (t / T) * (1.0 - float(eps))
        return alpha_bar.clamp(min=float(eps), max=1.0)

    def _build_alpha_bar_cosine(self, T, eps, s=0.008):
        t = torch.arange(0, T + 1, dtype=torch.float32) / T
        factor = (t + s) / (1.0 + s)
        f = torch.cos(factor * (math.pi / 2)) ** 2
        f0 = torch.cos(torch.tensor(s / (1.0 + s)) * (math.pi / 2)) ** 2
        alpha_bar = (f / f0).clamp(min=float(eps), max=1.0)
        alpha_bar[0] = 1.0
        return torch.minimum(alpha_bar, torch.cummin(alpha_bar, dim=0)[0])

    def _alpha_bar_to_beta(self, alpha_bar):
        prev = alpha_bar[:-1]
        curr = alpha_bar[1:]
        beta = 1.0 - (curr / prev)
        return beta.clamp(min=0.0, max=1.0 - 1e-8)

    def _build_Q_matrices(self, beta, k, prior_kernel, token_mask):
        T = beta.shape[0]
        D = prior_kernel.shape[0]
        I = torch.eye(k, dtype=beta.dtype, device=beta.device).view(1, 1, k, k)
        M = prior_kernel.to(device=beta.device, dtype=beta.dtype).view(1, D, k, k)
        b = beta.view(T, 1, 1, 1)
        Q = (1.0 - b) * I + b * M

        if token_mask is not None:
            token_mask = token_mask.to(device=beta.device, dtype=beta.dtype)
            Q = Q * token_mask.view(1, D, 1, k)
        return Q / Q.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    def _build_Qbar_matrices(self, Q):
        T, D, K, _ = Q.shape
        Qbar = torch.empty((T + 1, D, K, K), dtype=Q.dtype, device=Q.device)
        Qbar[0] = torch.eye(K, dtype=Q.dtype, device=Q.device).view(1, K, K).repeat(D, 1, 1)
        running = Qbar[0]
        for t in range(1, T + 1):
            running = torch.matmul(running, Q[t - 1])
            running = running / running.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            Qbar[t] = running
        return Qbar

    def get_Q(self, t):
        if not (1 <= t <= self.T):
            raise ValueError(f"t must be in [1..{self.T}]")
        return self.Q[t - 1]

    def get_Q_bar(self, t):
        if not (0 <= t <= self.T):
            raise ValueError(f"t must be in [0..{self.T}]")
        return self.Qbar[t]
