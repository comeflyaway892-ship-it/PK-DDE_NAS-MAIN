import math

import torch


class PresetPriorMatrixScheduler:
    def __init__(self, num_steps, vocab_size, prior_kernel, eps=1e-5, schedule="cosine", cosine_s=0.008):
        self.T = int(num_steps)
        self.K = int(vocab_size)
        self.prior_kernel = self._prepare_prior_kernel(prior_kernel)
        self.alpha_bar = self._build_alpha_bar(self.T, float(eps), schedule, float(cosine_s))
        self.beta = self._alpha_bar_to_beta(self.alpha_bar)
        self.Q = self._build_Q_matrices(self.beta, self.K, self.prior_kernel)
        self.Qbar = self._build_Qbar_matrices(self.Q)

    def _prepare_prior_kernel(self, prior_kernel):
        prior_kernel = torch.as_tensor(prior_kernel, dtype=torch.float32)
        if prior_kernel.dim() != 3 or prior_kernel.shape[-2:] != (self.K, self.K):
            raise ValueError(f"prior_kernel must have shape [D,{self.K},{self.K}], got {tuple(prior_kernel.shape)}")
        if torch.any(prior_kernel < 0):
            raise ValueError("prior_kernel cannot contain negative entries")
        return prior_kernel / prior_kernel.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    def _build_alpha_bar(self, T, eps, schedule, cosine_s):
        schedule = schedule.lower()
        if schedule in ("linear", "uniform"):
            t = torch.arange(0, T + 1, dtype=torch.float32)
            return (1.0 - (t / T) * (1.0 - eps)).clamp(min=eps, max=1.0)
        if schedule in ("cosine", "cos"):
            t = torch.arange(0, T + 1, dtype=torch.float32) / T
            factor = (t + cosine_s) / (1.0 + cosine_s)
            f = torch.cos(factor * (math.pi / 2)) ** 2
            f0 = torch.cos(torch.tensor(cosine_s / (1.0 + cosine_s)) * (math.pi / 2)) ** 2
            alpha_bar = (f / f0).clamp(min=eps, max=1.0)
            alpha_bar[0] = 1.0
            return torch.minimum(alpha_bar, torch.cummin(alpha_bar, dim=0)[0])
        raise ValueError(f"Unknown schedule={schedule!r}")

    def _alpha_bar_to_beta(self, alpha_bar):
        return (1.0 - (alpha_bar[1:] / alpha_bar[:-1])).clamp(min=0.0, max=1.0 - 1e-8)

    def _build_Q_matrices(self, beta, k, prior_kernel):
        T = beta.shape[0]
        D = prior_kernel.shape[0]
        I = torch.eye(k, dtype=beta.dtype, device=beta.device).view(1, 1, k, k)
        M = prior_kernel.to(device=beta.device, dtype=beta.dtype).view(1, D, k, k)
        Q = (1.0 - beta.view(T, 1, 1, 1)) * I + beta.view(T, 1, 1, 1) * M
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
