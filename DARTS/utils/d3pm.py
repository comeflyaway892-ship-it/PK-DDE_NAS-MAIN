"""D3PM scheduler with a separate vocabulary size for every token position."""

import math

import torch


class D3PMVariableVocabScheduler:
    """Categorical D3PM with a separate vocabulary/prior for each position."""

    def __init__(
        self,
        num_steps,
        vocab_sizes,
        eps=1e-5,
        schedule="cosine",
        cosine_s=0.008,
        prior_kernels=None,
    ):
        self.T = int(num_steps)
        if self.T < 1:
            raise ValueError("num_steps must be positive")
        self.vocab_sizes = tuple(int(size) for size in vocab_sizes)
        if not self.vocab_sizes or any(size < 2 for size in self.vocab_sizes):
            raise ValueError("Every token position must have at least two categories")

        self.eps = float(eps)
        self.alpha_bar = self._build_alpha_bar(schedule, float(cosine_s))
        self.beta = self._alpha_bar_to_beta(self.alpha_bar)
        self.prior_kernels = []
        for position, vocab_size in enumerate(self.vocab_sizes):
            if prior_kernels is None:
                prior = torch.full((vocab_size, vocab_size), 1.0 / vocab_size)
            else:
                prior = torch.as_tensor(prior_kernels[position], dtype=torch.float32)
                if prior.shape != (vocab_size, vocab_size):
                    raise ValueError(
                        "prior kernel at position {} must have shape ({}, {}), got {}".format(
                            position, vocab_size, vocab_size, tuple(prior.shape)
                        )
                    )
                if torch.any(prior < 0):
                    raise ValueError("prior kernels cannot contain negative entries")
            self.prior_kernels.append(
                prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            )

        # Cache matrices only when positions share both vocabulary and prior.
        self.Q = {}
        self.Qbar = {}
        self._position_q = {}
        self._position_qbar = {}
        for position, vocab_size in enumerate(self.vocab_sizes):
            prior = self.prior_kernels[position]
            key = (vocab_size, prior.detach().cpu().numpy().tobytes())
            if key not in self.Q:
                q = self._build_q(self.beta, vocab_size, prior)
                self.Q[key] = q
                self.Qbar[key] = self._build_qbar(q)
            self._position_q[position] = self.Q[key]
            self._position_qbar[position] = self.Qbar[key]

    def _build_alpha_bar(self, schedule, cosine_s):
        schedule = schedule.lower()
        if schedule in ("linear", "uniform"):
            t = torch.arange(0, self.T + 1, dtype=torch.float32)
            alpha_bar = 1.0 - (t / self.T) * (1.0 - self.eps)
        elif schedule in ("cosine", "cos"):
            t = torch.arange(0, self.T + 1, dtype=torch.float32) / self.T
            factor = (t + cosine_s) / (1.0 + cosine_s)
            f = torch.cos(factor * (math.pi / 2.0)).square()
            f0 = math.cos(cosine_s / (1.0 + cosine_s) * math.pi / 2.0) ** 2
            alpha_bar = f / f0
        else:
            raise ValueError(
                "Unknown schedule '{}'; use 'linear' or 'cosine'".format(schedule)
            )

        alpha_bar = alpha_bar.clamp(min=self.eps, max=1.0)
        alpha_bar[0] = 1.0
        return torch.minimum(alpha_bar, torch.cummin(alpha_bar, dim=0)[0])

    @staticmethod
    def _alpha_bar_to_beta(alpha_bar):
        beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return beta.clamp(min=0.0, max=1.0 - 1e-8)

    @staticmethod
    def _build_q(beta, vocab_size, prior=None):
        identity = torch.eye(vocab_size, dtype=beta.dtype)
        if prior is None:
            prior = torch.full((vocab_size, vocab_size), 1.0 / vocab_size)
        prior = prior.to(dtype=beta.dtype)
        weight = beta.view(-1, 1, 1)
        q = (1.0 - weight) * identity + weight * prior
        return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    @staticmethod
    def _build_qbar(q):
        steps, vocab_size, _ = q.shape
        qbar = torch.empty(
            (steps + 1, vocab_size, vocab_size), dtype=q.dtype, device=q.device
        )
        qbar[0] = torch.eye(vocab_size, dtype=q.dtype, device=q.device)
        for step in range(1, steps + 1):
            qbar[step] = qbar[step - 1] @ q[step - 1]
            qbar[step] /= qbar[step].sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return qbar

    def matrices_for_position(self, position):
        return self._position_q[position], self._position_qbar[position]
