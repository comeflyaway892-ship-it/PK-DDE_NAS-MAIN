import math
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, Sequence[float]]


class D3PMPresetPriorMatrixScheduler:
    """
    DiGress-style D3PM transition scheduler with a preset prior distribution.

    Compared with the uniform kernel
        Q_t = (1 - beta_t) I + beta_t * U,   U_ij = 1 / K,
    this class uses a preset prior pi:
        Q_t = (1 - beta_t) I + beta_t * P,
    where
        P_ij = pi_j.

    That means each row of P is the same prior distribution pi, so the forward
    process gradually mixes every token towards the preset prior instead of the
    uniform distribution.

    This is the same structural idea used by DiGress's marginal transition:
        q_t = (1 - beta_t) I + beta_t * prior_matrix.

    Indexing convention:
      - t in [1..T] are diffusion steps
      - beta[t-1] corresponds to step t
      - Q[t-1] is the KxK matrix for step t
      - alpha_bar has length T+1 for t=0..T
      - Qbar[t] is cumulative transition x0->xt, length T+1 with Qbar[0]=I

    Args:
        num_steps: Number of diffusion steps T.
        vocab_size: Number of categories K.
        prior_probs: Preset prior distribution of shape [K]. It will be
            normalized automatically.
        eps: Minimum alpha_bar value for numerical stability.
        schedule: "linear" or "cosine".
        cosine_s: Cosine schedule offset.
        dtype: Floating dtype used to build the matrices.
        device: Optional device.
    """

    def __init__(
        self,
        num_steps: int,
        vocab_size: int,
        prior_probs: TensorLike,
        eps: float = 1e-5,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.T = int(num_steps)
        self.K = int(vocab_size)
        self.eps = float(eps)
        self.schedule = str(schedule)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None

        self.prior = self._normalize_prior(prior_probs, self.K, dtype=self.dtype, device=self.device)  # [K]
        self.P = self._build_prior_matrix(self.prior)                                                   # [K,K]

        self.alpha_bar = self._build_alpha_bar(self.T, self.eps, self.schedule, cosine_s)              # [T+1]
        self.beta = self._alpha_bar_to_beta(self.alpha_bar)                                             # [T]
        self.Q = self._build_Q_matrices(self.beta, self.K, self.P)                                      # [T,K,K]
        self.Qbar = self._build_Qbar_matrices(self.alpha_bar, self.K, self.P)                           # [T+1,K,K]

    @staticmethod
    def _normalize_prior(
        prior_probs: TensorLike,
        K: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
    ) -> torch.Tensor:
        prior = torch.as_tensor(prior_probs, dtype=dtype, device=device).flatten()
        if prior.numel() != K:
            raise ValueError(f"prior_probs must have length {K}, got {prior.numel()}")
        if torch.any(prior < 0):
            raise ValueError("prior_probs must be non-negative")

        total = prior.sum()
        if not torch.isfinite(total) or total.item() <= 0:
            raise ValueError("prior_probs must have a positive finite sum")

        prior = prior / total
        prior = prior.clamp_min(0.0)
        prior = prior / prior.sum().clamp_min(1e-30)
        return prior

    def _build_prior_matrix(self, prior: torch.Tensor) -> torch.Tensor:
        # Each row equals the preset prior pi.
        return prior.unsqueeze(0).expand(self.K, -1).contiguous()

    def _build_alpha_bar(self, T: int, eps: float, schedule: str, cosine_s: float) -> torch.Tensor:
        schedule = schedule.lower()
        if schedule in ("linear", "uniform"):
            return self._build_alpha_bar_linear(T, eps)
        if schedule in ("cosine", "cos"):
            return self._build_alpha_bar_cosine(T, eps, cosine_s)
        raise ValueError(f"Unknown schedule='{schedule}'. Use 'linear' or 'cosine'.")

    def _build_alpha_bar_linear(self, T: int, eps: float) -> torch.Tensor:
        t = torch.arange(0, T + 1, dtype=self.dtype, device=self.device)
        alpha_bar = 1.0 - (t / T) * (1.0 - float(eps))
        alpha_bar = alpha_bar.clamp(min=float(eps), max=1.0)
        alpha_bar[0] = 1.0
        return alpha_bar

    def _build_alpha_bar_cosine(self, T: int, eps: float, s: float = 0.008) -> torch.Tensor:
        t = torch.arange(0, T + 1, dtype=self.dtype, device=self.device) / T
        factor = (t + s) / (1.0 + s)
        f = torch.cos(factor * (math.pi / 2)) ** 2
        f0 = torch.cos(torch.tensor(s / (1.0 + s), dtype=self.dtype, device=t.device) * (math.pi / 2)) ** 2
        alpha_bar = f / f0
        alpha_bar = alpha_bar.clamp(min=float(eps), max=1.0)
        alpha_bar[0] = 1.0
        alpha_bar = torch.minimum(alpha_bar, torch.cummin(alpha_bar, dim=0)[0])
        return alpha_bar

    def _alpha_bar_to_beta(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        prev = alpha_bar[:-1]
        curr = alpha_bar[1:]
        beta = 1.0 - (curr / prev)
        return beta.clamp(min=0.0, max=1.0 - 1e-8)

    def _build_Q_matrices(self, beta: torch.Tensor, K: int, P: torch.Tensor) -> torch.Tensor:
        T = beta.shape[0]
        I = torch.eye(K, dtype=beta.dtype, device=beta.device)
        b = beta.view(T, 1, 1)
        Q = (1.0 - b) * I + b * P.unsqueeze(0)
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return Q

    def _build_Qbar_matrices(self, alpha_bar: torch.Tensor, K: int, P: torch.Tensor) -> torch.Tensor:
        # Because P is idempotent (P @ P = P) and I @ P = P,
        # the cumulative transition has a closed form:
        #   Qbar_t = alpha_bar_t * I + (1 - alpha_bar_t) * P
        T = alpha_bar.shape[0] - 1
        I = torch.eye(K, dtype=alpha_bar.dtype, device=alpha_bar.device)
        a = alpha_bar.view(T + 1, 1, 1)
        Qbar = a * I.unsqueeze(0) + (1.0 - a) * P.unsqueeze(0)
        Qbar = Qbar / Qbar.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return Qbar

    def get_Q(self, t: int) -> torch.Tensor:
        if not (1 <= t <= self.T):
            raise ValueError(f"t must be in [1..{self.T}]")
        return self.Q[t - 1]

    def get_Q_bar(self, t: int) -> torch.Tensor:
        if not (0 <= t <= self.T):
            raise ValueError(f"t must be in [0..{self.T}]")
        return self.Qbar[t]

    def q_probs(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Return q(x_t | x_0) as categorical probabilities.

        Args:
            x0: Long tensor of token ids with arbitrary shape [...].
            t: diffusion step in [0..T].

        Returns:
            probs: Tensor with shape [..., K].
        """
        if x0.dtype != torch.long:
            raise TypeError("x0 must be a torch.long tensor of token ids")
        Qbar_t = self.get_Q_bar(t).to(device=x0.device)
        x0_onehot = F.one_hot(x0, num_classes=self.K).to(Qbar_t.dtype)
        probs = x0_onehot @ Qbar_t
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return probs

    @torch.no_grad()
    def sample_xt(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0).
        """
        probs = self.q_probs(x0, t)
        flat = probs.reshape(-1, self.K)
        xt = torch.multinomial(flat, num_samples=1).reshape(x0.shape)
        return xt

    @torch.no_grad()
    def sample_stationary(self, shape, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Sample from the terminal prior distribution pi.
        In the DiGress-style preset-prior setting, x_T is approximately drawn
        from this distribution when alpha_bar_T is very small.
        """
        device = torch.device(device) if device is not None else self.prior.device
        flat_size = int(torch.tensor(shape).prod().item()) if not isinstance(shape, int) else int(shape)
        prior = self.prior.to(device)
        samples = torch.multinomial(prior.expand(flat_size, -1), num_samples=1).reshape(shape)
        return samples

    def to(self, device: Union[str, torch.device]) -> "D3PMPresetPriorMatrixScheduler":
        device = torch.device(device)
        self.device = device
        self.prior = self.prior.to(device)
        self.P = self.P.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.beta = self.beta.to(device)
        self.Q = self.Q.to(device)
        self.Qbar = self.Qbar.to(device)
        return self

    @classmethod
    def from_counts(
        cls,
        num_steps: int,
        counts: TensorLike,
        eps: float = 1e-5,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "D3PMPresetPriorMatrixScheduler":
        counts_tensor = torch.as_tensor(counts, dtype=dtype)
        if counts_tensor.ndim != 1:
            raise ValueError("counts must be a 1D tensor/list")
        return cls(
            num_steps=num_steps,
            vocab_size=int(counts_tensor.numel()),
            prior_probs=counts_tensor,
            eps=eps,
            schedule=schedule,
            cosine_s=cosine_s,
            dtype=dtype,
            device=device,
        )


if __name__ == "__main__":
    # Example: a 5-class D3PM with a manually preset prior.
    scheduler = D3PMPresetPriorMatrixScheduler(
        num_steps=100,
        vocab_size=5,
        prior_probs=[0.50, 0.20, 0.15, 0.10, 0.05],
        schedule="cosine",
    )

    print("prior:", scheduler.prior)
    print("Q_1 shape:", scheduler.get_Q(1).shape)
    print("Qbar_T row 0:", scheduler.get_Q_bar(scheduler.T)[0])

    x0 = torch.tensor([0, 1, 4, 2], dtype=torch.long)
    xt = scheduler.sample_xt(x0, t=50)
    print("x0:", x0)
    print("xt:", xt)
