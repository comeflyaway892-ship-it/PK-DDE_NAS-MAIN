import math
import torch

class D3PMUniformMatrixScheduler:
    """
    Build D3PM transition matrices with:
      - uniform kernel: Q_t = (1-β_t) I + β_t * U,  U_ij = 1/K
      - schedule on alpha_bar: linear or cosine

    Indexing convention:
      - t in [1..T] are diffusion steps
      - beta[t-1] corresponds to step t
      - Q[t-1] is the KxK matrix for step t
      - alpha_bar has length T+1 for t=0..T
      - Qbar[t] is cumulative transition x0->xt, length T+1 with Qbar[0]=I
    """

    def __init__(
        self,
        num_steps: int,
        vocab_size: int,
        eps: float = 1e-5,
        schedule: str = "cosine",   # <- 新增： "linear" or "cosine"
        cosine_s: float = 0.008,    # <- 新增：cosine 常用 s
        token_mask: torch.Tensor = None,
    ):
        self.T = int(num_steps)
        self.K = int(vocab_size)
        self.eps = float(eps)
        self.schedule = schedule
        self.token_mask = None if token_mask is None else token_mask.bool()

        self.alpha_bar = self._build_alpha_bar(self.T, self.eps, schedule, cosine_s)  # [T+1]
        self.beta = self._alpha_bar_to_beta(self.alpha_bar)                           # [T]
        self.Q = self._build_Q_matrices(self.beta, self.K, self.token_mask)             # [T,K,K] or [T,D,K,K]
        self.Qbar = self._build_Qbar_matrices(self.Q)                                  # [T+1,K,K]

    def _build_alpha_bar(self, T: int, eps: float, schedule: str, cosine_s: float) -> torch.Tensor:
        schedule = schedule.lower()
        if schedule in ("linear", "uniform"):
            return self._build_alpha_bar_linear(T, eps)
        elif schedule in ("cosine", "cos"):
            return self._build_alpha_bar_cosine(T, eps, cosine_s)
        else:
            raise ValueError(f"Unknown schedule='{schedule}'. Use 'linear' or 'cosine'.")

    def _build_alpha_bar_linear(self, T: int, eps: float) -> torch.Tensor:
        # 你原来的逻辑：从 1 线性降到 eps
        t = torch.arange(0, T + 1, dtype=torch.float32)
        alpha_bar = 1.0 - (t / T) * (1.0 - float(eps))
        return alpha_bar.clamp(min=float(eps), max=1.0)

    def _build_alpha_bar_cosine(self, T: int, eps: float, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule on alpha_bar (cumulative):
          alpha_bar(t) = cos(((t/T) + s)/(1+s) * pi/2)^2 / cos(s/(1+s)*pi/2)^2
        Then clamp to [eps, 1].
        """
        t = torch.arange(0, T + 1, dtype=torch.float32) / T  # [0..1]
        factor = (t + s) / (1.0 + s)
        f = torch.cos(factor * (math.pi / 2)) ** 2
        f0 = torch.cos(torch.tensor(s / (1.0 + s)) * (math.pi / 2)) ** 2
        alpha_bar = f / f0
        # 数值安全 + 保证非增趋势（避免浮点抖动）
        alpha_bar = alpha_bar.clamp(min=float(eps), max=1.0)
        alpha_bar[0] = 1.0
        # 可选：强制单调不增（更稳）
        alpha_bar = torch.minimum(alpha_bar, torch.cummin(alpha_bar, dim=0)[0])
        return alpha_bar

    def _alpha_bar_to_beta(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        prev = alpha_bar[:-1]
        curr = alpha_bar[1:]
        beta = 1.0 - (curr / prev)
        return beta.clamp(min=0.0, max=1.0 - 1e-8)

    def _build_Q_matrices(self, beta: torch.Tensor, K: int, token_mask: torch.Tensor = None) -> torch.Tensor:
        T = beta.shape[0]
        I = torch.eye(K, dtype=beta.dtype, device=beta.device)
        b = beta.view(T, 1, 1)
        if token_mask is None:
            U = torch.full((K, K), 1.0 / K, dtype=beta.dtype, device=beta.device)
            Q = (1.0 - b) * I + b * U
            Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            return Q

        token_mask = token_mask.to(device=beta.device, dtype=beta.dtype)
        D = token_mask.shape[0]
        U = token_mask[:, None, :].repeat(1, K, 1)
        U = U / U.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        Q = (1.0 - b[:, None, :, :]) * I.view(1, 1, K, K) + b[:, None, :, :] * U.view(1, D, K, K)
        Q = Q * token_mask.view(1, D, 1, K)
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        return Q

    def _build_Qbar_matrices(self, Q: torch.Tensor) -> torch.Tensor:
        if Q.dim() == 3:
            T, K, _ = Q.shape
            Qbar = torch.empty((T + 1, K, K), dtype=Q.dtype, device=Q.device)
            Qbar[0] = torch.eye(K, dtype=Q.dtype, device=Q.device)
            running = Qbar[0]
            for t in range(1, T + 1):
                running = running @ Q[t - 1]
                running = running / running.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                Qbar[t] = running
            return Qbar

        T, D, K, _ = Q.shape
        Qbar = torch.empty((T + 1, D, K, K), dtype=Q.dtype, device=Q.device)
        Qbar[0] = torch.eye(K, dtype=Q.dtype, device=Q.device).view(1, K, K).repeat(D, 1, 1)
        running = Qbar[0]
        for t in range(1, T + 1):
            running = torch.matmul(running, Q[t - 1])
            running = running / running.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            Qbar[t] = running
        return Qbar

    def get_Q(self, t: int) -> torch.Tensor:
        if not (1 <= t <= self.T):
            raise ValueError(f"t must be in [1..{self.T}]")
        return self.Q[t - 1]

    def get_Q_bar(self, t: int) -> torch.Tensor:
        if not (0 <= t <= self.T):
            raise ValueError(f"t must be in [0..{self.T}]")
        return self.Qbar[t]
