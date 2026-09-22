# from d3pm import D3PMUniformMatrixScheduler
import torch.nn.functional as F

import torch

def d3pm_step(xt, x0, q, qbar, t, eps=1e-20):

    xt_flat = xt.reshape(-1)  # [N]
    x0_flat = x0.reshape(-1)  # [N]
    N = xt_flat.numel()
    q_t = q[t-1]     # q(x_t | x_{t-1})

    qbar_tm1 = qbar[t-1]  # q(x_{t-1} | x0)
    like = q_t[:, xt_flat].transpose(0, 1).contiguous()  # [N, K]
    prior = qbar_tm1[x0_flat]  # [N, K]
    post_unnorm = like * prior  # [N, K]

    # 归一化
    denom = post_unnorm.sum(dim=-1, keepdim=True).clamp_min(eps)  # [N,1]
    post = post_unnorm / denom  # [N,K]
    x_tm1_flat = torch.multinomial(post, num_samples=1).squeeze(-1)  # [N]

    # reshape 回原形状
    x_tm1 = x_tm1_flat.view_as(xt)
    return x_tm1


class BayesianEstimator:
    def __init__(self, x: torch.Tensor, fitness: torch.Tensor, q: torch.Tensor, qbar: torch.Tensor, t,eps: float = 1e-12):
        """
        x:      (N, D) int tensor, candidate x0 architectures (each row)
        fitness:(N,)   tensor
        q:      (T, K, K)   one-step transition (not used in estimate for now)
        qbar:   (T+1, K, K) t-step marginal transition, qbar[t,i,j] = q(x_t=j | x0=i)
        """
        self.x = x
        self.t= t
        self.fitness = fitness
        self.q = q
        self.qbar = qbar
        self.eps = float(eps)

    def estimate(self, eps: float = 1e-20, sigma: float = 1.0, temperature: float = 0.4):
        x_t = self.x.long()  # (N, D)
        fitness = self.fitness.float()  # (N,)
        t = int(self.t)
        eps = getattr(self, "eps", eps)
        N, D = x_t.shape
        K = self.qbar.shape[-1]
        qbar_t = self.qbar[t].clamp_min(eps)  # (K, K),  qbar_t[i,j]=P(x_t=j|x0=i)
        # ---------- 1) fitness base weight: temperature softmax ----------
        temp = max(float(temperature), eps)
        f_center = fitness - fitness.mean()
        base = torch.softmax(f_center / temp, dim=0).clamp_min(eps)  # (N,)
        base = base / (base.sum() + eps)  # 再保一下 sum=1
        # -------- 2) diffusion-consistent kernel: κ_ij = Π_d qbar_t[x_jd, x_id] --------
        log_kernel = torch.zeros((N, N), device=x_t.device, dtype=torch.float32)
        for d in range(D):
            a = x_t[:, d].clamp(0, K - 1)  # (N,) 作为 "from"：x_jd
            b = x_t[:, d].clamp(0, K - 1)  # (N,) 作为 "to"  ：x_id
            M = qbar_t[a.view(1, N), b.view(N, 1)].clamp_min(eps)  # (N, N)
            log_kernel += torch.log(M)

        log_kernel = log_kernel / D
        kernel = torch.exp(log_kernel/sigma).clamp_min(eps)  # (N, N)
        # ---------- 3) 组合得到个体化权重矩阵 W_{ij} ----------
        W = kernel * base.view(1, N)  # (N, N)
        W = W / (W.sum(dim=1, keepdim=True) + eps)  # 每行归一化：sum_j W_ij = 1
        # ---------- 4) 通过个体化权重获得x_0的分布 p_0^{(i)}(d,k) ----------
        onehot = F.one_hot(x_t.clamp(0, K - 1), num_classes=K).float()  # (N, D, K)
        p0 = torch.einsum("ij,jdk->idk", W, onehot).clamp_min(eps)
        p0 = p0 / (p0.sum(dim=-1, keepdim=True) + eps)
        # ---------- 5) 逐 token 采样得到 x0_hat (N,D) ----------
        p0_flat = p0.reshape(N * D, K)
        x0_flat = torch.multinomial(p0_flat, num_samples=1).squeeze(1)
        x0_hat = x0_flat.view(N, D)
        return x0_hat

    def __call__(self, eps: float = 1e-20, sigma: float = 5.0, temperature: float = 0.4) -> torch.Tensor:
        # 让 BayesianGenerator 可以把关键超参（temperature/sigma/eps）一路传下来
        return self.estimate(eps=eps, sigma=sigma, temperature=temperature)
 


class BayesianGenerator:
    def __init__(
        self,
        x,
        fitness,
        q,
        qbar,
        t,
        *,
        estimator_eps: float = 1e-12,
        sigma: float = 5.0,
        temperature: float = 0.4,
    ):
        self.x = x
        self.fitness = fitness
        self.t=t
        self.q = q
        self.qbar = qbar
        self.sigma = float(sigma)
        self.temperature = float(temperature)
        self.estimator = BayesianEstimator(
            self.x,
            self.fitness,
            self.q,
            self.qbar,
            t=self.t,
            eps=float(estimator_eps),
        )

    def generate(self):
        # 将 predictor 超参显式传入，避免依赖 estimate() 默认值
        x0_est = self.estimator(sigma=self.sigma, temperature=self.temperature)
        x_next = d3pm_step(xt=self.x,x0=x0_est,q=self.q,qbar=self.qbar,t=self.t)

        # log_two_tensors_txt(x0_est, x_next, tag="d3pm")

        return x_next



def mean_entropy(p: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    p: (..., M) 概率分布张量，最后一维是类别维
    返回: 标量张量，平均熵
    """
    p = p.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)   # (...,)
    return H.mean()

@torch.no_grad()
def debug_print_step_entropy(pi: torch.Tensor, t: int, every: int = 1, eps: float = 1e-20) -> None:
    """
    打印当前步 responsibilities pi 的平均熵（以及可选的有效邻居数）
    pi: (N, N) 每行是一个分布
    """
    if (t % every) != 0:
        return
    H = mean_entropy(pi, eps=eps).item()
    # 也可以顺便打印 perplexity（更直观）
    ppl = float(torch.exp(torch.tensor(H)))
    print(f"[t={t:04d}] mean_entropy(pi)={H:.6f}, perplexity≈{ppl:.3f}")


import os
import time
import torch
from pathlib import Path
from itertools import count

_LOG_COUNTER = count(0)

def log_two_tensors_txt(x0_est: torch.Tensor,
                        x_next: torch.Tensor,
                        *,
                        tag: str = "d3pm",
                        log_dir: str = "log",
                        max_rows: int = 50,
                        max_cols: int = 50) -> str:
    """
    在当前运行目录创建 log/，每次调用写一个新的 txt 文件（不覆盖）。
    只记录：shape/dtype + 左上角截取的内容（方便直接打开看）。
    返回写入的 txt 路径。
    """
    out_dir = Path.cwd() / log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    ms = int((time.time() % 1) * 1000)
    pid = os.getpid()
    idx = next(_LOG_COUNTER)

    path = out_dir / f"{tag}_{ts}-{ms:03d}_pid{pid}_{idx:06d}.txt"

    # 转到 CPU，避免 CUDA tensor 写入/打印问题
    x0 = x0_est.detach().to("cpu")
    xn = x_next.detach().to("cpu")

    r = min(max_rows, x0.shape[0])
    c = min(max_cols, x0.shape[1])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"file: {path.name}\n")
        f.write(f"x0_est: shape={tuple(x0.shape)}, dtype={x0.dtype}\n")
        f.write(f"x_next: shape={tuple(xn.shape)}, dtype={xn.dtype}\n\n")

        f.write(f"[x0_est top-left {r}x{c}]\n")
        f.write(str(x0[:r, :c]) + "\n\n")

        f.write(f"[x_next top-left {r}x{c}]\n")
        f.write(str(xn[:r, :c]) + "\n")

    return str(path)
