import torch
import torch.nn.functional as F


def _normalise_posterior(post, like, xt_flat, num_classes, eps):
    post_mass = post.sum(dim=-1, keepdim=True)
    normalised = post / post_mass.clamp_min(eps)

    like_mass = like.sum(dim=-1, keepdim=True)
    identity = F.one_hot(xt_flat, num_classes=num_classes).to(dtype=like.dtype)
    fallback = torch.where(like_mass > eps, like / like_mass.clamp_min(eps), identity)
    return torch.where(post_mass > eps, normalised, fallback)


def d3pm_step(xt, x0, q, qbar, t, eps=1e-20):
    xt = xt.long()
    x0 = x0.long()
    N, D = xt.shape
    K = q.shape[-1]
    q_t = q[t - 1]
    qbar_tm1 = qbar[t - 1]
    xt_flat = xt.reshape(-1).clamp(0, K - 1)
    x0_flat = x0.reshape(-1).clamp(0, K - 1)
    d_idx = torch.arange(D, device=xt.device).repeat(N)
    like = q_t[d_idx, :, xt_flat]
    prior = qbar_tm1[d_idx, x0_flat, :]
    post = like * prior
    post = _normalise_posterior(post, like, xt_flat, K, eps)
    return torch.multinomial(post, num_samples=1).squeeze(-1).view_as(xt)


class BayesianEstimator:
    def __init__(self, x, fitness, q, qbar, t, eps=1e-12):
        self.x = x
        self.fitness = fitness
        self.q = q
        self.qbar = qbar
        self.t = t
        self.eps = float(eps)

    def estimate(self, eps=1e-20, sigma=1.0, temperature=0.4):
        x_t = self.x.long()
        fitness = self.fitness.float()
        t = int(self.t)
        eps = getattr(self, "eps", eps)
        N, D = x_t.shape
        K = self.qbar.shape[-1]
        qbar_t = self.qbar[t].clamp_min(eps)
        temp = max(float(temperature), eps)
        base = torch.softmax((fitness - fitness.mean()) / temp, dim=0).clamp_min(eps)
        base = base / (base.sum() + eps)
        log_kernel = torch.zeros((N, N), device=x_t.device, dtype=torch.float32)
        for d in range(D):
            a = x_t[:, d].clamp(0, K - 1)
            b = x_t[:, d].clamp(0, K - 1)
            M = qbar_t[d, a.view(1, N), b.view(N, 1)].clamp_min(eps)
            log_kernel += torch.log(M)
        kernel = torch.exp((log_kernel / D) / sigma).clamp_min(eps)
        W = kernel * base.view(1, N)
        W = W / (W.sum(dim=1, keepdim=True) + eps)
        onehot = F.one_hot(x_t.clamp(0, K - 1), num_classes=K).float()
        p0 = torch.einsum("ij,jdk->idk", W, onehot).clamp_min(eps)
        p0 = p0 / (p0.sum(dim=-1, keepdim=True) + eps)
        return torch.multinomial(p0.reshape(N * D, K), num_samples=1).squeeze(1).view(N, D)

    def __call__(self, eps=1e-20, sigma=1.0, temperature=0.4):
        return self.estimate(eps=eps, sigma=sigma, temperature=temperature)


class BayesianGenerator:
    def __init__(self, x, fitness, q, qbar, t, *, estimator_eps=1e-12, sigma=1.0, temperature=0.4):
        self.x = x
        self.fitness = fitness
        self.q = q
        self.qbar = qbar
        self.t = t
        self.sigma = float(sigma)
        self.temperature = float(temperature)
        self.estimator = BayesianEstimator(x, fitness, q, qbar, t=t, eps=float(estimator_eps))

    def generate(self):
        x0_est = self.estimator(sigma=self.sigma, temperature=self.temperature)
        return d3pm_step(xt=self.x, x0=x0_est, q=self.q, qbar=self.qbar, t=self.t)
