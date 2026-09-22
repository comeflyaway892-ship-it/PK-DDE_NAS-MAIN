"""Bayesian D3PM predictor for heterogeneous NAS-Bench-301 tokens."""

import torch
import torch.nn.functional as F


def d3pm_step(xt, x0, scheduler, t, eps=1e-20):
    """Sample x_(t-1) independently with each position's valid vocabulary."""
    xt = xt.long()
    x0 = x0.long()
    result = torch.empty_like(xt)

    for position, vocab_size in enumerate(scheduler.vocab_sizes):
        q, qbar = scheduler.matrices_for_position(position)
        q_t = q[t - 1].to(xt.device)
        qbar_tm1 = qbar[t - 1].to(xt.device)

        current = xt[:, position]
        origin = x0[:, position]
        likelihood = q_t[:, current].transpose(0, 1).contiguous()
        prior = qbar_tm1[origin]
        # Multiplication can underflow for the late reverse steps when a
        # position has a non-uniform prior (especially vocab size 10).  Work
        # in log space, then normalize after subtracting the row maximum.
        log_posterior = torch.log(likelihood.clamp_min(eps)) + torch.log(
            prior.clamp_min(eps)
        )
        posterior = torch.softmax(log_posterior, dim=-1)
        posterior = posterior.clamp_min(eps)
        posterior /= posterior.sum(dim=-1, keepdim=True).clamp_min(eps)
        result[:, position] = torch.multinomial(posterior, 1).squeeze(1)

        if torch.any(result[:, position] >= vocab_size):
            raise RuntimeError("D3PM sampled an invalid token category")
    return result


class BayesianEstimator:
    def __init__(self, x, fitness, scheduler, t, eps=1e-12):
        self.x = x.long()
        self.fitness = fitness.float()
        self.scheduler = scheduler
        self.t = int(t)
        self.eps = float(eps)

    def estimate(self, sigma=1.0, temperature=0.4):
        x_t = self.x
        fitness = self.fitness
        population, dimensions = x_t.shape
        if dimensions != len(self.scheduler.vocab_sizes):
            raise ValueError("Architecture width does not match scheduler vocabularies")

        temperature = max(float(temperature), self.eps)
        sigma = max(float(sigma), self.eps)
        centered = fitness - fitness.mean()
        base = torch.softmax(centered / temperature, dim=0).clamp_min(self.eps)
        base /= base.sum().clamp_min(self.eps)

        # Diffusion-consistent similarity between every pair of architectures.
        log_kernel = torch.zeros(
            (population, population), device=x_t.device, dtype=torch.float32
        )
        for position in range(dimensions):
            _, qbar = self.scheduler.matrices_for_position(position)
            transition = qbar[self.t].to(x_t.device).clamp_min(self.eps)
            categories = x_t[:, position]
            probabilities = transition[
                categories.view(1, population), categories.view(population, 1)
            ]
            log_kernel += torch.log(probabilities.clamp_min(self.eps))

        kernel = torch.exp((log_kernel / dimensions) / sigma).clamp_min(self.eps)
        weights = kernel * base.view(1, population)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(self.eps)

        # Estimate and sample x0 one position at a time because K is variable.
        x0_hat = torch.empty_like(x_t)
        for position, vocab_size in enumerate(self.scheduler.vocab_sizes):
            one_hot = F.one_hot(
                x_t[:, position], num_classes=vocab_size
            ).float()
            p0 = weights @ one_hot
            p0 = p0.clamp_min(self.eps)
            p0 /= p0.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            x0_hat[:, position] = torch.multinomial(p0, 1).squeeze(1)
        return x0_hat


class BayesianGenerator:
    def __init__(
        self,
        x,
        fitness,
        scheduler,
        t,
        estimator_eps=1e-12,
        sigma=1.0,
        temperature=0.4,
    ):
        self.x = x.long()
        self.scheduler = scheduler
        self.t = int(t)
        self.sigma = float(sigma)
        self.temperature = float(temperature)
        self.estimator = BayesianEstimator(
            x=self.x,
            fitness=fitness,
            scheduler=scheduler,
            t=t,
            eps=estimator_eps,
        )

    def generate(self):
        x0_est = self.estimator.estimate(
            sigma=self.sigma, temperature=self.temperature
        )
        return d3pm_step(
            xt=self.x,
            x0=x0_est,
            scheduler=self.scheduler,
            t=self.t,
        )
