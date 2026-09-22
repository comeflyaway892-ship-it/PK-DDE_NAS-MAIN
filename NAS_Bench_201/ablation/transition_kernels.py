"""Categorical corruption kernels from Eqs. (18)--(22), plus dynamic marginals."""

import torch

from prior_exp.prior_kernels import build_prior_knowledge_kernel

NUM_EDGES = 6
NUM_OPS = 5
MASK = 5  # Extra latent category; never an alias for the NB201 'none' operation.
OP_NAMES = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
KERNELS = {
    "uniform": ("Uniform", "Nominal or ordinal"),
    "absorbing": ("Absorbing / mask", "Mask-aware encoding"),
    "gaussian": ("Discrete Gaussian", "Ordered decisions"),
    "distance": ("Distance-aware", "Ordered decisions"),
    "fixed_marginal": ("Fixed marginal", "Prespecified frequencies"),
    "dynamic_marginal": ("Dynamic marginal", "Frequencies learned from high-fitness architectures"),
}


def per_edge_scales(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if values.numel() not in (1, NUM_EDGES):
        raise ValueError("Expected one scale or six per-edge scales")
    if not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Kernel scales must be finite and positive")
    return values.repeat(NUM_EDGES) if values.numel() == 1 else values


def marginal_probabilities(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() not in (NUM_OPS, NUM_EDGES * NUM_OPS):
        raise ValueError("Fixed marginal requires 5 values, or 6 rows of 5 values")
    values = values.reshape(-1, NUM_OPS)
    if not torch.isfinite(values).all() or (values < 0).any() or (values.sum(-1) <= 0).any():
        raise ValueError("Marginal probabilities must be finite, nonnegative, with positive row sums")
    values = values / values.sum(-1, keepdim=True)
    return values.repeat(NUM_EDGES, 1) if len(values) == 1 else values


def build_kernel(kind, *, gaussian_sigma=(1.0,), distance_tau=(1.0,),
                 fixed_probs=(0.1, 0.1, 0.3, 0.3, 0.2), knowledge=None, prior_strength=0.99):
    if kind == "uniform":
        matrix = torch.full((NUM_EDGES, NUM_OPS, NUM_OPS), 1.0 / NUM_OPS)
    elif kind == "absorbing":
        matrix = torch.zeros((NUM_EDGES, NUM_OPS + 1, NUM_OPS + 1))
        matrix[:, :, MASK] = 1.0
    elif kind in ("gaussian", "distance"):
        tokens = torch.arange(NUM_OPS, dtype=torch.float32)
        delta = (tokens[:, None] - tokens[None, :]).abs().unsqueeze(0)
        if kind == "gaussian":
            scales = per_edge_scales(gaussian_sigma).view(NUM_EDGES, 1, 1)
            logits = -delta.square() / (2 * scales.square())
        else:
            scales = per_edge_scales(distance_tau).view(NUM_EDGES, 1, 1)
            logits = -delta / scales
        matrix = torch.softmax(logits, dim=-1)
    elif kind == "fixed_marginal":
        rho = marginal_probabilities(fixed_probs)
        matrix = rho[:, None, :].repeat(1, NUM_OPS, 1)
    elif kind == "dynamic_marginal":
        if not knowledge:
            raise ValueError("Dynamic marginal requires evaluated top-k knowledge")
        pool = torch.cat(knowledge)
        if pool.ndim != 2 or pool.shape[1] != NUM_EDGES or not ((pool >= 0) & (pool < NUM_OPS)).all():
            raise ValueError("Knowledge must contain only clean NAS-Bench-201 operations")
        matrix = build_prior_knowledge_kernel(
            NUM_EDGES, NUM_OPS, pool, torch.ones(len(pool)),
            topk=len(pool), lumda=prior_strength,
        )
    else:
        raise ValueError(f"Unknown kernel {kind}")
    if not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Kernel contains invalid transition mass")
    if not torch.allclose(matrix.sum(-1), torch.ones(matrix.shape[:-1]), atol=1e-6):
        raise ValueError("Kernel rows must sum to one")
    return matrix
