"""Position-aware prior kernels for the 22-token NB301 encoding."""

import torch

from config.config import nb301_vocab_sizes


def _uniform(size):
    return torch.full((size,), 1.0 / size, dtype=torch.float32)


def _kernel_from_probs(probs, size):
    probs = torch.as_tensor(probs, dtype=torch.float32).flatten()
    if probs.numel() != size or torch.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("invalid prior probabilities for vocabulary size {}".format(size))
    probs = probs / probs.sum()
    return probs.view(1, size).repeat(size, 1)


def build_structural_prior_kernels(tau=1.0, shuffle_seed=None):
    """Return one distance prior matrix for each heterogeneous token position."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    kernels = []
    generator = torch.Generator().manual_seed(int(shuffle_seed)) if shuffle_seed is not None else None
    for size in nb301_vocab_sizes:
        ids = torch.arange(size, dtype=torch.float32)
        distance = (ids[:, None] - ids[None, :]).abs()
        kernel = torch.softmax(-distance / float(tau), dim=-1)
        if generator is not None:
            permutation = torch.randperm(size, generator=generator)
            kernel = kernel[permutation][:, permutation]
        kernels.append(kernel)
    return kernels


def build_prior_knowledge_kernels(
    initial_population, initial_scores, topk=10, lumda=1.0, groups=None
):
    """Build marginal priors from the top-k observed NB301 architectures."""
    if not 0.0 <= float(lumda) <= 1.0:
        raise ValueError("lumda must be in [0, 1]")
    x = torch.as_tensor(initial_population).detach().cpu().long()
    scores = torch.as_tensor(initial_scores).detach().cpu().float().reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(nb301_vocab_sizes) or scores.numel() != x.shape[0]:
        raise ValueError("knowledge population/scores have incompatible shapes")
    k = min(int(topk), x.shape[0])
    if k <= 0:
        raise ValueError("topk must be positive")
    top = x[torch.topk(scores, k=k).indices]
    # Pool corresponding operation tokens across Normal/Reduction cells,
    # while keeping p3, p4 and p5 topology tokens separate because their
    # vocabularies (3, 6 and 10) have different meanings.
    if groups is None:
        operation_offsets = (0, 1, 3, 4, 6, 7, 9, 10)
        groups = [
            [offset, offset + 11] for offset in operation_offsets
        ] + [[2, 13], [5, 16], [8, 19]]

    kernels = [None] * len(nb301_vocab_sizes)
    for group in groups:
        positions = [int(position) for position in group]
        size = nb301_vocab_sizes[positions[0]]
        if any(nb301_vocab_sizes[position] != size for position in positions):
            raise ValueError("a prior group may only contain equal-size vocabularies")
        counts = torch.bincount(top[:, positions].reshape(-1), minlength=size).float()
        freq = counts / counts.sum().clamp_min(1e-30)
        probs = float(lumda) * freq + (1.0 - float(lumda)) * _uniform(size)
        for position in positions:
            kernels[position] = _kernel_from_probs(probs, size)
    for position, size in enumerate(nb301_vocab_sizes):
        if kernels[position] is None:
            kernels[position] = _kernel_from_probs(_uniform(size), size)
    return kernels


def build_prior_kernels(prior_type="structural", **kwargs):
    name = str(prior_type).lower()
    if name in ("structural", "structure", "distance"):
        return build_structural_prior_kernels(tau=kwargs.get("tau", 1.0))
    if name in ("shuffled_structural", "shuffle", "shuffled", "random"):
        return build_structural_prior_kernels(
            tau=kwargs.get("tau", 1.0), shuffle_seed=kwargs.get("shuffle_seed", 0)
        )
    if name in ("prior_knowledge", "prior_konwleage", "knowledge"):
        return build_prior_knowledge_kernels(
            kwargs["initial_population"], kwargs["initial_scores"],
            topk=kwargs.get("topk", 10), lumda=kwargs.get("lumda", 1.0)
        )
    if name in ("uniform", "none", "baseline"):
        return None
    raise ValueError("Unknown prior_type={!r}".format(prior_type))
