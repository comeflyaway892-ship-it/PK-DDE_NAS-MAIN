import torch

from config.config import num_depth_tokens, num_edges, vocab_size


DEPTH_TOKEN_TO_VALUE = {
    0: 2,
    1: 3,
    2: 4,
}

OP_TOKEN_TO_FEATURE = {
    0: (3, 3),
    1: (3, 4),
    2: (3, 6),
    3: (5, 3),
    4: (5, 4),
    5: (5, 6),
    6: (7, 3),
    7: (7, 4),
    8: (7, 6),
}


def _normalise_rows(matrix, eps=1e-30):
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)


def _masked_uniform(valid_tokens, k):
    probs = torch.zeros(k, dtype=torch.float32)
    probs[valid_tokens] = 1.0 / len(valid_tokens)
    return probs


def _distance_softmax_kernel(features, valid_tokens, k, tau):
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    kernel = torch.zeros(k, k, dtype=torch.float32)
    valid_tokens = list(valid_tokens)
    feature_tensor = torch.tensor([features[token] for token in valid_tokens], dtype=torch.float32)

    for row_idx, source_token in enumerate(valid_tokens):
        source = feature_tensor[row_idx : row_idx + 1]
        distance = (feature_tensor - source).abs().sum(dim=-1)
        probs = torch.softmax(-distance / float(tau), dim=-1)
        kernel[source_token, valid_tokens] = probs

    fallback = _masked_uniform(valid_tokens, k)
    invalid_tokens = [token for token in range(k) if token not in valid_tokens]
    if invalid_tokens:
        kernel[invalid_tokens] = fallback
    return _normalise_rows(kernel)


def _op_distance_softmax_kernel(valid_tokens, k, tau, kernel_weight, expand_weight):
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    kernel = torch.zeros(k, k, dtype=torch.float32)
    valid_tokens = list(valid_tokens)
    features = torch.tensor([OP_TOKEN_TO_FEATURE[token] for token in valid_tokens], dtype=torch.float32)
    weights = torch.tensor([float(kernel_weight), float(expand_weight)], dtype=torch.float32)

    for row_idx, source_token in enumerate(valid_tokens):
        source = features[row_idx : row_idx + 1]
        distance = ((features - source).abs() * weights).sum(dim=-1)
        probs = torch.softmax(-distance / float(tau), dim=-1)
        kernel[source_token, valid_tokens] = probs

    fallback = _masked_uniform(valid_tokens, k)
    invalid_tokens = [token for token in range(k) if token not in valid_tokens]
    if invalid_tokens:
        kernel[invalid_tokens] = fallback
    return _normalise_rows(kernel)


def build_structural_prior_kernel(
    *,
    depth_tau=1.0,
    op_tau=1.0,
    kernel_weight=1.0,
    expand_weight=1.0,
    depth_weight=1.0,
):
    """Build M_d(a,b) from MBv3 token structure, independent of target accuracy."""
    depth_tokens = sorted(DEPTH_TOKEN_TO_VALUE)
    op_tokens = sorted(OP_TOKEN_TO_FEATURE)

    depth_features = {
        token: (float(value) * float(depth_weight),)
        for token, value in DEPTH_TOKEN_TO_VALUE.items()
    }
    depth_kernel = _distance_softmax_kernel(
        depth_features,
        valid_tokens=depth_tokens,
        k=vocab_size,
        tau=depth_tau,
    )
    op_kernel = _op_distance_softmax_kernel(
        valid_tokens=op_tokens,
        k=vocab_size,
        tau=op_tau,
        kernel_weight=kernel_weight,
        expand_weight=expand_weight,
    )

    prior_kernel = torch.empty(num_edges, vocab_size, vocab_size, dtype=torch.float32)
    prior_kernel[:num_depth_tokens] = depth_kernel
    prior_kernel[num_depth_tokens:] = op_kernel
    return prior_kernel


def _shuffle_kernel(kernel, valid_tokens, seed):
    valid_tokens = list(valid_tokens)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(len(valid_tokens), generator=generator)

    shuffled = kernel.clone()
    valid_index = torch.tensor(valid_tokens, dtype=torch.long)
    block = kernel[valid_index][:, valid_index]
    shuffled[valid_index[:, None], valid_index[None, :]] = block[permutation][:, permutation]
    return _normalise_rows(shuffled)


def build_shuffled_structural_prior_kernel(
    *,
    shuffle_seed=0,
    shuffle_depth=False,
    depth_tau=1.0,
    op_tau=1.0,
    kernel_weight=1.0,
    expand_weight=1.0,
    depth_weight=1.0,
):
    """
    Build a negative-control prior by destroying token semantics.

    Operation-token kernels are permuted as P M P^T, preserving the row/column
    probability shape while breaking the mapping between MBv3 operations and
    their structural neighborhoods. Depth kernels are kept structural by
    default, because the proposed control is about operation semantics.
    """
    prior_kernel = build_structural_prior_kernel(
        depth_tau=depth_tau,
        op_tau=op_tau,
        kernel_weight=kernel_weight,
        expand_weight=expand_weight,
        depth_weight=depth_weight,
    )
    op_tokens = sorted(OP_TOKEN_TO_FEATURE)
    prior_kernel[num_depth_tokens:] = _shuffle_kernel(
        prior_kernel[num_depth_tokens],
        valid_tokens=op_tokens,
        seed=shuffle_seed,
    )

    if shuffle_depth:
        depth_tokens = sorted(DEPTH_TOKEN_TO_VALUE)
        prior_kernel[:num_depth_tokens] = _shuffle_kernel(
            prior_kernel[0],
            valid_tokens=depth_tokens,
            seed=int(shuffle_seed) + 1000003,
        )
    return prior_kernel


def build_marginal_prior_kernel(depth_prior=None, op_prior=None):
    """Build rank-one M_d = 1 pi_d^T kernels for preset categorical priors."""
    depth_tokens = sorted(DEPTH_TOKEN_TO_VALUE)
    op_tokens = sorted(OP_TOKEN_TO_FEATURE)

    depth_pi = _preset_distribution(depth_prior, depth_tokens, vocab_size, "depth_prior")
    op_pi = _preset_distribution(op_prior, op_tokens, vocab_size, "op_prior")

    prior_kernel = torch.empty(num_edges, vocab_size, vocab_size, dtype=torch.float32)
    prior_kernel[:num_depth_tokens] = depth_pi.view(1, 1, vocab_size).repeat(
        num_depth_tokens,
        vocab_size,
        1,
    )
    prior_kernel[num_depth_tokens:] = op_pi.view(1, 1, vocab_size).repeat(
        num_edges - num_depth_tokens,
        vocab_size,
        1,
    )
    return prior_kernel


def build_prior_knowledge_kernel(
    initial_population,
    initial_scores,
    *,
    topk=10,
    lumda=0.99,
):
    """
    Build a runtime preset prior from the top-k uniformly sampled first generation.

    The first five depth tokens and the last twenty operation tokens have
    different semantics, so their frequencies are counted separately.
    lumda=1 uses only top-k frequencies; lumda=0 recovers the uniform prior.
    """
    lumda = float(lumda)
    if not 0.0 <= lumda <= 1.0:
        raise ValueError(f"lumda must be in [0, 1], got {lumda}")

    x = torch.as_tensor(initial_population).detach().cpu().long()
    scores = torch.as_tensor(initial_scores).detach().cpu().float()
    if x.dim() != 2 or x.shape[1] != num_edges:
        raise ValueError(f"initial_population must have shape [N, {num_edges}], got {tuple(x.shape)}")
    if scores.numel() != x.shape[0]:
        raise ValueError(
            f"initial_scores must have one score per candidate, got {scores.numel()} scores for {x.shape[0]} candidates"
        )

    topk = min(int(topk), x.shape[0])
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    top_indices = torch.topk(scores.view(-1), k=topk, largest=True).indices
    top_x = x[top_indices]

    depth_tokens = sorted(DEPTH_TOKEN_TO_VALUE)
    op_tokens = sorted(OP_TOKEN_TO_FEATURE)
    depth_uniform = _masked_uniform(depth_tokens, vocab_size)
    op_uniform = _masked_uniform(op_tokens, vocab_size)

    depth_counts = torch.bincount(
        top_x[:, :num_depth_tokens].reshape(-1).clamp(0, vocab_size - 1),
        minlength=vocab_size,
    ).float()
    op_counts = torch.bincount(
        top_x[:, num_depth_tokens:].reshape(-1).clamp(0, vocab_size - 1),
        minlength=vocab_size,
    ).float()

    depth_prior = _normalise_group_counts(depth_counts, depth_tokens, depth_uniform)
    op_prior = _normalise_group_counts(op_counts, op_tokens, op_uniform)

    depth_pi = (lumda * depth_prior) + ((1.0 - lumda) * depth_uniform)
    op_pi = (lumda * op_prior) + ((1.0 - lumda) * op_uniform)
    return build_marginal_prior_kernel(depth_prior=depth_pi, op_prior=op_pi)


def _normalise_group_counts(counts, valid_tokens, fallback):
    probs = torch.zeros(vocab_size, dtype=torch.float32)
    valid_tokens = list(valid_tokens)
    valid_count_sum = counts[valid_tokens].sum()
    if valid_count_sum <= 0:
        return fallback.clone()
    probs[valid_tokens] = counts[valid_tokens] / valid_count_sum
    return probs


def _preset_distribution(prior, valid_tokens, k, name):
    if prior is None:
        return _masked_uniform(valid_tokens, k)

    probs = torch.zeros(k, dtype=torch.float32)
    prior = torch.as_tensor(prior, dtype=torch.float32)
    if prior.numel() == len(valid_tokens):
        probs[valid_tokens] = prior
    elif prior.numel() == k:
        probs = prior.clone()
        invalid_tokens = [token for token in range(k) if token not in valid_tokens]
        if invalid_tokens:
            probs[invalid_tokens] = 0.0
    else:
        raise ValueError(f"{name} must have length {len(valid_tokens)} or {k}, got {prior.numel()}")

    if torch.any(probs < 0):
        raise ValueError(f"{name} cannot contain negative probabilities")
    if probs.sum() <= 0:
        raise ValueError(f"{name} must contain at least one positive probability")
    return probs / probs.sum()


def build_prior_kernel(prior_type, **kwargs):
    prior_type = str(prior_type).lower()
    if prior_type in ("structural", "structure", "distance"):
        return build_structural_prior_kernel(**kwargs)
    if prior_type in ("shuffled_structural", "shuffle", "shuffled", "random"):
        return build_shuffled_structural_prior_kernel(**kwargs)
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        return build_prior_knowledge_kernel(**kwargs)
    if prior_type in ("marginal", "preset"):
        return build_marginal_prior_kernel(
            depth_prior=kwargs.get("depth_prior"),
            op_prior=kwargs.get("op_prior"),
        )
    raise ValueError(
        "Unknown prior_type="
        f"{prior_type!r}; use 'structural', 'shuffled_structural', 'prior_Konwleage', or 'marginal'."
    )
