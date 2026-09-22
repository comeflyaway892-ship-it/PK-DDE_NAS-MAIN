import torch


def _normalise_rows(matrix, eps=1e-30):
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)


def _uniform_probs(vocab_size):
    return torch.full((vocab_size,), 1.0 / int(vocab_size), dtype=torch.float32)


def build_marginal_prior_kernel(num_edges, vocab_size, prior_probs=None):
    probs = _uniform_probs(vocab_size) if prior_probs is None else torch.as_tensor(prior_probs, dtype=torch.float32)
    if probs.numel() != vocab_size:
        raise ValueError(f"prior_probs must have length {vocab_size}, got {probs.numel()}")
    if torch.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("prior_probs must be non-negative and contain positive mass")
    probs = probs / probs.sum()
    return probs.view(1, 1, vocab_size).repeat(num_edges, vocab_size, 1)


def build_structural_prior_kernel(num_edges, vocab_size, tau=1.0):
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    tokens = torch.arange(vocab_size, dtype=torch.float32)
    distance = (tokens.view(-1, 1) - tokens.view(1, -1)).abs()
    kernel = torch.softmax(-distance / float(tau), dim=-1)
    return kernel.view(1, vocab_size, vocab_size).repeat(num_edges, 1, 1)


def build_shuffled_structural_prior_kernel(num_edges, vocab_size, tau=1.0, shuffle_seed=0):
    base = build_structural_prior_kernel(num_edges, vocab_size, tau=tau)
    generator = torch.Generator()
    generator.manual_seed(int(shuffle_seed))
    permutation = torch.randperm(vocab_size, generator=generator)
    shuffled = base.clone()
    shuffled[:] = base[0][permutation][:, permutation]
    return _normalise_rows(shuffled)


def _group_marginal_kernel(num_edges, vocab_size, group_probs):
    kernel = torch.empty(num_edges, vocab_size, vocab_size, dtype=torch.float32)
    for positions, probs in group_probs:
        probs = probs / probs.sum().clamp_min(1e-30)
        kernel[positions] = probs.view(1, 1, vocab_size).repeat(len(positions), vocab_size, 1)
    return kernel


def build_prior_knowledge_kernel(num_edges, vocab_size, initial_population, initial_scores, topk=10, lumda=0.99, groups=None):
    lumda = float(lumda)
    if not 0.0 <= lumda <= 1.0:
        raise ValueError(f"lumda must be in [0, 1], got {lumda}")
    x = torch.as_tensor(initial_population).detach().cpu().long()
    scores = torch.as_tensor(initial_scores).detach().cpu().float()
    if x.dim() != 2 or x.shape[1] != num_edges:
        raise ValueError(f"initial_population must have shape [N,{num_edges}], got {tuple(x.shape)}")
    if scores.numel() != x.shape[0]:
        raise ValueError("initial_scores must have one score per candidate")
    topk = min(int(topk), x.shape[0])
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    top_idx = torch.topk(scores.view(-1), k=topk, largest=True).indices
    top_x = x[top_idx]
    if groups is None:
        groups = [list(range(num_edges))]

    uniform = _uniform_probs(vocab_size)
    group_probs = []
    covered = set()
    for group in groups:
        positions = torch.as_tensor(group, dtype=torch.long)
        covered.update(int(pos) for pos in positions.tolist())
        counts = torch.bincount(top_x[:, positions].reshape(-1).clamp(0, vocab_size - 1), minlength=vocab_size).float()
        freq = counts / counts.sum().clamp_min(1e-30)
        probs = (lumda * freq) + ((1.0 - lumda) * uniform)
        group_probs.append((positions, probs))
    missing = [pos for pos in range(num_edges) if pos not in covered]
    if missing:
        group_probs.append((torch.as_tensor(missing, dtype=torch.long), uniform))
    return _group_marginal_kernel(num_edges, vocab_size, group_probs)


def build_prior_kernel(prior_type, num_edges, vocab_size, **kwargs):
    prior_type = str(prior_type).lower()
    if prior_type in ("structural", "structure", "distance"):
        return build_structural_prior_kernel(num_edges, vocab_size, tau=kwargs.get("tau", 1.0))
    if prior_type in ("shuffled_structural", "shuffle", "shuffled", "random"):
        return build_shuffled_structural_prior_kernel(
            num_edges,
            vocab_size,
            tau=kwargs.get("tau", 1.0),
            shuffle_seed=kwargs.get("shuffle_seed", 0),
        )
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        return build_prior_knowledge_kernel(
            num_edges,
            vocab_size,
            initial_population=kwargs["initial_population"],
            initial_scores=kwargs["initial_scores"],
            topk=kwargs.get("topk", 10),
            lumda=kwargs.get("lumda", 0.99),
            groups=kwargs.get("groups"),
        )
    if prior_type in ("marginal", "preset"):
        return build_marginal_prior_kernel(num_edges, vocab_size, prior_probs=kwargs.get("prior_probs"))
    raise ValueError(f"Unknown prior_type={prior_type!r}")
