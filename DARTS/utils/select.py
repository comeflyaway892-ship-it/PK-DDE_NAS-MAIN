"""Population selection shared in spirit with the NAS-Bench-201 workflow."""

import torch


def select_population(
    x_t,
    fit_t,
    x_candidate,
    fit_candidate,
    population_size,
    elite_frac=0.3,
    eps=1e-20,
    fill_uniform=True,
    return_fitness=False,
):
    pool_x = torch.cat((x_t, x_candidate), dim=0).long()
    pool_fitness = torch.cat((fit_t, fit_candidate), dim=0).float()
    unique_x, inverse = torch.unique(pool_x, dim=0, return_inverse=True)

    unique_count = unique_x.shape[0]
    unique_fitness = torch.full(
        (unique_count,), -torch.inf, device=pool_fitness.device
    )
    if hasattr(unique_fitness, "scatter_reduce_"):
        unique_fitness.scatter_reduce_(
            0, inverse, pool_fitness, reduce="amax", include_self=True
        )
    else:
        for index in range(unique_count):
            unique_fitness[index] = pool_fitness[inverse == index].max()

    if unique_count < population_size:
        extra_count = population_size - unique_count
        if fill_uniform:
            extra = torch.randint(
                0, unique_count, (extra_count,), device=unique_x.device
            )
        else:
            weights = (unique_fitness - unique_fitness.min()).clamp_min(0.0) + eps
            weights /= weights.sum().clamp_min(eps)
            extra = torch.multinomial(weights, extra_count, replacement=True)
        indices = torch.cat(
            (torch.arange(unique_count, device=unique_x.device), extra), dim=0
        )
    elif unique_count == population_size:
        indices = torch.arange(unique_count, device=unique_x.device)
    else:
        elite_count = min(
            population_size, max(1, int(elite_frac * population_size + 0.999999))
        )
        order = torch.argsort(unique_fitness, descending=True)
        elite = order[:elite_count]
        remainder = order[elite_count:]
        needed = population_size - elite_count
        if needed:
            remainder_fitness = unique_fitness[remainder]
            weights = (
                remainder_fitness - remainder_fitness.min()
            ).clamp_min(0.0) + eps
            weights /= weights.sum().clamp_min(eps)
            chosen = torch.multinomial(weights, needed, replacement=False)
            indices = torch.cat((elite, remainder[chosen]), dim=0)
        else:
            indices = elite

    next_x = unique_x[indices]
    next_fitness = unique_fitness[indices]
    if return_fitness:
        return next_x, next_fitness
    return next_x

