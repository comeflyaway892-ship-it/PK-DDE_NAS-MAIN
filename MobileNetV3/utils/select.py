import torch

def select_population(
    x_t: torch.Tensor,          # (N, D) 父代
    fit_t: torch.Tensor,        # (N,)   父代fitness
    x_cand: torch.Tensor,       # (N, D) 子代候选（例如你从x_t估计x0后再d3pm一步得到的候选）
    fit_cand: torch.Tensor,     # (N,)   子代fitness
    N: int,
    elite_frac: float = 0.3,
    eps: float = 1e-20,
    fill_uniform: bool = True,  # True: 不看fitness随机复制；False: 按fitness加权复制
    return_fitness: bool = False,
):
    """
    返回:
      x_next:  (N, D)
      fit_next:(N,)
      idx_src: (N,)  选中个体在“去重后池”中的索引（方便debug）
    说明:
      - fitness 越大越好
      - 轮盘赌权重不使用softmax，而是 (f - min + eps) 归一化
      - 去重用 torch.unique(dim=0)，并用 scatter_reduce 取重复项中的最大fitness
    """
    device = x_t.device
    dtype_fit = fit_t.dtype

    # 0) 合并父子代池
    pool_x = torch.cat([x_t, x_cand], dim=0).long()              # (2N, D)
    pool_f = torch.cat([fit_t, fit_cand], dim=0).to(dtype_fit)   # (2N,)

    # 1) 去重（按结构完全相同去
    # uniq_x: (M, D), inv: (2N,) 表示每个原样本对应的uniq索引
    uniq_x, inv = torch.unique(pool_x, dim=0, return_inverse=True)

    M = uniq_x.shape[0]
    # 取每个uniq结构对应的最大fitness（重复里选最好的）
    uniq_f = torch.full((M,), -torch.inf, device=device, dtype=dtype_fit)
    # 需要 torch>=1.12 的 scatter_reduce_
    if hasattr(uniq_f, "scatter_reduce_"):
        uniq_f.scatter_reduce_(0, inv, pool_f, reduce="amax", include_self=True)
    else:
        # 兼容兜底：M一般不大，用循环也能跑
        for k in range(M):
            uniq_f[k] = pool_f[inv == k].max()

    # 2) 若去重后不足 N：随机复制补齐（你指定的策略）
    if M < N:
        # 先把uniq全保留，再补 N-M 个复制
        num_extra = N - M
        if fill_uniform:
            extra_idx = torch.randint(low=0, high=M, size=(num_extra,), device=device)
        else:
            # 可选：按fitness加权复制（仍不softmax）
            w = (uniq_f - uniq_f.min()).clamp_min(0.0) + eps
            w = w / (w.sum() + eps)
            extra_idx = torch.multinomial(w, num_samples=num_extra, replacement=True)

        idx = torch.cat([torch.arange(M, device=device), extra_idx], dim=0)
        x_next = uniq_x[idx]
        fit_next = uniq_f[idx]
        if return_fitness:
            return x_next, fit_next
        return x_next

    # 3) 若去重后恰好 N：直接返回
    if M == N:
        idx = torch.arange(M, device=device)
        if return_fitness:
            return uniq_x, uniq_f
        return uniq_x

    # 4) 若去重后超过 N：精英 + 轮盘赌
    E = max(1, int((elite_frac * N) + 0.999999))  # ceil
    E = min(E, N)

    # 排序（降序）
    order = torch.argsort(uniq_f, descending=True)
    elite_idx = order[:E]         # 精英
    rest_idx = order[E:]          # 剩余候选

    # 轮盘赌选 N-E 个（无放回）
    need = N - E
    if need > 0:
        # fitness based
        rest_f = uniq_f[rest_idx]
        w = (rest_f - rest_f.min()).clamp_min(0.0) + eps
        w = w / (w.sum() + eps)

        # multinomial 无放回抽样
        chosen_local = torch.multinomial(w, num_samples=need, replacement=False)
        roulette_idx = rest_idx[chosen_local]

        # rank based by linear
        # m = rest_idx.numel()
        # rank_w = torch.arange(m, 0, -1, device=device, dtype=torch.float)  # [m..1]
        # w = rank_w / (rank_w.sum() + eps)
        #
        # chosen_local = torch.multinomial(w, num_samples=need, replacement=False)
        # roulette_idx = rest_idx[chosen_local]

        #rank based by exp
        # m = rest_idx.numel()
        # eta = 0.97  # 可调：0.90更偏前，0.99更均匀
        # r = torch.arange(m, device=device, dtype=torch.float)
        # rank_w = torch.pow(torch.tensor(eta, device=device), r)
        # w = rank_w / (rank_w.sum() + eps)
        #
        # chosen_local = torch.multinomial(w, num_samples=need, replacement=False)
        # roulette_idx = rest_idx[chosen_local]

        idx = torch.cat([elite_idx, roulette_idx], dim=0)
    else:
        idx = elite_idx

    x_next = uniq_x[idx]
    fit_next = uniq_f[idx]
    if return_fitness:
        return x_next, fit_next
    return x_next
