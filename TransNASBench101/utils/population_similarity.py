import torch


def individual_similarity(ind_a: torch.Tensor, ind_b: torch.Tensor) -> float:
    """计算两个个体之间的相似度（基于位置匹配比例）。

    输入可以是任意离散编码的向量，输出为匹配位置比例。
    """
    assert ind_a.shape == ind_b.shape, "individual shapes must match"
    return (ind_a == ind_b).to(torch.float32).mean().item()


def pairwise_similarity_matrix(
    pop_a: torch.Tensor,
    pop_b: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """计算两个种群之间的成对相似度矩阵。

    返回一个形状为 (len(pop_a), len(pop_b)) 的矩阵。
    """
    pop_a = pop_a.to(device)
    pop_b = pop_b.to(device)
    assert pop_a.dim() == 2 and pop_b.dim() == 2, "populations must be 2D tensors"
    assert pop_a.size(1) == pop_b.size(1), "genotype lengths must match"

    # 这里使用广播计算匹配比例
    eq = pop_a.unsqueeze(1) == pop_b.unsqueeze(0)
    sim = eq.to(torch.float32).mean(dim=-1)
    return sim


def average_pairwise_similarity(pop_a: torch.Tensor, pop_b: torch.Tensor, device: str = "cpu") -> float:
    """返回两个种群之间的平均成对相似度。"""
    sim_matrix = pairwise_similarity_matrix(pop_a, pop_b, device=device)
    return float(sim_matrix.mean().item())


def mean_best_match_similarity(pop_a: torch.Tensor, pop_b: torch.Tensor, device: str = "cpu") -> float:
    """返回每个个体与对侧种群中最相似个体的平均相似度。

    这个指标对不等大小种群更稳定。
    """
    sim_matrix = pairwise_similarity_matrix(pop_a, pop_b, device=device)
    best_a = sim_matrix.max(dim=1).values
    best_b = sim_matrix.max(dim=0).values
    return float((best_a.mean() + best_b.mean()).item() / 2.0)


def population_similarity(
    pop_a: torch.Tensor,
    pop_b: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """返回两个种群之间的相似度汇总指标。"""
    return {
        "average_pairwise_similarity": average_pairwise_similarity(pop_a, pop_b, device=device),
        "mean_best_match_similarity": mean_best_match_similarity(pop_a, pop_b, device=device),
    }
