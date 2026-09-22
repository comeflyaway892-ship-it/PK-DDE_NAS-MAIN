import torch


def compute_uniqueness(arch_op_matrices: torch.Tensor):
    """
    计算张量的独特率：不同行的数量 / 总行数

    参数:
        arch_op_matrices: 输入的torch张量，第一维为样本数（行）

    返回:
        float: 独特率，范围在(0, 1]之间
    """
    # 获取总样本数
    population = arch_op_matrices.shape[0]

    # 如果只有1个样本，独特率直接为1.0
    if population == 0:
        return 0.0
    if population == 1:
        return 1.0

    # Macro architectures have 7 tokens and micro architectures have 6.
    arch_op_matrices = arch_op_matrices.reshape(population, -1)

    # 找出所有唯一的行
    # unique返回：唯一行张量, 索引, 逆索引, 计数
    unique_rows= torch.unique(
        arch_op_matrices,
        dim=0,  # 按行维度去重
    )

    # 计算独特率：唯一行的数量 / 总行数
    uniqueness_rate = float(len(unique_rows)) / population

    return uniqueness_rate
