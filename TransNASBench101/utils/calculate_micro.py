import os
import csv
import json
from collections import Counter, defaultdict

import torch

# 根据你的项目结构保留这两行
import sys
sys.path.append("TransNASBench101")
sys.path.append("./")

from api import TransNASBenchAPI as API


# =========================
# 基本配置
# =========================
PATH2NAS_BENCH = "transnas-bench_v10141024.pth"
OUTPUT_DIR = "micro_top50_op_frequency_results"
TOPK = 50

# 按你常用顺序列 7 个任务
TASKS = [
    "class_object",      # OC
    "class_scene",       # SC
    "autoencoder",       # AE
    "normal",            # SE
    "segmentsemantic",   # SS
    "room_layout",       # RP
    "jigsaw",            # JS
]

# 各任务评价指标
# 注意：room_layout 统计的是 loss，越小越好
METRIC_DICT = {
    "class_scene": "valid_top1",
    "class_object": "valid_top1",
    "room_layout": "train_loss",
    "jigsaw": "valid_top1",
    "segmentsemantic": "valid_mIoU",
    "normal": "valid_ssim",
    "autoencoder": "valid_ssim",
}

# 哪些任务是越小越好
LOWER_IS_BETTER_TASKS = {"room_layout"}

# micro 搜索空间中单条边的操作编码
OP_ID2NAME = {
    0: "None",
    1: "Skip-Connect",
    2: "Conv1x1",
    3: "Conv3x3",
}

# micro 一共有 6 条边，编码顺序与 arch string 一致
EDGE_NAMES = [
    "edge_0_1",  # 第1位
    "edge_0_2",  # 第2位
    "edge_1_2",  # 第3位
    "edge_0_3",  # 第4位
    "edge_1_3",  # 第5位
    "edge_2_3",  # 第6位
]


# =========================
# 工具函数
# =========================
def trans_micro_arch_str_to_ops(arch_str: str):
    """
    将 micro 架构字符串解析为 6 条边的操作编码列表
    例如:
        64-41414-1_02_333
    对应:
        [1, 0, 2, 3, 3, 3]
    """
    try:
        cell_part = arch_str.split("-")[-1]   # 1_02_333
        parts = cell_part.split("_")
        if len(parts) != 3:
            raise ValueError(f"Unexpected micro cell format: {cell_part}")

        ops = []
        ops.append(int(parts[0]))      # edge_0_1
        ops.append(int(parts[1][0]))   # edge_0_2
        ops.append(int(parts[1][1]))   # edge_1_2
        ops.append(int(parts[2][0]))   # edge_0_3
        ops.append(int(parts[2][1]))   # edge_1_3
        ops.append(int(parts[2][2]))   # edge_2_3

        if len(ops) != 6:
            raise ValueError(f"Parsed ops length != 6, got {len(ops)}")

        for x in ops:
            if x not in OP_ID2NAME:
                raise ValueError(f"Invalid op id {x} in arch {arch_str}")

        return ops

    except Exception as e:
        raise ValueError(f"Failed to parse micro arch string: {arch_str}") from e


def get_metric_value(api: API, arch_str: str, task: str):
    """
    获取某个架构在某个任务上的排序指标
    """
    metric = METRIC_DICT[task]
    value = api.get_single_metric(arch_str, task, metric, mode="best")
    return float(value)


def is_micro_arch(arch_str: str):
    """
    判断是否为 micro 搜索空间架构
    根据你之前脚本中的说明，micro 形如:
        64-41414-x_xx_xxx
    """
    try:
        parts = arch_str.split("-")
        if len(parts) != 3:
            return False
        # micro 的最后一段应包含两个下划线
        return parts[-1].count("_") == 2
    except Exception:
        return False


def rank_architectures(api: API, task: str, micro_archs):
    """
    对 micro 架构按任务指标排序，返回:
        ranked_list: [(arch_str, metric_value), ...]
    """
    results = []

    metric_name = METRIC_DICT[task]
    if metric_name not in api.metrics_dict[task]:
        raise ValueError(
            f"Task {task} does not contain metric {metric_name}. "
            f"Available metrics: {api.metrics_dict[task]}"
        )

    for arch_str in micro_archs:
        try:
            val = get_metric_value(api, arch_str, task)
            results.append((arch_str, val))
        except Exception as e:
            print(f"[Warning] Skip arch {arch_str} on task {task}: {e}")

    reverse = task not in LOWER_IS_BETTER_TASKS
    results.sort(key=lambda x: x[1], reverse=reverse)
    return results


def count_op_frequency(top_archs):
    """
    统计 Top-K 架构中的操作出现频率
    返回:
        edgewise_counter: {
            edge_name: Counter({op_name: count, ...}),
            ...
        }
        overall_counter: Counter({op_name: count, ...})
    """
    edgewise_counter = {edge: Counter() for edge in EDGE_NAMES}
    overall_counter = Counter()

    for arch_str, _ in top_archs:
        ops = trans_micro_arch_str_to_ops(arch_str)
        for edge_name, op_id in zip(EDGE_NAMES, ops):
            op_name = OP_ID2NAME[op_id]
            edgewise_counter[edge_name][op_name] += 1
            overall_counter[op_name] += 1

    return edgewise_counter, overall_counter


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_top_archs_csv(task_dir, task, metric_name, top_archs):
    path = os.path.join(task_dir, f"{task}_top{len(top_archs)}_architectures.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "arch_str", metric_name])
        for i, (arch_str, val) in enumerate(top_archs, start=1):
            writer.writerow([i, arch_str, val])


def save_edgewise_csv(task_dir, task, edgewise_counter, topk):
    path = os.path.join(task_dir, f"{task}_edgewise_op_frequency.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["edge", "op_name", "count", "frequency"])

        for edge in EDGE_NAMES:
            total = sum(edgewise_counter[edge].values())
            for op_name in ["None", "Skip-Connect", "Conv1x1", "Conv3x3"]:
                count = edgewise_counter[edge][op_name]
                freq = count / total if total > 0 else 0.0
                writer.writerow([edge, op_name, count, freq])


def save_overall_csv(task_dir, task, overall_counter, total_edges):
    path = os.path.join(task_dir, f"{task}_overall_op_frequency.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["op_name", "count", "frequency"])

        for op_name in ["None", "Skip-Connect", "Conv1x1", "Conv3x3"]:
            count = overall_counter[op_name]
            freq = count / total_edges if total_edges > 0 else 0.0
            writer.writerow([op_name, count, freq])


def save_summary_json(task_dir, task, metric_name, top_archs, edgewise_counter, overall_counter):
    summary = {
        "task": task,
        "metric": metric_name,
        "topk": len(top_archs),
        "sort_order": "ascending" if task in LOWER_IS_BETTER_TASKS else "descending",
        "edge_names": EDGE_NAMES,
        "op_id2name": OP_ID2NAME,
        "top1_architecture": top_archs[0][0] if len(top_archs) > 0 else None,
        "top1_metric_value": top_archs[0][1] if len(top_archs) > 0 else None,
        "edgewise_count": {
            edge: dict(edgewise_counter[edge]) for edge in EDGE_NAMES
        },
        "overall_count": dict(overall_counter),
    }

    path = os.path.join(task_dir, f"{task}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_task_report(task, metric_name, top_archs, edgewise_counter, overall_counter):
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Metric: {metric_name}")
    print(f"Top-{len(top_archs)} selection rule: {'lower is better' if task in LOWER_IS_BETTER_TASKS else 'higher is better'}")

    if len(top_archs) > 0:
        print(f"Top-1 arch: {top_archs[0][0]}")
        print(f"Top-1 metric value: {top_archs[0][1]}")

    print("\n[Edge-wise frequency]")
    for edge in EDGE_NAMES:
        total = sum(edgewise_counter[edge].values())
        print(f"  {edge}:")
        for op_name in ["None", "Skip-Connect", "Conv1x1", "Conv3x3"]:
            count = edgewise_counter[edge][op_name]
            freq = count / total if total > 0 else 0.0
            print(f"    {op_name:<13s} count={count:>3d}, freq={freq:.4f}")

    print("\n[Overall frequency]")
    total_edges = sum(overall_counter.values())
    for op_name in ["None", "Skip-Connect", "Conv1x1", "Conv3x3"]:
        count = overall_counter[op_name]
        freq = count / total_edges if total_edges > 0 else 0.0
        print(f"  {op_name:<13s} count={count:>3d}, freq={freq:.4f}")
    print()


# =========================
# 主流程
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading TransNASBench101 API...")
    api = API(PATH2NAS_BENCH)

    print(f"Total architectures: {len(api)}")
    print(f"Tasks: {api.task_list}")
    print(f"Search spaces: {api.search_spaces}")

    # 收集所有 micro 架构
    # 优先使用 all_arch_dict["micro"]
    if hasattr(api, "all_arch_dict") and "micro" in api.all_arch_dict:
        micro_archs = list(api.all_arch_dict["micro"])
    else:
        # 兜底：遍历所有 index，筛出 micro
        micro_archs = []
        for i in range(len(api)):
            arch_str = api.index2arch(i)
            if is_micro_arch(arch_str):
                micro_archs.append(arch_str)

    print(f"Number of micro architectures: {len(micro_archs)}")

    if len(micro_archs) == 0:
        raise RuntimeError("No micro architectures found.")

    for task in TASKS:
        print(f"\nProcessing task: {task}")

        if task not in api.task_list:
            print(f"[Warning] Task {task} not found in api.task_list, skip.")
            continue

        metric_name = METRIC_DICT[task]
        ranked_archs = rank_architectures(api, task, micro_archs)
        top_archs = ranked_archs[:TOPK]

        edgewise_counter, overall_counter = count_op_frequency(top_archs)

        task_dir = os.path.join(OUTPUT_DIR, task)
        ensure_dir(task_dir)

        save_top_archs_csv(task_dir, task, metric_name, top_archs)
        save_edgewise_csv(task_dir, task, edgewise_counter, len(top_archs))
        save_overall_csv(task_dir, task, overall_counter, len(top_archs) * 6)
        save_summary_json(task_dir, task, metric_name, top_archs, edgewise_counter, overall_counter)

        print_task_report(task, metric_name, top_archs, edgewise_counter, overall_counter)

    print(f"All done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()