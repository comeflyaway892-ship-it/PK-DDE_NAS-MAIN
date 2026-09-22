import os
import csv
import json
from collections import Counter

import sys
sys.path.append("TransNASBench101")
sys.path.append("./")

from api import TransNASBenchAPI as API


# =========================
# 基本配置
# =========================
PATH2NAS_BENCH = "transnas-bench_v10141024.pth"
OUTPUT_DIR = "macro_top50_op_frequency_results"
TOPK = 50

# 按你常用顺序
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
# room_layout 仍然按 loss，越小越好
METRIC_DICT = {
    "class_scene": "valid_top1",
    "class_object": "valid_top1",
    "room_layout": "train_loss",
    "jigsaw": "valid_top1",
    "segmentsemantic": "valid_mIoU",
    "normal": "valid_ssim",
    "autoencoder": "valid_ssim",
}

LOWER_IS_BETTER_TASKS = {"room_layout"}

# macro 空间中单个 module 的操作编码
# 根据官方 README:
# 1=normal, 2=channelx2, 3=resolution/2, 4=channelx2 & resolution/2
MACRO_OP_ID2NAME = {
    1: "normal",
    2: "channelx2",
    3: "resolution/2",
    4: "channelx2_resolution/2",
}


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_macro_arch(arch_str: str):
    """
    判断是否为 macro 搜索空间架构
    典型格式:
        64-1234-basic
        64-414141-basic
    """
    try:
        parts = arch_str.split("-")
        if len(parts) != 3:
            return False

        skeleton = parts[1]
        cell = parts[2]

        # macro 的 cell 一般是 basic
        if cell != "basic":
            return False

        if len(skeleton) < 4 or len(skeleton) > 6:
            return False

        for ch in skeleton:
            if ch not in {"1", "2", "3", "4"}:
                return False

        return True
    except Exception:
        return False


def trans_macro_arch_str_to_ops(arch_str: str):
    """
    将 macro 架构字符串解析为 skeleton 中每个 module 的操作列表
    例如:
        64-1234-basic -> [1, 2, 3, 4]
        64-414141-basic -> [4, 1, 4, 1, 4, 1]
    """
    try:
        parts = arch_str.split("-")
        if len(parts) != 3:
            raise ValueError(f"Unexpected macro arch format: {arch_str}")

        skeleton = parts[1]
        cell = parts[2]

        if cell != "basic":
            raise ValueError(f"Not a macro architecture (cell != basic): {arch_str}")

        ops = [int(ch) for ch in skeleton]

        if not (4 <= len(ops) <= 6):
            raise ValueError(f"Macro skeleton length should be 4~6, got {len(ops)} in {arch_str}")

        for x in ops:
            if x not in MACRO_OP_ID2NAME:
                raise ValueError(f"Invalid macro op id {x} in arch {arch_str}")

        return ops
    except Exception as e:
        raise ValueError(f"Failed to parse macro arch string: {arch_str}") from e


def get_metric_value(api: API, arch_str: str, task: str):
    metric = METRIC_DICT[task]
    value = api.get_single_metric(arch_str, task, metric, mode="best")
    return float(value)


def rank_architectures(api: API, task: str, macro_archs):
    """
    对 macro 架构按任务指标排序
    返回:
        [(arch_str, metric_value), ...]
    """
    results = []

    metric_name = METRIC_DICT[task]
    if metric_name not in api.metrics_dict[task]:
        raise ValueError(
            f"Task {task} does not contain metric {metric_name}. "
            f"Available metrics: {api.metrics_dict[task]}"
        )

    for arch_str in macro_archs:
        try:
            val = get_metric_value(api, arch_str, task)
            results.append((arch_str, val))
        except Exception as e:
            print(f"[Warning] Skip arch {arch_str} on task {task}: {e}")

    reverse = task not in LOWER_IS_BETTER_TASKS
    results.sort(key=lambda x: x[1], reverse=reverse)
    return results


def count_macro_op_frequency(top_archs):
    """
    统计 Top-K macro 架构中的操作出现频率

    返回:
        positionwise_counter: {
            "module_1": Counter(...),
            ...
            "module_6": Counter(...)
        }
        overall_counter: Counter(...)
        length_counter: Counter({4:xx, 5:xx, 6:xx})
    """
    max_modules = 6
    position_names = [f"module_{i}" for i in range(1, max_modules + 1)]
    positionwise_counter = {pos: Counter() for pos in position_names}
    overall_counter = Counter()
    length_counter = Counter()

    for arch_str, _ in top_archs:
        ops = trans_macro_arch_str_to_ops(arch_str)
        length_counter[len(ops)] += 1

        for i, op_id in enumerate(ops):
            pos_name = f"module_{i+1}"
            op_name = MACRO_OP_ID2NAME[op_id]
            positionwise_counter[pos_name][op_name] += 1
            overall_counter[op_name] += 1

    return positionwise_counter, overall_counter, length_counter


def save_top_archs_csv(task_dir, task, metric_name, top_archs):
    path = os.path.join(task_dir, f"{task}_top{len(top_archs)}_architectures.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "arch_str", metric_name])
        for i, (arch_str, val) in enumerate(top_archs, start=1):
            writer.writerow([i, arch_str, val])


def save_positionwise_csv(task_dir, task, positionwise_counter):
    path = os.path.join(task_dir, f"{task}_positionwise_op_frequency.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "op_name", "count", "frequency"])

        op_order = [
            "normal",
            "channelx2",
            "resolution/2",
            "channelx2_resolution/2",
        ]

        for pos_name, counter in positionwise_counter.items():
            total = sum(counter.values())
            for op_name in op_order:
                count = counter[op_name]
                freq = count / total if total > 0 else 0.0
                writer.writerow([pos_name, op_name, count, freq])


def save_overall_csv(task_dir, task, overall_counter):
    path = os.path.join(task_dir, f"{task}_overall_op_frequency.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["op_name", "count", "frequency"])

        op_order = [
            "normal",
            "channelx2",
            "resolution/2",
            "channelx2_resolution/2",
        ]

        total = sum(overall_counter.values())
        for op_name in op_order:
            count = overall_counter[op_name]
            freq = count / total if total > 0 else 0.0
            writer.writerow([op_name, count, freq])


def save_length_csv(task_dir, task, length_counter):
    path = os.path.join(task_dir, f"{task}_skeleton_length_frequency.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["num_modules", "count", "frequency"])

        total = sum(length_counter.values())
        for k in [4, 5, 6]:
            count = length_counter[k]
            freq = count / total if total > 0 else 0.0
            writer.writerow([k, count, freq])


def save_summary_json(task_dir, task, metric_name, top_archs,
                      positionwise_counter, overall_counter, length_counter):
    summary = {
        "task": task,
        "metric": metric_name,
        "topk": len(top_archs),
        "sort_order": "ascending" if task in LOWER_IS_BETTER_TASKS else "descending",
        "macro_op_id2name": MACRO_OP_ID2NAME,
        "top1_architecture": top_archs[0][0] if len(top_archs) > 0 else None,
        "top1_metric_value": top_archs[0][1] if len(top_archs) > 0 else None,
        "positionwise_count": {
            pos: dict(positionwise_counter[pos]) for pos in positionwise_counter
        },
        "overall_count": dict(overall_counter),
        "skeleton_length_count": dict(length_counter),
    }

    path = os.path.join(task_dir, f"{task}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_task_report(task, metric_name, top_archs,
                      positionwise_counter, overall_counter, length_counter):
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Metric: {metric_name}")
    print(f"Top-{len(top_archs)} selection rule: {'lower is better' if task in LOWER_IS_BETTER_TASKS else 'higher is better'}")

    if len(top_archs) > 0:
        print(f"Top-1 arch: {top_archs[0][0]}")
        print(f"Top-1 metric value: {top_archs[0][1]}")

    print("\n[Skeleton length frequency]")
    total_archs = sum(length_counter.values())
    for k in [4, 5, 6]:
        count = length_counter[k]
        freq = count / total_archs if total_archs > 0 else 0.0
        print(f"  length={k}: count={count}, freq={freq:.4f}")

    print("\n[Position-wise frequency]")
    op_order = [
        "normal",
        "channelx2",
        "resolution/2",
        "channelx2_resolution/2",
    ]
    for pos_name, counter in positionwise_counter.items():
        total = sum(counter.values())
        if total == 0:
            continue
        print(f"  {pos_name}:")
        for op_name in op_order:
            count = counter[op_name]
            freq = count / total if total > 0 else 0.0
            print(f"    {op_name:<25s} count={count:>3d}, freq={freq:.4f}")

    print("\n[Overall frequency]")
    total_ops = sum(overall_counter.values())
    for op_name in op_order:
        count = overall_counter[op_name]
        freq = count / total_ops if total_ops > 0 else 0.0
        print(f"  {op_name:<25s} count={count:>3d}, freq={freq:.4f}")
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

    # 收集所有 macro 架构
    if hasattr(api, "all_arch_dict") and "macro" in api.all_arch_dict:
        macro_archs = list(api.all_arch_dict["macro"])
    else:
        macro_archs = []
        for i in range(len(api)):
            arch_str = api.index2arch(i)
            if is_macro_arch(arch_str):
                macro_archs.append(arch_str)

    print(f"Number of macro architectures: {len(macro_archs)}")

    if len(macro_archs) == 0:
        raise RuntimeError("No macro architectures found.")

    for task in TASKS:
        print(f"\nProcessing task: {task}")

        if task not in api.task_list:
            print(f"[Warning] Task {task} not found in api.task_list, skip.")
            continue

        metric_name = METRIC_DICT[task]
        ranked_archs = rank_architectures(api, task, macro_archs)
        top_archs = ranked_archs[:TOPK]

        positionwise_counter, overall_counter, length_counter = count_macro_op_frequency(top_archs)

        task_dir = os.path.join(OUTPUT_DIR, task)
        ensure_dir(task_dir)

        save_top_archs_csv(task_dir, task, metric_name, top_archs)
        save_positionwise_csv(task_dir, task, positionwise_counter)
        save_overall_csv(task_dir, task, overall_counter)
        save_length_csv(task_dir, task, length_counter)
        save_summary_json(
            task_dir, task, metric_name, top_archs,
            positionwise_counter, overall_counter, length_counter
        )

        print_task_report(
            task, metric_name, top_archs,
            positionwise_counter, overall_counter, length_counter
        )

    print(f"All done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()