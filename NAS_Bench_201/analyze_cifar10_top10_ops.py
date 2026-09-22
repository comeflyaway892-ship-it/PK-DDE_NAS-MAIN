#!/usr/bin/env python3
"""Count operation frequencies in the top 10% of NAS-Bench-201 on CIFAR-10."""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_API_PATH = (
    SCRIPT_DIR / "nas_201_api" / "NAS_Bench_201-v1_1-096897.pth"
)
DEFAULT_OUTPUT_PATH = (
    SCRIPT_DIR / "results" / "cifar10_top10_operation_frequency.txt"
)
MAX_CPU_THREADS = 32
OPERATIONS = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank all NAS-Bench-201 architectures by their averaged 200-epoch "
            "CIFAR-10 test accuracy and count operations in the top 10%."
        )
    )
    parser.add_argument(
        "--api-path",
        type=Path,
        default=DEFAULT_API_PATH,
        help="Path to the complete NAS-Bench-201 v1.1 benchmark file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Text file to receive the frequency summary.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(MAX_CPU_THREADS, os.cpu_count() or 1),
        help="Number of CPU threads to use (maximum: 32).",
    )
    return parser.parse_args()


def configure_cpu_only(threads):
    if not 1 <= threads <= MAX_CPU_THREADS:
        raise ValueError("--threads must be between 1 and 32")

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = sorted(os.sched_getaffinity(0))
        selected_cpus = available_cpus[: min(threads, len(available_cpus))]
        if not selected_cpus:
            raise RuntimeError("No CPU is available to this process")
        os.sched_setaffinity(0, selected_cpus)
        threads = len(selected_cpus)

    # These variables must be set before importing torch/numpy through the API.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(threads)

    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch, threads


def load_api(api_path):
    if not api_path.is_file():
        raise FileNotFoundError("NAS-Bench-201 API file not found: {}".format(api_path))

    sys.path.insert(0, str(SCRIPT_DIR))
    from nas_201_api import NASBench201API

    return NASBench201API(str(api_path), verbose=False)


def collect_ranked_accuracies(api):
    ranked = []
    for index in range(len(api)):
        accuracy = float(api.query_test_acc_by_index(index, "cifar10"))
        ranked.append((accuracy, index))

    # Architecture index gives deterministic behavior when accuracies are tied.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def count_operations(api, selected):
    counts = Counter({operation: 0 for operation in OPERATIONS})
    for _, index in selected:
        architecture = api.arch(index)
        architecture_operations = [
            operation
            for node in api.str2lists(architecture)
            for operation, _ in node
        ]
        if len(architecture_operations) != 6:
            raise ValueError(
                "Architecture {} has {} operations instead of 6: {}".format(
                    index, len(architecture_operations), architecture
                )
            )
        unknown = set(architecture_operations).difference(OPERATIONS)
        if unknown:
            raise ValueError(
                "Architecture {} contains unknown operations: {}".format(
                    index, sorted(unknown)
                )
            )
        counts.update(architecture_operations)
    return counts


def format_summary(api_path, threads, ranked, selected, counts):
    total_architectures = len(ranked)
    selected_count = len(selected)
    total_occurrences = sum(counts.values())
    cutoff_accuracy = selected[-1][0]
    cutoff_tie_total = sum(
        accuracy == cutoff_accuracy for accuracy, _ in ranked
    )
    cutoff_tie_selected = sum(
        accuracy == cutoff_accuracy for accuracy, _ in selected
    )

    lines = [
        "NAS-Bench-201 CIFAR-10 top-10% operation frequency",
        "api_path={}".format(api_path.resolve()),
        "dataset=cifar10",
        "ranking_metric=mean_200_epoch_ori-test_accuracy",
        "tie_breaker=architecture_index_ascending",
        "cpu_threads={}".format(threads),
        "gpu_enabled=false",
        "total_architectures={}".format(total_architectures),
        "selection_rule=ceil(total_architectures * 0.10)",
        "selected_architectures={}".format(selected_count),
        "best_accuracy={:.6f}".format(selected[0][0]),
        "cutoff_accuracy={:.6f}".format(cutoff_accuracy),
        "cutoff_tie_selected={}".format(cutoff_tie_selected),
        "cutoff_tie_total={}".format(cutoff_tie_total),
        "operations_per_architecture=6",
        "total_operation_occurrences={}".format(total_occurrences),
        "",
        "operation\tcount\tfrequency\tpercentage",
    ]
    for operation in OPERATIONS:
        frequency = counts[operation] / float(total_occurrences)
        lines.append(
            "{}\t{}\t{:.8f}\t{:.4f}%".format(
                operation, counts[operation], frequency, frequency * 100.0
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    torch, effective_threads = configure_cpu_only(args.threads)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible even though CPU-only execution was requested")

    print("Loading the complete NAS-Bench-201 API from {}".format(args.api_path))
    api = load_api(args.api_path)
    ranked = collect_ranked_accuracies(api)
    if not ranked:
        raise RuntimeError("The NAS-Bench-201 API contains no architectures")

    top_count = int(math.ceil(len(ranked) * 0.10))
    selected = ranked[:top_count]
    counts = count_operations(api, selected)
    summary = format_summary(
        args.api_path, effective_threads, ranked, selected, counts
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print("Saved summary to {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
