#!/usr/bin/env python3
"""Plot ten-seed ImageNet mean curves for DiffusionNAG, ours, and EDNAG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "imagenet_population_trace_20260921_114106"
DIFFUSIONNAG = ROOT / "diffusionnag_all_seeds.csv"
OUTPUT_PNG = ROOT / "imagenet_three_methods_mean_curve.png"
OUTPUT_CSV = ROOT / "imagenet_three_methods_mean_curve.csv"


def load_ours(path, method):
    data = pd.read_csv(path)
    required = {"seed", "step", "population_mean_accuracy"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data.rename(columns={"population_mean_accuracy": "accuracy"})
    data["method"] = method
    data["curve_step"] = data["step"].astype(int)
    return data[["method", "seed", "curve_step", "accuracy"]]


def load_diffusionnag(path):
    data = pd.read_csv(path)
    required = {"seed", "step", "test_acc"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data["seed"] = data["seed"].astype(int)
    data["step"] = data["step"].astype(int)
    data["curve_step"] = ((data["step"] - 1) // 20) + 1
    # One plotted point represents the mean of each consecutive 20 diffusion steps.
    data = data.groupby(["seed", "curve_step"], as_index=False)["test_acc"].mean()
    data = data.rename(columns={"test_acc": "accuracy"})
    data["method"] = "DiffusionNAG"
    return data[["method", "seed", "curve_step", "accuracy"]]


def main():
    curves = pd.concat(
        [
            load_diffusionnag(DIFFUSIONNAG),
            load_ours(TRACE_DIR / "NAS_Bench_201" / "per_step.csv", "PK-DDE_NAS (ours)"),
            load_ours(TRACE_DIR / "NAS_Bench_201_EDNAG" / "per_step.csv", "EDNAG"),
        ],
        ignore_index=True,
    )
    counts = curves.groupby("method").agg(seeds=("seed", "nunique"), points=("curve_step", "nunique"))
    if not ((counts["seeds"] == 10) & (counts["points"] == 50)).all():
        raise ValueError(f"Expected 10 seeds and 50 curve points per method, got:\n{counts}")

    summary = (
        curves.groupby(["method", "curve_step"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(OUTPUT_CSV, index=False)

    styles = {
        "PK-DDE_NAS (ours)": {"color": "#1565c0", "linewidth": 2.6},
        "EDNAG": {"color": "#d95f02", "linewidth": 2.3},
        "DiffusionNAG": {"color": "#2ca25f", "linewidth": 2.3},
    }
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for method in ("PK-DDE_NAS (ours)", "EDNAG", "DiffusionNAG"):
        part = summary[summary["method"] == method].sort_values("curve_step")
        style = styles[method]
        ax.plot(part["curve_step"], part["mean_accuracy"], label=method, **style)
        std = part["std_accuracy"].fillna(0)
        ax.fill_between(
            part["curve_step"], part["mean_accuracy"] - std, part["mean_accuracy"] + std,
            color=style["color"], alpha=0.10,
        )
    ax.set_xlabel("Normalized search step")
    ax.set_ylabel("Mean test accuracy (%)")
    ax.set_title("ImageNet16-120: ten-seed mean search curves")
    ax.set_xlim(1, 50)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
