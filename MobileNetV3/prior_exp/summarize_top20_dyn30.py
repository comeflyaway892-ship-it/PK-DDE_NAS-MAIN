"""Aggregate per-architecture retraining Max/Mean/Min over nine candidates."""

import csv
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent / "results" / "prior_Konwleage_top20_l1_dyn30"
    summary, details = [], []
    names = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
             "aircraft": "Aircraft", "pets": "Oxford-IIIT Pets"}
    for dataset, label in names.items():
        architecture_stats = []
        for seed in range(3):
            source = root / dataset / "mobilenet_retrain" / dataset / f"search_seed_{seed}" / "retrain_runs.jsonl"
            records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            assert len(records) == 9, source
            assert {r["arch_idx"] for r in records} == {0, 1, 2}, source
            for arch in range(3):
                runs = sorted((r for r in records if r["arch_idx"] == arch), key=lambda r: r["retrain_seed"])
                assert [r["retrain_seed"] for r in runs] == [777, 888, 999], source
                assert all(r["dataset"] == dataset and r["search_seed"] == seed for r in runs), source
                assert len({r["model_str"] for r in runs}) == 1, source
                values = [float(r["max_valid_acc"]) for r in runs]
                assert np.isfinite(values).all(), source
                arch_stats = [max(values), float(np.mean(values)), min(values)]
                architecture_stats.append(arch_stats)
                details.append(dict(dataset=dataset, search_seed=seed, arch_idx=arch,
                                    acc_777=values[0], acc_888=values[1], acc_999=values[2],
                                    arch_max=arch_stats[0], arch_mean=arch_stats[1],
                                    arch_min=arch_stats[2], source=str(source.relative_to(root))))
        assert len(architecture_stats) == 9
        for index, stat in enumerate(["Max", "Mean", "Min"]):
            values = np.asarray(architecture_stats)[:, index]
            summary.append(dict(dataset=label, stat=stat, mean=float(values.mean()),
                                std=float(values.std(ddof=1)), n_architectures=len(values), ddof=1))
    for filename, rows in [("table4_summary.csv", summary), ("table4_arch_details.csv", details)]:
        with (root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "prior_Konwleage_top20_l1_dyn30：四个数据集重训练统计（准确率 %）", "",
        "统计口径：使用 max_valid_acc（训练期间最高验证准确率）。每个架构对重训练种子",
        "777/888/999 分别计算 Max、Mean、Min；将搜索种子 0/1/2 各自的三个架构合并，",
        "对九个架构的 Max、Mean、Min 分别计算均值 ± 样本标准差（ddof=1）。",
        "按 (search_seed, arch_idx) 计九个候选，不跨搜索种子去重。这里的 ± 为标准差，不是置信区间。", "",
        "| Dataset | Stats. | Prior top20 λ=1 dyn30 |", "|---|---|---:|",
    ]
    lines.extend(f"| {r['dataset']} | {r['stat']} | {r['mean']:.2f} ± {r['std']:.2f} |" for r in summary)
    content = "\n".join(lines) + "\n"
    (root / "table4_summary.md").write_text(content)
    print(content)


if __name__ == "__main__":
    main()
