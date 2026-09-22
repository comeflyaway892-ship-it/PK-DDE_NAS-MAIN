#!/usr/bin/env python3
"""Run the Pets search in both NB201 implementations and save per-step traces.

The reported accuracy is the search-time meta-predictor accuracy of each newly
generated population. No architecture retraining is performed.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def seed_everything(seed):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(output, project, settings, records):
    project_dir = output / project
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    with (project_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    rows = []
    for record in records:
        trace = record["trace"]
        for index, step in enumerate(trace["step"]):
            rows.append({
                "project": project,
                "seed": record["seed"],
                "step": step,
                "population_mean_accuracy": trace["population_mean_accuracy"][index],
                "population_max_accuracy": trace["population_max_accuracy"][index],
                "valid_rate": trace["valid_rate"][index],
                "unique_rate": trace["unique_rate"][index],
            })
    fields = ["project", "seed", "step", "population_mean_accuracy",
              "population_max_accuracy", "valid_rate", "unique_rate"]
    write_rows(project_dir / "per_step.csv", rows, fields)

    summary = []
    steps = sorted({row["step"] for row in rows})
    for step in steps:
        values = np.asarray([row["population_mean_accuracy"] for row in rows if row["step"] == step], dtype=float)
        summary.append({
            "project": project,
            "step": step,
            "mean_population_accuracy": float(values.mean()),
            "std_population_accuracy": float(values.std(ddof=0)),
            "std_sample_population_accuracy": float(values.std(ddof=1)) if values.size > 1 else None,
            "se_population_accuracy": float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else None,
            "runs": int(values.size),
        })
    write_rows(project_dir / "summary.csv", summary, list(summary[0]))
    (project_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def plot_outputs(output, project, rows, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_dir = output / project
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    fig, ax = plt.subplots(figsize=(10, 6))
    for seed, values in sorted(by_seed.items()):
        values.sort(key=lambda row: row["step"])
        ax.plot([row["step"] for row in values], [row["population_mean_accuracy"] for row in values],
                color="0.70", alpha=0.45, linewidth=0.9)
    ax.plot([row["step"] for row in summary], [row["mean_population_accuracy"] for row in summary],
            color="#1565c0", linewidth=2.5, label="mean across 10 seeds")
    if len(summary) > 1:
        mean = np.asarray([row["mean_population_accuracy"] for row in summary])
        sd = np.asarray([row["std_population_accuracy"] for row in summary])
        steps = [row["step"] for row in summary]
        ax.fill_between(steps, mean - sd, mean + sd, color="#90caf9", alpha=0.35, label="± population SD")
    ax.set_xlabel("Search step")
    ax.set_ylabel("Population mean predicted accuracy (%)")
    ax.set_title(f"Pets population accuracy trace: {project}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(project_dir / "population_mean_accuracy.png", dpi=180)
    plt.close(fig)


def run_nb201(args, output):
    os.chdir(ROOT / "NAS_Bench_201")
    sys.path.insert(0, str(ROOT / "NAS_Bench_201"))
    import torch
    from config.config import meta_hyper_params_setting, nb201_hyper_params_setting, nb201_api_path
    from prior_exp.evo_diff_prior import evo_diff_meta_with_prior, evo_diff_with_prior
    from prior_exp.experiments_prior import set_random_seed
    from utils.nb201_fitness import load_nb201_api
    from config.config import (predictor_estimator_eps, predictor_sigma,
                               select_elite_frac, select_eps, select_fill_uniform)

    dataset = "pets" if args.dataset == "pets" else "ImageNet16-120"
    is_meta = dataset == "pets"
    base = dict(meta_hyper_params_setting["pets"] if is_meta else nb201_hyper_params_setting[dataset])
    steps = args.steps if args.steps is not None else int(base["num_step"])
    population = args.population if args.population is not None else int(base["population_num"])
    temperature = args.temperature if args.temperature is not None else 0.4
    api = load_nb201_api(nb201_api_path, verbose=False)
    records = []
    prior_kwargs = {"topk": args.topk, "lumda": 0.99, "dynamic_steps": args.dynamic_steps}
    settings = {
        "project": "NAS_Bench_201", "dataset": dataset, "runs": args.runs,
        "seeds": list(range(args.runs)), "steps": steps, "population": population,
        "temperature": temperature, "prior_type": "prior_Konwleage",
        "prior_kwargs": prior_kwargs,
        "search_accuracy": "meta-predictor predicted accuracy" if is_meta else "NAS-Bench-201 API test accuracy",
        "retraining": False,
        "report_split": "search-time meta predictor; no retraining" if is_meta else "NAS-Bench-201 API x-test; no retraining",
    }
    for seed in range(args.runs):
        print(f"[NAS_Bench_201] seed={seed} start", flush=True)
        set_random_seed(seed)
        if is_meta:
            result = evo_diff_meta_with_prior(
                dataset=dataset, api=api, num_step=steps, population_num=population,
                seed=seed, plot_results=False, save_dir=str(output / "NAS_Bench_201"),
                prior_type="prior_Konwleage", prior_kwargs=dict(prior_kwargs),
                predictor_estimator_eps=predictor_estimator_eps,
                predictor_temperature=temperature, predictor_sigma=predictor_sigma,
                select_elite_frac=select_elite_frac, select_eps=select_eps,
                select_fill_uniform=select_fill_uniform, return_trace=True,
            )
        else:
            result = evo_diff_with_prior(
                dataset=dataset, api=api, num_step=steps, population_num=population,
                seed=seed, plot_results=False, save_dir=str(output / "NAS_Bench_201"),
                prior_type="prior_Konwleage", prior_kwargs=dict(prior_kwargs),
                predictor_estimator_eps=predictor_estimator_eps,
                predictor_temperature=temperature, predictor_sigma=predictor_sigma,
                select_elite_frac=select_elite_frac, select_eps=select_eps,
                select_fill_uniform=select_fill_uniform, return_trace=True,
            )
        trace = result[4]
        records.append({"seed": seed, "max_accuracy": result[0], "duration_seconds": result[1], "trace": trace})
        print(f"[NAS_Bench_201] seed={seed} done max={result[0]:.4f}", flush=True)
    rows, summary = save_outputs(output, "NAS_Bench_201", settings, records)
    plot_outputs(output, "NAS_Bench_201", rows, summary)


def run_ednag(args, output):
    os.chdir(ROOT / "NAS_Bench_201_EDNAG")
    sys.path.insert(0, str(ROOT / "NAS_Bench_201_EDNAG"))
    from config.config import meta_hyper_params_setting, nb201_hyper_params_setting
    from evo_diff import evo_diff, evo_diff_meta
    from experiments import set_random_seed
    from utils.nb201_fitness import load_nb201_api
    from utils.paths import NB201_API_V1_1

    dataset = "pets" if args.dataset == "pets" else "ImageNet16-120"
    is_meta = dataset == "pets"
    base = dict(meta_hyper_params_setting["pets"] if is_meta else nb201_hyper_params_setting[dataset])
    steps = args.steps if args.steps is not None else int(base["num_step"])
    population = args.population if args.population is not None else int(base["population_num"])
    temperature = args.temperature if args.temperature is not None else float(base["temperature"])
    api = load_nb201_api(NB201_API_V1_1, verbose=False)
    records = []
    settings = {
        "project": "NAS_Bench_201_EDNAG", "dataset": dataset, "runs": args.runs,
        "seeds": list(range(args.runs)), "steps": steps, "population": population,
        "temperature": temperature, "algorithm": "EDNAG meta predictor" if is_meta else "EDNAG",
        "search_accuracy": "meta-predictor predicted accuracy" if is_meta else "NAS-Bench-201 API test accuracy",
        "retraining": False,
        "report_split": "search-time meta predictor; no retraining" if is_meta else "NAS-Bench-201 API x-test; no retraining",
    }
    for seed in range(args.runs):
        print(f"[NAS_Bench_201_EDNAG] seed={seed} start", flush=True)
        set_random_seed(seed)
        if is_meta:
            result = evo_diff_meta(
                dataset=dataset, api=api, num_step=steps, population_num=population,
                geno_shape=base["geno_shape"], temperature=temperature,
                diver_rate=base["diver_rate"], noise_scale=base["noise_scale"],
                mutate_rate=base["mutate_rate"], elite_rate=base["elite_rate"],
                mutate_distri_index=base["mutate_distri_index"], seed=seed,
                plot_results=False, save_dir=str(output / "NAS_Bench_201_EDNAG"),
                nb201_or_meta="meta", max_iter_time=base["max_iter_time"], return_trace=True,
            )
        else:
            result = evo_diff(
                dataset=dataset, api=api, num_step=steps, population_num=population,
                temperature=temperature, seed=seed, plot_results=False,
                save_dir=str(output / "NAS_Bench_201_EDNAG"), nb201_or_meta="nb201",
                max_iter_time=base["max_iter_time"], return_trace=True,
            )
        trace = result[4]
        records.append({"seed": seed, "max_accuracy": result[0], "duration_seconds": result[1], "trace": trace})
        print(f"[NAS_Bench_201_EDNAG] seed={seed} done max={result[0]:.4f}", flush=True)
    rows, summary = save_outputs(output, "NAS_Bench_201_EDNAG", settings, records)
    plot_outputs(output, "NAS_Bench_201_EDNAG", rows, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=["both", "nb201", "ednag"], default="both")
    parser.add_argument("--dataset", choices=["pets", "imagenet"], default="pets")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.project == "both":
        output = args.output or (ROOT / ("pets_population_trace_" + time.strftime("%Y%m%d_%H%M%S")))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "retraining": False}
        (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for project in ("nb201", "ednag"):
            command = [PYTHON, "-u", str(Path(__file__).resolve()), "--project", project,
                       "--dataset", args.dataset, "--runs", str(args.runs), "--output", str(output)]
            if args.steps is not None: command += ["--steps", str(args.steps)]
            if args.population is not None: command += ["--population", str(args.population)]
            if args.temperature is not None: command += ["--temperature", str(args.temperature)]
            command += ["--topk", str(args.topk), "--dynamic-steps", str(args.dynamic_steps)]
            subprocess.run(command, cwd=ROOT, check=True)
        return
    if args.output is None:
        parser.error("--output is required for a single project")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.project == "nb201":
        run_nb201(args, output)
    else:
        run_ednag(args, output)


if __name__ == "__main__":
    main()
