"""Run 100 seeds at 30 steps / 20 candidates, reusing meta retraining caches."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


NAS_ROOT = Path(__file__).resolve().parents[1]
RESULTS = NAS_ROOT / "prior_exp" / "results"
OLD_PRIOR = "prior_Konwleage_top20_l1_dyn40"
DATASETS = ("cifar10", "cifar100", "aircraft", "pets")


def worker(args, prior_name):
    os.chdir(NAS_ROOT)
    sys.path.insert(0, str(NAS_ROOT))
    from config import config as search_config
    # Set before importing experiments_prior, which imports the value by name.
    # This override is local to the worker process; config.py stays unchanged.
    search_config.predictor_temperature = args.temperature
    for settings in (search_config.meta_hyper_params_setting, search_config.nb201_hyper_params_setting):
        for config in settings.values():
            config.update(num_step=args.steps, population_num=args.population)
    from prior_exp.experiments_prior import (
        exp_with_fixed_seed_in_meta_predictor_prior,
        exp_with_fixed_seed_in_nb201_prior,
    )
    dataset = args.worker
    kind = "nb201" if dataset.startswith("cifar") else "meta"
    out = RESULTS / kind / prior_name / dataset
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset=dataset, steps=args.steps, population=args.population,
                    topk=args.knowledge_topk, lumda=0.99, dynamic_steps=args.dynamic_steps,
                    temperature=args.temperature,
                    cache_prior=args.cache_prior if kind == "meta" else None)
    manifest = out / "search_settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f"Search settings differ from {manifest}; use another output name")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    kwargs = dict(prior_type="prior_Konwleage", prior_name=prior_name,
                  prior_kwargs=dict(topk=args.knowledge_topk, lumda=0.99, dynamic_steps=args.dynamic_steps),
                  seed_list=list(range(args.runs)))
    if kind == "meta":
        cache = RESULTS / "meta" / args.cache_prior / dataset / "retrain_artifacts"
        exp_with_fixed_seed_in_meta_predictor_prior(dataset, retrain_artifact_dir=str(cache), **kwargs)
    else:
        exp_with_fixed_seed_in_nb201_prior(dataset, **kwargs)
    import torch
    import numpy as np
    records = torch.load(out / "search_log" / f"{dataset}_{prior_name}_search.pth", map_location="cpu")
    values = np.array([float(records[s]["max_acc"]) for s in range(args.runs)])
    summary = dict(metadata, runs=args.runs, mean=float(values.mean()),
                   std_sample=float(values.std(ddof=1)) if args.runs > 1 else None,
                   std_population=float(values.std()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--knowledge-topk", type=int, default=6)
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=7.0)
    parser.add_argument("--cache-prior", default=OLD_PRIOR)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=DATASETS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.runs, args.steps, args.population, args.knowledge_topk) <= 0 or args.dynamic_steps < 0:
        parser.error("runs, steps, population and topk must be positive; dynamic-steps must be nonnegative")
    if args.knowledge_topk > args.population:
        parser.error("knowledge-topk must not exceed population")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    prior_name = (f"prior_Konwleage_top{args.knowledge_topk}_l0.99_dyn{args.dynamic_steps}"
                  f"_steps{args.steps}_pop{args.population}_temp{args.temperature:g}")
    if args.worker:
        worker(args, prior_name)
        return
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for dataset in args.datasets:
        kind = "nb201" if dataset.startswith("cifar") else "meta"
        print(f"{dataset}: seeds=0..{args.runs - 1}, steps={args.steps}, population={args.population}, temperature={args.temperature:g}", flush=True)
        print(f"  output: {RESULTS / kind / prior_name / dataset}", flush=True)
        if kind == "meta":
            cache = RESULTS / "meta" / args.cache_prior / dataset / "retrain_artifacts"
            print(f"  shared retrain cache: {cache} (exists={cache.is_dir()})", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", dataset,
                   "--runs", str(args.runs), "--steps", str(args.steps),
                   "--population", str(args.population), "--knowledge-topk", str(args.knowledge_topk),
                   "--dynamic-steps", str(args.dynamic_steps), "--temperature", str(args.temperature),
                   "--cache-prior", args.cache_prior]
        if not args.dry_run:
            subprocess.run(command, cwd=NAS_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
