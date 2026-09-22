"""Command-line entry point for surrogate-only NAS-Bench-301 search."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    nb301_hyper_params_setting,
    nb301_surrogate_path,
    nb301_with_noise,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from evo_diff import evo_diff
from utils.NB301 import genotype_to_dict, load_nb301_surrogate, tokens_to_genotype
from utils.seeds import DEFAULT_SEEDS, parse_seed_list


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_from_module(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def run_searches(
    predictor,
    seeds,
    num_step,
    population_num,
    plot,
    save_dir,
    prior_type="uniform",
    prior_kwargs=None,
):
    """Run one surrogate search per seed and return every Top-1 record."""
    top1_records = []
    for seed in seeds:
        set_random_seed(seed)
        print(">>> Running CIFAR-10 surrogate search with seed {}".format(seed))
        max_accuracy, duration, uniqueness, architectures = evo_diff(
            predictor=predictor,
            num_step=num_step,
            population_num=population_num,
            seed=seed,
            plot_results=plot,
            save_dir=str(save_dir),
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
            prior_type=prior_type,
            prior_kwargs=prior_kwargs,
        )
        best_architecture = architectures[0]
        best_prediction = predictor.predict_tokens(best_architecture)
        for architecture in architectures[1:]:
            prediction = predictor.predict_tokens(architecture)
            if prediction > best_prediction:
                best_architecture = architecture
                best_prediction = prediction

        tokens = best_architecture.detach().cpu().long().tolist()
        record = {
            "search_seed": seed,
            "prior_type": prior_type,
            "tokens": tokens,
            "genotype": genotype_to_dict(tokens_to_genotype(tokens)),
            "predicted_accuracy": float(best_prediction),
            "max_trace_accuracy": float(max_accuracy),
            "duration_seconds": float(duration),
            "final_uniqueness": float(uniqueness),
        }
        top1_records.append(record)
        print(
            ">>> seed={}, Top-1 prediction={:.4f}, duration={:.2f}s, "
            "final uniqueness={:.2f}".format(
                seed, best_prediction, duration, uniqueness
            )
        )
        print(">>> Top-1 tokens: {}".format(tokens))
        print(">>> Top-1 genotype: {}".format(record["genotype"]))
    return top1_records


def main(args):
    settings = nb301_hyper_params_setting
    seeds = args.seeds
    num_step = settings["num_step"] if args.num_step is None else args.num_step
    population_num = (
        settings["population_num"]
        if args.population_num is None
        else args.population_num
    )
    surrogate_path = _resolve_from_module(args.surrogate_path)
    save_dir = _resolve_from_module(settings["save_dir"])
    results_file = _resolve_from_module(args.results_file)

    print(">>> Loading NAS-Bench-301 performance surrogate from {}".format(surrogate_path))
    predictor = load_nb301_surrogate(
        surrogate_path, with_noise=args.with_noise
    )

    records = run_searches(
        predictor=predictor,
        seeds=seeds,
        num_step=num_step,
        population_num=population_num,
        plot=args.plot,
        save_dir=save_dir,
        prior_type=args.prior_type,
        prior_kwargs={
            "tau": args.prior_tau,
            "shuffle_seed": args.shuffle_seed,
            "topk": args.knowledge_topk,
            "lumda": args.knowledge_lumda,
            "dynamic_steps": args.knowledge_dynamic_steps,
        },
    )
    _atomic_write_json(results_file, {"seeds": seeds, "architectures": records})
    maximum_accuracies = [record["predicted_accuracy"] for record in records]
    durations = [record["duration_seconds"] for record in records]

    print(
        ">>> {} run(s): mean max prediction={:.4f}, mean duration={:.2f}s".format(
            len(seeds), float(np.mean(maximum_accuracies)), float(np.mean(durations))
        )
    )
    print(">>> Saved all Top-1 architectures to {}".format(results_file))


def build_parser():
    parser = argparse.ArgumentParser(
        description="DDE-NAG search on the NAS-Bench-301 performance surrogate"
    )
    parser.add_argument(
        "--surrogate-path",
        default=nb301_surrogate_path,
        help="official xgb NAS-Bench-301 ensemble directory",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list(DEFAULT_SEEDS),
        help="explicit search seed list separated by English commas (default: 0-9)",
    )
    parser.add_argument("--num-step", type=int, default=None)
    parser.add_argument("--population-num", type=int, default=None)
    parser.add_argument(
        "--with-noise",
        action="store_true",
        default=nb301_with_noise,
        help="sample noisy ensemble predictions (deterministic mean is the default)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--prior-type",
        choices=("uniform", "structural", "shuffled_structural", "prior_knowledge"),
        default="uniform",
        help="forward D3PM prior; prior_knowledge uses top-k initial/generated candidates",
    )
    parser.add_argument("--prior-tau", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--knowledge-topk", type=int, default=10)
    parser.add_argument("--knowledge-lumda", type=float, default=1.0)
    parser.add_argument("--knowledge-dynamic-steps", type=int, default=0)
    parser.add_argument(
        "--results-file",
        default="./results/nb301_surrogate/cifar10/top1_architectures.json",
        help="JSON file receiving every search seed's Top-1 architecture",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
