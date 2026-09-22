"""Run prior-aware discrete diffusion experiments on NAS-Bench-301."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import nb301_hyper_params_setting, nb301_surrogate_path
from main import run_searches, _atomic_write_json, _resolve_from_module
from utils.NB301 import load_nb301_surrogate
from utils.seeds import parse_seed_list


def main(args):
    settings = nb301_hyper_params_setting
    seeds = parse_seed_list(args.seeds)
    predictor = load_nb301_surrogate(_resolve_from_module(args.surrogate_path))
    records = run_searches(
        predictor=predictor,
        seeds=seeds,
        num_step=args.num_step or settings["num_step"],
        population_num=args.population_num or settings["population_num"],
        plot=args.plot,
        save_dir=_resolve_from_module(settings["save_dir"]),
        prior_type=args.prior_type,
        prior_kwargs={
            "tau": args.tau,
            "shuffle_seed": args.shuffle_seed,
            "topk": args.topk,
            "lumda": args.lumda,
            "dynamic_steps": args.dynamic_steps,
        },
    )
    results_file = _resolve_from_module(args.results_file)
    _atomic_write_json(
        results_file,
        {
            "seeds": seeds,
            "prior_type": args.prior_type,
            "prior_kwargs": {
                "tau": args.tau,
                "shuffle_seed": args.shuffle_seed,
                "topk": args.topk,
                "lumda": args.lumda,
                "dynamic_steps": args.dynamic_steps,
            },
            "architectures": records,
        },
    )
    print("Completed {} prior-aware NAS-Bench-301 searches; saved {}".format(len(records), results_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate-path", default=nb301_surrogate_path)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-step", type=int, default=None)
    parser.add_argument("--population-num", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--prior-type",
        choices=("uniform", "structural", "shuffled_structural", "prior_knowledge"),
        default="structural",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lumda", type=float, default=1.0)
    parser.add_argument("--dynamic-steps", type=int, default=0)
    parser.add_argument(
        "--results-file",
        default="./results/nb301_prior/top1_architectures.json",
    )
    main(parser.parse_args())
