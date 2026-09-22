import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PRIOR = REPO_ROOT / "NAS_Bench_201" / "prior_exp" / "main_prior.py"
DATASETS = ("pets", "aircraft")


def build_command(dataset, runs):
    return [
        sys.executable,
        str(MAIN_PRIOR),
        "--dataset",
        dataset,
        "--prior-type",
        "prior_Konwleage",
        "--knowledge-topk",
        "20",
        "--knowledge-dynamic-steps",
        "40",
        "--runs",
        str(runs),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run the 100-seed pets and aircraft prior_Konwleage experiments."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="CUDA device id exposed to each experiment, for example 0 or 1.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to run sequentially.",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for dataset in args.datasets:
        command = build_command(dataset, args.runs)
        visible_gpu = env.get("CUDA_VISIBLE_DEVICES", "<inherited>")
        print(
            f">>> dataset={dataset} seeds=0..{args.runs - 1} "
            f"topk=20 dynamic_steps=40 CUDA_VISIBLE_DEVICES={visible_gpu}",
            flush=True,
        )
        printable_command = " ".join(shlex.quote(part) for part in command)
        print(f">>> {printable_command}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
