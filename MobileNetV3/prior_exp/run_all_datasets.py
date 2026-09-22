import argparse
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PRIOR = REPO_ROOT / "MobileNetV3" / "prior_exp" / "main_prior.py"
DATASETS = ("cifar10", "cifar100", "aircraft", "pets")
SEARCH_RUNS = 3
GPU_MEMORY_LAUNCHER_ENV = "DDE_NAG_GPU_MEMORY_LAUNCHER"


def _conflicts_with_fixed_arguments(arguments):
    fixed_options = ("--dataset", "--runs", "--gpu")
    return [
        argument
        for argument in arguments
        if any(argument == option or argument.startswith(option + "=") for option in fixed_options)
    ]


def _run_main_prior_in_process(command):
    """Run main_prior.py without losing a launcher-owned PyTorch CUDA cache."""
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        sys.argv = command[1:]
        os.chdir(REPO_ROOT)
        runpy.run_path(str(MAIN_PRIOR), run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


def main():
    default_gpu = os.environ.get("DDE_NAG_GPU", os.environ.get("EDNAG_GPU", "0"))
    parser = argparse.ArgumentParser(
        description="Run three search seeds serially on all four MobileNetV3 datasets."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=default_gpu,
        help=(
            "logical GPU index when CUDA_VISIBLE_DEVICES is set; otherwise the "
            "physical GPU id to expose"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Optional subset; the default order is cifar10, cifar100, aircraft, pets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded_args = parser.parse_known_args()
    in_gpu_memory_launcher = os.environ.get(GPU_MEMORY_LAUNCHER_ENV) == "1"
    inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if inherited_visible is not None:
        print(
            f">>> inherited CUDA_VISIBLE_DEVICES={inherited_visible!r}; "
            f"--gpu {args.gpu} is a logical index within that visible list",
            flush=True,
        )
    else:
        print(
            f">>> CUDA_VISIBLE_DEVICES is unset; --gpu {args.gpu} selects the physical GPU",
            flush=True,
        )

    conflicts = _conflicts_with_fixed_arguments(forwarded_args)
    if conflicts:
        parser.error(
            "dataset, gpu, and search seeds are controlled by this runner; "
            f"remove: {' '.join(conflicts)}"
        )

    for dataset in args.datasets:
        command = [
            sys.executable,
            str(MAIN_PRIOR),
            "--dataset",
            dataset,
            "--gpu",
            args.gpu,
            "--runs",
            str(SEARCH_RUNS),
        ] + forwarded_args
        printable_command = " ".join(shlex.quote(part) for part in command)
        print(f">>> dataset={dataset} search_seeds=0..{SEARCH_RUNS - 1}", flush=True)
        print(f">>> {printable_command}", flush=True)
        if not args.dry_run:
            if in_gpu_memory_launcher:
                print(
                    ">>> GPU memory launcher detected; executing in the shared process.",
                    flush=True,
                )
                _run_main_prior_in_process(command)
            else:
                subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
