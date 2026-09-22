import argparse
import itertools
import os
import sys


PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRANS_ROOT = os.path.dirname(PRIOR_EXP_ROOT)
REPO_ROOT = os.path.dirname(TRANS_ROOT)
for path in (REPO_ROOT, TRANS_ROOT):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
os.chdir(TRANS_ROOT)

from config.config import hyper_params_setting
from prior_exp.experiments_prior import main_exp_prior
from TransNASBench101.api import TransNASBenchAPI as API


TASKS = (
    "class_scene",
    "class_object",
    "room_layout",
    "jigsaw",
    "segmentsemantic",
    "normal",
    "autoencoder",
)
SEARCH_SPACES = ("macro", "micro")
PHASES = {
    "default": {
        "num_step": None,
        "population_num": None,
        "knowledge_topk": (5, 10, 15, 20),
        "knowledge_dynamic_steps": (10, 20, 30, 40),
    },
    "reduced": {
        "num_step": 30,
        "population_num": 20,
        "knowledge_topk": (5, 6, 8, 10),
        "knowledge_dynamic_steps": (10, 20, 30),
    },
}


def _prior_configs(phase):
    yield {
        "prior_type": "structural",
        "prior_kwargs": {"tau": 1.0},
        "base_name": "structural_tau1",
    }
    yield {
        "prior_type": "shuffled_structural",
        "prior_kwargs": {"tau": 1.0, "shuffle_seed": 0},
        "base_name": "shuffled_s0_tau1",
    }
    for topk, dynamic_steps in itertools.product(
        phase["knowledge_topk"],
        phase["knowledge_dynamic_steps"],
    ):
        yield {
            "prior_type": "prior_Konwleage",
            "prior_kwargs": {
                "topk": topk,
                "lumda": 0.99,
                "dynamic_steps": dynamic_steps,
            },
            "base_name": f"prior_Konwleage_top{topk}_l0.99_dyn{dynamic_steps}",
        }


def build_sweep_specs(phase_names=("default", "reduced")):
    specs = []
    for phase_name in phase_names:
        phase = PHASES[phase_name]
        for search_space in SEARCH_SPACES:
            for task in TASKS:
                defaults = hyper_params_setting[search_space][task]
                num_step = phase["num_step"] if phase["num_step"] is not None else defaults["num_step"]
                population_num = (
                    phase["population_num"]
                    if phase["population_num"] is not None
                    else defaults["population_num"]
                )
                for prior in _prior_configs(phase):
                    spec = dict(prior)
                    spec.update(
                        {
                            "phase": phase_name,
                            "task": task,
                            "search_space": search_space,
                            "num_step": int(num_step),
                            "population_num": int(population_num),
                        }
                    )
                    spec["prior_name"] = (
                        f"sweep_step{spec['num_step']}_pop{spec['population_num']}_{spec['base_name']}"
                    )
                    specs.append(spec)
    return specs


def _format_params(spec):
    parts = [
        f"task={spec['task']}",
        f"space={spec['search_space']}",
        f"num_step={spec['num_step']}",
        f"population_num={spec['population_num']}",
        f"prior={spec['prior_type']}",
    ]
    if spec["prior_type"] == "prior_Konwleage":
        parts.extend(
            [
                f"topk={spec['prior_kwargs']['topk']}",
                f"lumda={spec['prior_kwargs']['lumda']:g}",
                f"dynamic_steps={spec['prior_kwargs']['dynamic_steps']}",
            ]
        )
    return " ".join(parts)


def _summary_header(num_seeds):
    return f"# TransNASBench101 prior sweep; seeds={num_seeds}; std=population(ddof=0)"


def _prepare_summary_file(path, num_seeds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = _summary_header(num_seeds)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header + "\n")
        return set()

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if lines and lines[0] != header:
        raise ValueError(
            f"Summary file header does not match this run: expected {header!r}, got {lines[0]!r}"
        )
    return {
        line.split(" max_acc=", 1)[0]
        for line in lines
        if not line.startswith("#") and " max_acc=" in line
    }


def _append_summary(path, params, result):
    line = (
        f"{params} max_acc={result['max_acc_mean']:.4f}"
        f"+/-{result['max_acc_std']:.4f}\n"
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def run_sweep(specs, *, num_seeds, summary_file, plot_results=False, dry_run=False):
    summary_file = os.path.abspath(summary_file)
    if dry_run:
        for index, spec in enumerate(specs, start=1):
            print(f"[{index}/{len(specs)}] {_format_params(spec)}")
        print(f">>> Dry run: {len(specs)} configurations, {len(specs) * num_seeds} seed runs.")
        return

    completed_params = _prepare_summary_file(summary_file, num_seeds)
    seeds = list(range(num_seeds))
    api = API("transnas-bench_v10141024.pth")
    for index, spec in enumerate(specs, start=1):
        params = _format_params(spec)
        if params in completed_params:
            print(f"[{index}/{len(specs)}] Summary exists, skip: {params}")
            continue

        print(f"\n[{index}/{len(specs)}] Start: {params}")
        result = main_exp_prior(
            task=spec["task"],
            search_space=spec["search_space"],
            prior_type=spec["prior_type"],
            prior_name=spec["prior_name"],
            prior_kwargs=spec["prior_kwargs"],
            seed_list=seeds,
            api=api,
            num_step=spec["num_step"],
            population_num=spec["population_num"],
            plot_results=plot_results,
        )
        if result["completed_seeds"] != num_seeds:
            raise RuntimeError(
                f"Expected {num_seeds} completed seeds for {params}, "
                f"got {result['completed_seeds']}"
            )
        _append_summary(summary_file, params, result)
        completed_params.add(params)
        print(f">>> Summary appended to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run the full TransNASBench101 prior sweep")
    parser.add_argument(
        "--phase",
        choices=["all", "default", "reduced"],
        default="all",
        help="run both phases or only one phase",
    )
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument(
        "--summary-file",
        default=os.path.join(PRIOR_EXP_ROOT, "results", "prior_sweep_summary.txt"),
    )
    parser.add_argument("--plot-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_seeds <= 0:
        raise ValueError(f"--num-seeds must be positive, got {args.num_seeds}")
    phase_names = ("default", "reduced") if args.phase == "all" else (args.phase,)
    specs = build_sweep_specs(phase_names)
    run_sweep(
        specs,
        num_seeds=args.num_seeds,
        summary_file=args.summary_file,
        plot_results=args.plot_results,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
