"""Run one dynamic-prior configuration on all TransNASBench101 tasks/spaces."""

import argparse
import contextlib
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

TRANS_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("class_scene", "class_object", "room_layout", "jigsaw", "segmentsemantic", "normal", "autoencoder")
SPACES = ("macro", "micro")


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def stamp(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=list(SPACES))
    parser.add_argument("--run-name", help="Use the same name to resume an interrupted batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.population, args.topk, args.runs, args.threads) <= 0:
        parser.error("steps/population/topk/runs/threads must be positive")
    if args.topk > args.population or not 0 <= args.dynamic_steps <= args.steps:
        parser.error("topk <= population and 0 <= dynamic-steps <= steps required")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        parser.error("run-name must be a single directory name")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.spaces)) != len(args.spaces):
        parser.error("tasks and spaces must not repeat")
    return args


def run(args, name, output, metadata):
    os.chdir(TRANS_ROOT)
    sys.path.insert(0, str(TRANS_ROOT))
    from config import config as cfg
    # Apply before experiments_prior imports these settings by name.
    cfg.predictor_temperature = args.temperature
    for space in args.spaces:
        for task in args.tasks:
            cfg.hyper_params_setting[space][task].update(
                num_step=args.steps, population_num=args.population, seed=list(range(args.runs))
            )
    import torch
    torch.set_num_threads(args.threads)
    from prior_exp import experiments_prior
    from TransNASBench101.api import TransNASBenchAPI
    if experiments_prior.predictor_temperature != args.temperature:
        raise RuntimeError("Temperature override did not reach the experiment module")
    effective = dict(metadata, shared_config={k: getattr(cfg, k) for k in (
        "d3pm_eps", "d3pm_schedule", "d3pm_cosine_s", "predictor_estimator_eps",
        "predictor_sigma", "select_elite_frac", "select_eps", "select_fill_uniform")})
    (output / "effective_settings.json").write_text(json.dumps(effective, indent=2) + "\n")
    stamp(f"Loading API; settings={metadata}")
    api = TransNASBenchAPI(str(TRANS_ROOT / "transnas-bench_v10141024.pth"))
    rows = []
    for space in args.spaces:
        for task in args.tasks:
            stamp(f"START {space}/{task}, seeds=0..{args.runs - 1}, temperature={args.temperature}")
            result = experiments_prior.main_exp_prior(
                task=task, search_space=space, prior_type="prior_Konwleage", prior_name=name,
                prior_kwargs=dict(topk=args.topk, lumda=0.99, dynamic_steps=args.dynamic_steps),
                seed_list=list(range(args.runs)), api=api, num_step=args.steps,
                population_num=args.population, plot_results=False,
            )
            if result["completed_seeds"] != args.runs:
                raise RuntimeError(f"Incomplete results: {space}/{task}: {result}")
            row = dict(search_space=space, task=task, **result)
            row["max_acc_std_sample"] = (
                result["max_acc_std"] * math.sqrt(args.runs / (args.runs - 1)) if args.runs > 1 else None
            )
            rows.append(row)
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            with (output / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            stamp(f"DONE {space}/{task}: {result['max_acc_mean']:.4f} +/- "
                  f"{result['max_acc_std']:.4f} (population SD); completed={result['completed_seeds']}")
    return sum(row["completed_seeds"] for row in rows)


def main():
    args = parse_args()
    name = args.run_name or (
        f"prior_Konwleage_top{args.topk}_l0.99_dyn{args.dynamic_steps}_steps{args.steps}"
        f"_pop{args.population}_temp{args.temperature:g}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output = TRANS_ROOT / "prior_exp" / "results" / name
    metadata = {k: v for k, v in vars(args).items() if k not in ("dry_run", "run_name")}
    metadata.update(prior_type="prior_Konwleage", prior_strength=0.99, seed_start=0,
                    total_seed_runs=len(args.tasks) * len(args.spaces) * args.runs,
                    metric_protocol="Existing transnasbench101_fitness task metrics, mode=best; room_layout minimized")
    if args.dry_run:
        print(json.dumps(dict(output=str(output), **metadata), indent=2))
        return
    if not (TRANS_ROOT / "transnas-bench_v10141024.pth").is_file():
        raise FileNotFoundError("TransNASBench101 benchmark file is missing")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "settings.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Existing run settings differ; use a new run-name")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output directory has no settings manifest")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    started = time.perf_counter()
    status = dict(status="running", started_at=datetime.now().astimezone().isoformat(), pid=os.getpid())
    status_path = output / "status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    with (output / "run.log").open("a", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            stamp(f"Output directory: {output}")
            try:
                status["completed_seed_runs"] = run(args, name, output, metadata)
                status["status"] = "complete"
                stamp(f"Completed {status['completed_seed_runs']} seed runs")
            except BaseException as error:
                status.update(status="failed", error=repr(error))
                traceback.print_exc()
                raise
            finally:
                status.update(finished_at=datetime.now().astimezone().isoformat(),
                              wall_seconds=time.perf_counter() - started)
                status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
