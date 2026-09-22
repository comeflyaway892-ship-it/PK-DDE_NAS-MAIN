"""Table 6: paired, fixed-budget component ablations on ImageNet16-120."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

NAS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAS_ROOT))

from config import config as cfg
from prior_exp.predictor_prior import _normalise_posterior, d3pm_step
from prior_exp.prior_d3pm import PresetPriorMatrixScheduler
from prior_exp.prior_kernels import build_marginal_prior_kernel, build_prior_knowledge_kernel
from utils.mapping import ReScale
from utils.nb201_fitness import get_nb201_arch_str
from utils.select import select_population

DATASET = "ImageNet16-120"
VARIANTS = {
    "full": ("Full PK-DDE-NAS", True, True, True, True),
    "no_dynamic_prior": ("w/o dynamic prior", False, True, True, True),
    "no_fitness_guidance": ("w/o fitness guidance", True, False, True, True),
    "no_diffusion_consistency": ("w/o diffusion consistency", True, True, False, True),
    "no_posterior_sampling": ("w/o posterior sampling", True, True, True, False),
    "random_mutation": ("Random mutation", False, False, False, False),
}


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def connected(x):
    """Non-none input-to-output path; reported separately from API validity."""
    reachable = [True, False, False, False]
    for op, (source, dest) in zip(x, ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))):
        reachable[dest] |= reachable[source] and int(op) != 0
    return reachable[-1]


class BudgetEvaluator:
    """Charge every proposed row, including duplicate/invalid architectures."""

    def __init__(self, api, budget, hp="200"):
        self.api, self.budget, self.hp = api, budget, hp
        self.count = self.valid_count = self.connected_count = 0
        self.cache = {}
        self.best_score, self.best_arch, self.best_index = -math.inf, None, None

    def metric(self, index, split):
        # Match the existing NB201 workflow: search and report on x-test.
        info = self.api.arch2infos_dict[index][self.hp]
        value = float(info.get_metrics(DATASET, split, is_random=False)["accuracy"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {split} accuracy for architecture {index}")
        return value

    def evaluate(self, x):
        if self.count + len(x) > self.budget:
            raise RuntimeError("Architecture-evaluation budget exceeded")
        scores, valid, path_valid = [], 0, 0
        for row in x:
            key = tuple(int(v) for v in row)
            self.count += 1
            if key not in self.cache:
                in_space = len(key) == 6 and all(0 <= v < 5 for v in key)
                index = self.api.query_index_by_arch(get_nb201_arch_str(row)) if in_space else -1
                # A missing benchmark metric is an error, not an invalid graph.
                score = self.metric(index, "x-test") if index >= 0 else 0.0
                self.cache[key] = (index, score)
            index, score = self.cache[key]
            valid += int(index >= 0)
            path_valid += int(index >= 0 and connected(key))
            scores.append(score)
            if index >= 0 and score > self.best_score:
                self.best_score, self.best_arch, self.best_index = score, list(key), index
        self.valid_count += valid
        self.connected_count += path_valid
        accuracy = torch.tensor(scores, dtype=torch.float32)
        return accuracy, ReScale()(accuracy * 2.0), 100.0 * valid / len(x), 100.0 * path_valid / len(x)

    def final_test_accuracy(self):
        if self.best_index is None:
            raise RuntimeError("No valid architecture was evaluated")
        # The selected score was already evaluated; do not spend another query.
        return self.best_score


def make_scheduler(args, knowledge=None):
    if knowledge is None:
        kernel = build_marginal_prior_kernel(6, 5)
    else:
        pool = torch.cat(knowledge)
        # Each generation was already top-k filtered; retain the entire archive.
        kernel = build_prior_knowledge_kernel(
            6, 5, pool, torch.ones(len(pool)), topk=len(pool), lumda=args.prior_strength
        )
    return PresetPriorMatrixScheduler(
        args.steps, 5, kernel, eps=cfg.d3pm_eps,
        schedule=cfg.d3pm_schedule, cosine_s=cfg.d3pm_cosine_s,
    )


def estimate_x0(x, fitness, scheduler, t, args, guidance=True, consistency=True):
    n, d = x.shape
    eps = cfg.predictor_estimator_eps
    if guidance:
        base = torch.softmax((fitness - fitness.mean()) / args.temperature, dim=0).clamp_min(eps)
        base /= base.sum() + eps
    else:
        base = torch.full((n,), 1.0 / n)
    kernel = torch.ones((n, n))
    if consistency:
        log_kernel = torch.zeros((n, n))
        for edge in range(d):
            tokens = x[:, edge]
            matrix = scheduler.Qbar[t, edge].clamp_min(eps)
            log_kernel += matrix[tokens.view(1, n), tokens.view(n, 1)].log()
        kernel = torch.exp(log_kernel / d / args.sigma).clamp_min(eps)
    weights = kernel * base.view(1, n)
    weights /= weights.sum(dim=1, keepdim=True) + eps
    probs = torch.einsum("ij,jdk->idk", weights, F.one_hot(x, 5).float()).clamp_min(eps)
    probs /= probs.sum(dim=-1, keepdim=True) + eps
    return torch.multinomial(probs.reshape(n * d, 5), 1).view(n, d)


def posterior_argmax(x, x0, scheduler, t):
    n, d = x.shape
    edges = torch.arange(d).repeat(n)
    like = scheduler.Q[t - 1][edges, :, x.reshape(-1)]
    prior = scheduler.Qbar[t - 1][edges, x0.reshape(-1), :]
    probs = _normalise_posterior(like * prior, like, x.reshape(-1), 5, 1e-20)
    return probs.argmax(dim=-1).view_as(x)


def generate(x, fitness, scheduler, t, variant, args):
    if variant == "random_mutation":
        mask = torch.rand(x.shape) < args.mutation_rate
        # Mutated tokens must actually change; expected changed edges = 6*p.
        alternatives = (x + torch.randint(1, 5, x.shape)) % 5
        return torch.where(mask, alternatives, x)
    x0 = estimate_x0(
        x, fitness, scheduler, t, args,
        guidance=VARIANTS[variant][2], consistency=VARIANTS[variant][3],
    )
    if variant == "no_posterior_sampling":
        if args.no_posterior_mode == "direct_x0":
            return x0
        return posterior_argmax(x, x0, scheduler, t)
    return d3pm_step(x, x0, scheduler.Q, scheduler.Qbar, t)


def run_variant(args, api, variant, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    evaluator = BudgetEvaluator(api, args.population * (args.steps + 1), args.hp)
    x = torch.randint(0, 5, (args.population, 6))
    accuracy, fitness, _, _ = evaluator.evaluate(x)
    knowledge = [x[torch.topk(accuracy, args.knowledge_topk).indices].clone()]
    use_prior = variant != "random_mutation" and not (
        variant == "no_dynamic_prior" and args.no_dynamic_mode == "uniform"
    )
    scheduler = make_scheduler(args, knowledge if use_prior else None)
    history = []
    for step in range(args.steps):
        candidate = generate(x, fitness, scheduler, args.steps - step, variant, args)
        accuracy, candidate_fitness, validity, path_validity = evaluator.evaluate(candidate)
        if VARIANTS[variant][1] and step < args.dynamic_steps:
            knowledge.append(candidate[torch.topk(accuracy, args.knowledge_topk).indices].clone())
            scheduler = make_scheduler(args, knowledge)
        # Shared survivor selection, with aligned fitness and no extra API evaluation.
        x, fitness = select_population(
            x, fitness, candidate, candidate_fitness, args.population,
            elite_frac=cfg.select_elite_frac, eps=cfg.select_eps,
            fill_uniform=cfg.select_fill_uniform, return_fitness=True,
        )
        history.append({
            "step": step + 1, "evaluations": evaluator.count,
            "best_test_accuracy": evaluator.best_score,
            "candidate_validity_pct": validity, "candidate_connected_pct": path_validity,
            "candidate_unique_pct": 100.0 * len(torch.unique(candidate, dim=0)) / args.population,
        })
        if args.log_every and (step + 1) % args.log_every == 0:
            log(f"{variant} seed={seed} step={step + 1}/{args.steps} "
                f"evaluations={evaluator.count} best_test={evaluator.best_score:.4f}")
    assert evaluator.count == evaluator.budget
    search_seconds = time.perf_counter() - started
    test_accuracy = evaluator.final_test_accuracy()
    return {
        "variant": variant, "seed": seed, "dataset": DATASET,
        "search_split": "x-test", "report_split": "x-test",
        "steps": args.steps, "population": args.population,
        "architecture_evaluations": evaluator.count,
        "unique_architectures": len(evaluator.cache), "final_test_queries": 0,
        "best_arch": evaluator.best_arch, "best_index": evaluator.best_index,
        "best_test_accuracy": evaluator.best_score, "test_accuracy": test_accuracy,
        "validity_pct": 100.0 * evaluator.valid_count / evaluator.count,
        "connected_pct": 100.0 * evaluator.connected_count / evaluator.count,
        "search_seconds": search_seconds, "total_seconds": time.perf_counter() - started,
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(),
        "history": history,
    }


def summarize(results, output):
    rows = []
    for variant, flags in VARIANTS.items():
        runs = [r for r in results if r["variant"] == variant]
        if not runs:
            continue
        row = dict(variant=flags[0], dynamic_prior=flags[1], fitness_guidance=flags[2],
                   diffusion_consistency=flags[3], posterior_sampling=flags[4], runs=len(runs))
        for field in ("test_accuracy", "best_test_accuracy", "validity_pct", "connected_pct",
                      "search_seconds", "unique_architectures"):
            values = [r[field] for r in runs]
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Variant | Dynamic prior | Fitness guidance | Diffusion consistency | Posterior sampling | Performance (%) | Validity (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    def display(row, field):
        mean, std = row[field + "_mean"], row[field + "_std"]
        return f"{mean:.2f}" if std is None else f"{mean:.2f} ± {std:.2f}"
    for row in rows:
        flags = ["Yes" if row[k] else "No" for k in
                 ("dynamic_prior", "fitness_guidance", "diffusion_consistency", "posterior_sampling")]
        lines.append("| " + " | ".join([row["variant"], *flags, display(row, "test_accuracy"),
                                        display(row, "validity_pct")]) + " |")
    lines.extend(["", "Performance: highest test accuracy among evaluated architectures (x-test search and selection); sample SD across search seeds.",
                  "Validity: API-queryable candidates / all evaluated candidates, including duplicates.",
                  "See settings.json for ablation definitions and per-run evaluation budget."])
    (output / "table6.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--knowledge-topk", type=int, default=6)
    parser.add_argument("--dynamic-steps", type=int, default=10)
    parser.add_argument("--prior-strength", type=float, default=0.99)
    parser.add_argument("--no-dynamic-mode", choices=("frozen", "uniform"), default="uniform")
    parser.add_argument("--no-posterior-mode", choices=("argmax", "direct_x0"), default="direct_x0")
    parser.add_argument("--mutation-rate", type=float, default=1.0 / 6)
    parser.add_argument("--temperature", type=float, default=7.0)
    parser.add_argument("--sigma", type=float, default=cfg.predictor_sigma)
    parser.add_argument("--hp", choices=("12", "200"), default="200")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--api-path", type=Path, default=NAS_ROOT / cfg.nb201_api_path)
    parser.add_argument("--output", type=Path, default=NAS_ROOT / "ablation" / "results" /
                        datetime.now().strftime("components_imagenet_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.steps, args.population, args.runs, args.knowledge_topk, args.threads) <= 0:
        parser.error("steps/population/runs/knowledge-topk/threads must be positive")
    if not 0 <= args.seed_start < 2**32 or args.seed_start + args.runs > 2**32:
        parser.error("seeds must be in [0, 2**32)")
    if args.knowledge_topk > args.population or not 0 <= args.dynamic_steps <= args.steps:
        parser.error("knowledge-topk <= population and 0 <= dynamic-steps <= steps required")
    if not 0 <= args.prior_strength <= 1 or not 0 <= args.mutation_rate <= 1:
        parser.error("prior-strength and mutation-rate must be in [0,1]")
    if not math.isfinite(args.temperature) or not math.isfinite(args.sigma) or min(args.temperature, args.sigma) <= 0:
        parser.error("temperature and sigma must be finite and positive")
    if args.log_every < 0 or len(set(args.variants)) != len(args.variants):
        parser.error("log-every must be nonnegative; variants must not repeat")
    return args


def settings(args):
    result = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("dry_run", "output")}
    result.update(dataset=DATASET, search_split="x-test", report_split="x-test",
                  architecture_evaluations_per_run=args.population * (args.steps + 1),
                  final_test_queries_per_run=0, validity="API-queryable / all evaluated rows",
                  guidance_scope="estimator fitness weights; shared fitness-based survivor selection and prior ranking",
                  torch_version=str(torch.__version__),
                  shared_config={key: getattr(cfg, key) for key in (
                      "d3pm_eps", "d3pm_schedule", "d3pm_cosine_s", "predictor_estimator_eps",
                      "select_elite_frac", "select_eps", "select_fill_uniform")})
    return result


def main():
    args = parse_args()
    metadata = settings(args)
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return
    if not args.api_path.is_file():
        raise FileNotFoundError(args.api_path)
    # New directories only: reruns cannot mix settings or overwrite old results.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.set_num_threads(args.threads)
    log(f"Output: {args.output.resolve()}")
    log(f"Loading API once: {args.api_path}")
    load_start = time.perf_counter()
    from nas_201_api import NASBench201API
    api = NASBench201API(str(args.api_path), verbose=False)
    load_seconds = time.perf_counter() - load_start
    log(f"API loaded in {load_seconds:.2f}s")
    results = []
    with (args.output / "runs.jsonl").open("w", encoding="utf-8") as f:
        for seed in range(args.seed_start, args.seed_start + args.runs):
            for variant in args.variants:
                log(f"START {variant} seed={seed}")
                result = run_variant(args, api, variant, seed)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results.append(result)
                summarize(results, args.output)
                log(f"DONE {variant} seed={seed} test={result['test_accuracy']:.4f} "
                    f"validity={result['validity_pct']:.2f}% time={result['search_seconds']:.2f}s")
    (args.output / "execution.json").write_text(json.dumps({"api_load_seconds": load_seconds,
                                                          "completed_runs": len(results)}, indent=2) + "\n")
    print(summarize(results, args.output))


if __name__ == "__main__":
    main()
