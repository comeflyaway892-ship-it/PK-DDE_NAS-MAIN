"""Table 7: compare six categorical transition kernels on ImageNet16-120."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_component_ablation as core
from transition_kernels import KERNELS, MASK, NUM_EDGES, NUM_OPS, OP_NAMES
from transition_kernels import build_kernel, marginal_probabilities, per_edge_scales
from prior_exp.predictor_prior import _normalise_posterior


def make_scheduler(args, kind, knowledge=None):
    matrix = build_kernel(
        kind, gaussian_sigma=args.gaussian_sigma, distance_tau=args.distance_tau,
        fixed_probs=args.fixed_probs, knowledge=knowledge, prior_strength=args.prior_strength,
    )
    return core.PresetPriorMatrixScheduler(
        args.steps, matrix.shape[-1], matrix, eps=core.cfg.d3pm_eps,
        schedule=core.cfg.d3pm_schedule, cosine_s=core.cfg.d3pm_cosine_s,
    )


def estimate_clean_tokens(population, observed, fitness, scheduler, t, args):
    """Clean x0 distribution uses only the five operations, even for mask kernels."""
    if not ((population >= 0) & (population < NUM_OPS)).all():
        raise ValueError("The evaluated population must contain clean operations")
    n, d = population.shape
    eps = core.cfg.predictor_estimator_eps
    base = torch.softmax((fitness - fitness.mean()) / args.temperature, dim=0).clamp_min(eps)
    base /= base.sum() + eps
    log_kernel = torch.zeros((n, n))
    for edge in range(d):
        matrix = scheduler.Qbar[t, edge].clamp_min(eps)
        source, dest = population[:, edge], observed[:, edge]
        log_kernel += matrix[source.view(1, n), dest.view(n, 1)].log()
    kernel = torch.exp(log_kernel / d / args.sigma).clamp_min(eps)
    weights = kernel * base.view(1, n)
    weights /= weights.sum(dim=1, keepdim=True) + eps
    probs = torch.einsum("ij,jdk->idk", weights, F.one_hot(population, NUM_OPS).float()).clamp_min(eps)
    probs /= probs.sum(dim=-1, keepdim=True) + eps
    return torch.multinomial(probs.reshape(n * d, NUM_OPS), 1).view(n, d)


def sample_posterior(observed, predicted, scheduler, t):
    """Same posterior/fallback as Table 6, with explicit singular-mass diagnostics."""
    edges = torch.arange(observed.shape[1]).repeat(observed.shape[0])
    flat = observed.reshape(-1)
    like = scheduler.Q[t - 1][edges, :, flat]
    prior = scheduler.Qbar[t - 1][edges, predicted.reshape(-1), :]
    unnormalized = like * prior
    zero_mass = int((unnormalized.sum(-1) <= 1e-20).sum())
    posterior = _normalise_posterior(unnormalized, like, flat, scheduler.K, 1e-20)
    output = torch.multinomial(posterior, 1).view_as(observed)
    return output, zero_mass


def generate(population, fitness, scheduler, t, args):
    observed = population
    if args.proposal_mode == "forward_reverse":
        edges = torch.arange(NUM_EDGES).repeat(len(population))
        probs = scheduler.Qbar[t][edges, population.reshape(-1), :]
        observed = torch.multinomial(probs, 1).view_as(population)
    predicted = estimate_clean_tokens(population, observed, fitness, scheduler, t, args)
    raw, zero_mass = sample_posterior(observed, predicted, scheduler, t)
    raw_valid = ((raw >= 0) & (raw < NUM_OPS)).all(dim=1)
    mask = raw == MASK
    if mask.any() and args.proposal_mode == "reverse_only":
        # Never silently turn a latent mask into a real operation in Table-6 mode.
        raise RuntimeError("Unexpected mask from a clean reverse-only population")
    # The explicit mask-aware adapter in forward_reverse mode completes remaining
    # latent masks with the same predicted x0 tokens before architecture evaluation.
    candidate = torch.where(mask, predicted, raw) if scheduler.K == NUM_OPS + 1 else raw
    return candidate, {
        "raw_valid_candidates": int(raw_valid.sum()),
        "observed_mask_tokens": int((observed == MASK).sum()),
        "completed_mask_tokens": int(mask.sum()),
        "posterior_zero_mass_tokens": zero_mass,
    }


def run_kernel(args, api, kind, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    evaluator = core.BudgetEvaluator(api, args.population * (args.steps + 1), args.hp)
    x = torch.randint(0, NUM_OPS, (args.population, NUM_EDGES))
    accuracy, fitness, _, _ = evaluator.evaluate(x)
    initial_best = evaluator.best_score
    initial_population = x.tolist()
    knowledge = [x[torch.topk(accuracy, args.knowledge_topk).indices].clone()]
    scheduler = make_scheduler(args, kind, knowledge)
    initial_kernel = scheduler.prior_kernel.tolist()
    history, prior_updates = [], []
    raw_valid_count = args.population
    totals = dict(observed_mask_tokens=0, completed_mask_tokens=0, posterior_zero_mass_tokens=0)
    for step in range(args.steps):
        candidate, diagnostics = generate(x, fitness, scheduler, args.steps - step, args)
        accuracy, candidate_fitness, validity, connected = evaluator.evaluate(candidate)
        if kind == "dynamic_marginal" and step < args.dynamic_steps:
            knowledge.append(candidate[torch.topk(accuracy, args.knowledge_topk).indices].clone())
            scheduler = make_scheduler(args, kind, knowledge)
            prior_updates.append({"after_step": step + 1, "rho": scheduler.prior_kernel[:, 0, :].tolist()})
        x, fitness = core.select_population(
            x, fitness, candidate, candidate_fitness, args.population,
            elite_frac=core.cfg.select_elite_frac, eps=core.cfg.select_eps,
            fill_uniform=core.cfg.select_fill_uniform, return_fitness=True,
        )
        raw_valid_count += diagnostics["raw_valid_candidates"]
        for key in totals:
            totals[key] += diagnostics[key]
        history.append(dict(
            step=step + 1, evaluations=evaluator.count, best_test_accuracy=evaluator.best_score,
            candidate_validity_pct=validity, candidate_connected_pct=connected,
            candidate_unique_pct=100.0 * len(torch.unique(candidate, dim=0)) / args.population,
            **diagnostics,
        ))
        if args.log_every and ((step + 1) % args.log_every == 0 or step + 1 == args.steps):
            core.log(f"{kind} seed={seed} step={step + 1}/{args.steps} evaluations={evaluator.count} "
                     f"best_test={evaluator.best_score:.4f} validity={validity:.2f}%")
    assert evaluator.count == evaluator.budget
    return dict(
        kernel=kind, seed=seed, dataset=core.DATASET, search_split="x-test", report_split="x-test",
        steps=args.steps, population=args.population, proposal_mode=args.proposal_mode,
        architecture_evaluations=evaluator.count, final_test_queries=0,
        unique_architectures=len(evaluator.cache), initial_best_test_accuracy=initial_best,
        initial_population=initial_population, best_arch=evaluator.best_arch, best_index=evaluator.best_index,
        test_accuracy=evaluator.final_test_accuracy(), validity_pct=100 * evaluator.valid_count / evaluator.count,
        raw_validity_pct=100 * raw_valid_count / evaluator.count,
        connected_pct=100 * evaluator.connected_count / evaluator.count,
        search_seconds=time.perf_counter() - started,
        started_at=started_at, finished_at=datetime.now().astimezone().isoformat(),
        initial_kernel=initial_kernel, final_kernel=scheduler.prior_kernel.tolist(),
        prior_updates=prior_updates, history=history, **totals,
    )


def summarize(results, output, args):
    rows = []
    for kind, (label, semantics) in KERNELS.items():
        runs = [r for r in results if r["kernel"] == kind]
        if not runs:
            continue
        row = dict(kernel=kind, label=label, applicable_token_semantics=semantics, runs=len(runs),
                   proposal_mode=args.proposal_mode)
        for field in ("test_accuracy", "validity_pct", "raw_validity_pct", "unique_architectures",
                      "search_seconds", "posterior_zero_mass_tokens", "completed_mask_tokens"):
            values = [r[field] for r in runs]
            row[field + "_mean"] = statistics.mean(values)
            row[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| Kernel | Applicable token semantics | Mean performance (%) | Standard deviation | Validity (%) |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        std = row["test_accuracy_std"]
        sd = "—" if std is None else f"{std:.2f}"
        lines.append(f"| {row['label']} | {row['applicable_token_semantics']} | "
                     f"{row['test_accuracy_mean']:.2f} | {sd} | {row['validity_pct_mean']:.2f} |")
    lines += ["", f"Proposal mode: {args.proposal_mode}. SD is sample SD across search seeds (ddof=1).",
              "Performance uses x-test for both search and reporting. Validity is API-queryable evaluated candidates / all evaluated rows (duplicates included).",
              "NB201 operations are nominal; Gaussian/distance kernels use the declared index order as a control, not a natural ordinal semantics."]
    if args.proposal_mode == "reverse_only":
        lines.append("Absorbing/mask has no forward corruption in this protocol: unmasked tokens cannot change under its reverse transition; it is a degenerate initial-population control.")
    else:
        lines.append("All kernels use forward corruption. Remaining latent masks are completed from predicted x0 before evaluation; raw_validity_pct and mask completion counts are reported separately in summary.csv.")
    (output / "table7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", nargs="+", choices=list(KERNELS), default=list(KERNELS))
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--knowledge-topk", type=int, default=6)
    parser.add_argument("--dynamic-steps", type=int, default=10)
    parser.add_argument("--prior-strength", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=7.0)
    parser.add_argument("--sigma", type=float, default=core.cfg.predictor_sigma,
                        help="Estimator kernel scale; distinct from --gaussian-sigma")
    parser.add_argument("--gaussian-sigma", type=float, nargs="+", default=[1.0])
    parser.add_argument("--distance-tau", type=float, nargs="+", default=[1.0])
    parser.add_argument("--fixed-probs", type=float, nargs="+", default=[0.1, 0.1, 0.3, 0.3, 0.2])
    parser.add_argument("--proposal-mode", choices=("reverse_only", "forward_reverse"), default="reverse_only")
    parser.add_argument("--hp", choices=("12", "200"), default="200")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--api-path", type=Path, default=core.NAS_ROOT / core.cfg.nb201_api_path)
    parser.add_argument("--output", type=Path, default=core.NAS_ROOT / "ablation" / "results" /
                        datetime.now().strftime("kernels_imagenet_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.steps, args.population, args.runs, args.knowledge_topk, args.threads) <= 0:
        parser.error("steps/population/runs/topk/threads must be positive")
    if args.knowledge_topk > args.population or not 0 <= args.dynamic_steps <= args.steps:
        parser.error("topk <= population and 0 <= dynamic-steps <= steps required")
    if not 0 <= args.prior_strength <= 1:
        parser.error("prior-strength must be in [0,1]")
    if not all(math.isfinite(v) and v > 0 for v in (args.temperature, args.sigma)):
        parser.error("temperature and sigma must be finite and positive")
    if args.seed_start < 0 or args.seed_start + args.runs > 2**32:
        parser.error("seeds must be in [0, 2**32)")
    if len(args.kernels) != len(set(args.kernels)) or args.log_every < 0:
        parser.error("kernels must not repeat and log-every must be nonnegative")
    try:
        per_edge_scales(args.gaussian_sigma)
        per_edge_scales(args.distance_tau)
        marginal_probabilities(args.fixed_probs)
    except ValueError as error:
        parser.error(str(error))
    return args


def settings(args):
    metadata = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("output", "dry_run")}
    metadata.update(
        dataset=core.DATASET, token_order=list(OP_NAMES), mask_token=MASK,
        search_split="x-test", report_split="x-test", architecture_evaluations_per_run=args.population * (args.steps + 1),
        final_test_queries_per_run=0, standard_deviation="sample (ddof=1)",
        fixed_rho=marginal_probabilities(args.fixed_probs).tolist(),
        gaussian_sigma_per_edge=per_edge_scales(args.gaussian_sigma).tolist(),
        distance_tau_per_edge=per_edge_scales(args.distance_tau).tolist(),
        ordered_kernel_caveat="NB201 operation IDs are nominal; fixed index order is an explicit surrogate ordering",
        validity="API-queryable evaluated architectures; duplicates included; raw latent validity saved separately",
        torch_version=str(torch.__version__),
        shared_config={key: getattr(core.cfg, key) for key in (
            "d3pm_eps", "d3pm_schedule", "d3pm_cosine_s", "predictor_estimator_eps",
            "select_elite_frac", "select_eps", "select_fill_uniform")},
    )
    return metadata


def main():
    args = parse_args()
    metadata = settings(args)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    if not args.api_path.is_file():
        raise FileNotFoundError(args.api_path)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.set_num_threads(args.threads)
    schedule = make_scheduler(args, "uniform")
    (args.output / "noise_schedule.json").write_text(json.dumps(
        dict(alpha_bar=schedule.alpha_bar.tolist(), beta=schedule.beta.tolist()), indent=2) + "\n")
    core.log(f"Output: {args.output.resolve()}")
    core.log(f"Proposal mode={args.proposal_mode}; fixed rho={args.fixed_probs}")
    if args.proposal_mode == "reverse_only" and "absorbing" in args.kernels:
        core.log("Absorbing/mask: clean reverse-only proposals preserve tokens; expect no new architectures.")
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    from nas_201_api import NASBench201API
    core.log(f"Loading API once: {args.api_path}")
    api = NASBench201API(str(args.api_path), verbose=False)
    load_seconds = time.perf_counter() - started
    core.log(f"API loaded in {load_seconds:.2f}s")
    results = []
    with (args.output / "runs.jsonl").open("w", encoding="utf-8") as logfile:
        for seed in range(args.seed_start, args.seed_start + args.runs):
            for kind in args.kernels:
                core.log(f"START {kind} seed={seed}")
                result = run_kernel(args, api, kind, seed)
                logfile.write(json.dumps(result) + "\n")
                logfile.flush()
                results.append(result)
                summarize(results, args.output, args)
                core.log(f"DONE {kind} seed={seed} test={result['test_accuracy']:.4f} "
                         f"validity={result['validity_pct']:.2f}% time={result['search_seconds']:.2f}s")
    (args.output / "execution.json").write_text(json.dumps(dict(
        completed_runs=len(results), api_load_seconds=load_seconds,
        total_seconds=time.perf_counter() - started, started_at=started_at,
        finished_at=datetime.now().astimezone().isoformat()), indent=2) + "\n")
    print(summarize(results, args.output, args))


if __name__ == "__main__":
    main()
