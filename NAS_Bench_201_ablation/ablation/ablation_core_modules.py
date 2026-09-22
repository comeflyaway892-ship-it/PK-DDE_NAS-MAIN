import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config.config import (  # noqa: E402
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    nb201_num_edges,
    nb201_vocab_size,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from nas_201_api import NASBench201API  # noqa: E402
from utils.analyse import compute_uniqueness  # noqa: E402
from utils.d3pm import D3PMUniformMatrixScheduler  # noqa: E402
from utils.nb201_fitness import arch_fitness  # noqa: E402
from utils.predictor import d3pm_step  # noqa: E402
from utils.select import select_population  # noqa: E402


DATASET = "ImageNet16-120"
VARIANTS = (
    "full",
    "no_d3pm",
    "xt_as_x0",
    "mutate_x0",
)


def default_api_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    original_api = repo_root / "NAS_Bench_201" / "nas_201_api" / "NAS_Bench_201-v1_1-096897.pth"
    local_api = repo_root / "NAS_Bench_201_ablation" / "nas_201_api" / "NAS_Bench_201-v1_1-096897.pth"
    if original_api.exists():
        return str(original_api)
    return str(local_api)


def default_output_path() -> str:
    return str(Path(__file__).resolve().parent / "results" / "core_modules_imagenet16_120.jsonl")


def load_nb201_api(path: str, verbose: bool = False):
    start = time.time()
    print(f">>> Creating the API for NAS-Bench-201: {path}")
    api = NASBench201API(path, verbose=verbose)
    if verbose:
        print(f">>> API loaded in {time.time() - start:.2f} s")
    return api


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_population(x: torch.Tensor, api):
    accuracies, fitness, valid_rate = arch_fitness(operation_matrix=x, api=api, dataset=DATASET)
    return accuracies, fitness, valid_rate


def random_mutation(x: torch.Tensor, mutation_rate: float, vocab_size: int):
    mask = torch.rand_like(x.float()) < mutation_rate
    rnd = torch.randint(low=0, high=vocab_size, size=x.shape, device=x.device)
    x_new = x.clone()
    x_new[mask] = rnd[mask]
    return x_new


def fitness_base_weights(
    fitness: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    temp = max(float(temperature), eps)
    centered = fitness.float() - fitness.float().mean()
    base = torch.softmax(centered / temp, dim=0).clamp_min(eps)
    return base / base.sum().clamp_min(eps)


def diffusion_consistent_weights(
    x: torch.Tensor,
    qbar: torch.Tensor,
    t: int,
    base: torch.Tensor,
    sigma: float,
    eps: float,
) -> torch.Tensor:
    n, d = x.shape
    k = qbar.shape[-1]
    qbar_t = qbar[int(t)].clamp_min(eps)
    x_t = x.long().clamp(0, k - 1)
    log_kernel = torch.zeros((n, n), device=x.device, dtype=torch.float32)

    for edge in range(d):
        from_token = x_t[:, edge]
        to_token = x_t[:, edge]
        matrix = qbar_t[from_token.view(1, n), to_token.view(n, 1)].clamp_min(eps)
        log_kernel += torch.log(matrix)

    log_kernel = log_kernel / d
    kernel = torch.exp(log_kernel / max(float(sigma), eps)).clamp_min(eps)
    weights = kernel * base.view(1, n)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)


def estimate_x0(
    x: torch.Tensor,
    fitness: torch.Tensor,
    qbar: torch.Tensor,
    t: int,
    temperature: float,
    sigma: float,
    eps: float,
) -> torch.Tensor:
    n, d = x.shape
    k = qbar.shape[-1]
    base = fitness_base_weights(fitness, temperature, eps)
    weights = diffusion_consistent_weights(x, qbar, t, base, sigma, eps)

    onehot = F.one_hot(x.long().clamp(0, k - 1), num_classes=k).float()
    p0 = torch.einsum("ij,jdk->idk", weights, onehot).clamp_min(eps)
    p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(eps)

    x0_flat = torch.multinomial(p0.reshape(n * d, k), num_samples=1).squeeze(1)
    return x0_flat.view(n, d)


def generate_candidate(
    x: torch.Tensor,
    fitness: torch.Tensor,
    scheduler: D3PMUniformMatrixScheduler,
    t: int,
    variant: str,
    temperature: float,
    sigma: float,
    estimator_eps: float,
    x0_mutation_rate: float,
) -> torch.Tensor:
    if variant == "xt_as_x0":
        x0_est = x
    elif variant == "mutate_x0":
        x0_est = random_mutation(x, mutation_rate=x0_mutation_rate, vocab_size=nb201_vocab_size)
    else:
        x0_est = estimate_x0(
            x=x,
            fitness=fitness,
            qbar=scheduler.Qbar,
            t=t,
            temperature=temperature,
            sigma=sigma,
            eps=estimator_eps,
        )
    if variant == "no_d3pm":
        return x0_est
    return d3pm_step(xt=x, x0=x0_est, q=scheduler.Q, qbar=scheduler.Qbar, t=t)


def run_variant(args, api, variant: str, seed: int):
    set_seed(seed)
    started = time.time()
    x = torch.randint(low=0, high=nb201_vocab_size, size=(args.population, nb201_num_edges))
    accuracies, fitness, valid_rate = evaluate_population(x, api)

    best_acc = accuracies.max().item()
    best_arch = x[accuracies.argmax()].clone()
    history = []

    scheduler = D3PMUniformMatrixScheduler(
        num_steps=args.num_step,
        vocab_size=nb201_vocab_size,
        eps=args.d3pm_eps,
        schedule=args.d3pm_schedule,
        cosine_s=args.d3pm_cosine_s,
    )

    iterator = range(args.num_step)
    if not args.no_progress:
        iterator = tqdm.tqdm(iterator, desc=f"{variant}/seed{seed}", ncols=120)

    for step in iterator:
        t = args.num_step - step
        x_cand = generate_candidate(
            x=x,
            fitness=fitness,
            scheduler=scheduler,
            t=t,
            variant=variant,
            temperature=args.predictor_temperature,
            sigma=args.predictor_sigma,
            estimator_eps=args.predictor_estimator_eps,
            x0_mutation_rate=args.x0_mutation_rate,
        )
        cand_acc, cand_fitness, cand_valid_rate = evaluate_population(x_cand, api)
        uniq_rate = compute_uniqueness(x_cand)

        x = select_population(
            x,
            fitness,
            x_cand,
            cand_fitness,
            args.population,
            elite_frac=args.select_elite_frac,
            eps=args.select_eps,
            fill_uniform=args.select_fill_uniform,
        )
        accuracies, fitness, valid_rate = evaluate_population(x, api)

        gen_best = max(cand_acc.max().item(), accuracies.max().item())
        if gen_best > best_acc:
            if cand_acc.max().item() >= accuracies.max().item():
                best_acc = cand_acc.max().item()
                best_arch = x_cand[cand_acc.argmax()].clone()
            else:
                best_acc = accuracies.max().item()
                best_arch = x[accuracies.argmax()].clone()

        row = {
            "step": step,
            "t": t,
            "candidate_best_acc": cand_acc.max().item(),
            "population_best_acc": accuracies.max().item(),
            "population_mean_acc": accuracies.mean().item(),
            "candidate_valid_rate": cand_valid_rate,
            "population_valid_rate": valid_rate,
            "uniq_rate": uniq_rate,
        }
        history.append(row)
        if not args.no_progress:
            iterator.set_postfix(
                {
                    "best": f"{best_acc:.2f}",
                    "pop": f"{row['population_best_acc']:.2f}",
                    "uniq": f"{uniq_rate:.2f}",
                }
            )

    return {
        "dataset": DATASET,
        "variant": variant,
        "seed": seed,
        "num_step": args.num_step,
        "population": args.population,
        "x0_mutation_rate": args.x0_mutation_rate if variant == "mutate_x0" else None,
        "best_acc": best_acc,
        "best_arch": best_arch.tolist(),
        "duration_sec": time.time() - started,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="NAS-Bench-201 ImageNet16-120 core-module ablations for DiffEvo-NAS"
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["all"],
        default="all",
        help="full, no_d3pm, xt_as_x0, mutate_x0, or all",
    )
    parser.add_argument("--num_step", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=30)
    parser.add_argument("--api_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=default_output_path())
    parser.add_argument("--no_progress", action="store_true")

    parser.add_argument("--d3pm_eps", type=float, default=d3pm_eps)
    parser.add_argument("--d3pm_schedule", choices=["linear", "cosine"], default=d3pm_schedule)
    parser.add_argument("--d3pm_cosine_s", type=float, default=d3pm_cosine_s)
    parser.add_argument("--predictor_estimator_eps", type=float, default=predictor_estimator_eps)
    parser.add_argument("--predictor_temperature", type=float, default=predictor_temperature)
    parser.add_argument("--predictor_sigma", type=float, default=predictor_sigma)
    parser.add_argument("--x0_mutation_rate", type=float, default=0.05)
    parser.add_argument("--select_elite_frac", type=float, default=select_elite_frac)
    parser.add_argument("--select_eps", type=float, default=select_eps)
    fill_group = parser.add_mutually_exclusive_group()
    fill_group.add_argument("--select_fill_uniform", dest="select_fill_uniform", action="store_true")
    fill_group.add_argument("--no_select_fill_uniform", dest="select_fill_uniform", action="store_false")
    parser.set_defaults(select_fill_uniform=select_fill_uniform)
    return parser.parse_args()


def summarize(results):
    by_variant = {}
    for item in results:
        by_variant.setdefault(item["variant"], []).append(item["best_acc"])

    print("\nSummary")
    for variant, values in by_variant.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        print(
            f"{variant}: mean={tensor.mean().item():.4f}, "
            f"std={tensor.std(unbiased=False).item():.4f}, "
            f"runs={len(values)}"
        )


def main():
    args = parse_args()
    api_path = args.api_path or default_api_path()
    api = load_nb201_api(api_path, verbose=False)

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running core-module ablations on {DATASET}")
    print(f"variants: {variants}")
    print(f"seeds: {args.seed_start}-{args.seed_end}")
    print(f"output: {output_path}")

    results = []
    with output_path.open("a", encoding="utf-8") as f:
        for variant in variants:
            for seed in seeds:
                result = run_variant(args, api, variant, seed)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results.append(result)
                print(f"variant={variant}, seed={seed:02d}, best_acc={result['best_acc']:.4f}")

    summarize(results)


if __name__ == "__main__":
    main()
