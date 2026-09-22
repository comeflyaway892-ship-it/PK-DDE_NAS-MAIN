from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
ORIGINAL_MOBILENETV3_ROOT = os.path.join(REPO_ROOT, "MobileNetV3")
for path in (ORIGINAL_MOBILENETV3_ROOT, PROJECT_DIR):
    if path in sys.path:
        sys.path.remove(path)
for path in (ORIGINAL_MOBILENETV3_ROOT, PROJECT_DIR):
    sys.path.insert(0, path)

from config.config import (  # noqa: E402
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    meta_dataset_list,
    num_depth_tokens,
    num_edges,
    ofa_proxy_acc_mean,
    ofa_proxy_acc_std,
    ofa_proxy_eval_repeats,
    ofa_proxy_hs,
    ofa_proxy_num_sample,
    ofa_proxy_nvt,
    ofa_proxy_nz,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
    vocab_size,
)


def set_single_gpu(gpu: str):
    gpu = str(gpu).strip()
    if "," in gpu:
        raise ValueError(f"MobileNetV3 ablation expects one GPU id, got: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["DDE_NAG_GPU"] = gpu
    os.environ["EDNAG_GPU"] = gpu
    print(f"==> MobileNetV3 ablation uses physical GPU {gpu}")


def load_runtime_deps():
    global torch
    global F
    global tqdm
    global compute_uniqueness
    global D3PMUniformMatrixScheduler
    global meta_arch_fitness
    global OFAProxyEvaluator
    global d3pm_step
    global select_population

    import torch as _torch
    import torch.nn.functional as _F
    import tqdm as _tqdm

    from utils.analyse import compute_uniqueness as _compute_uniqueness
    from utils.d3pm import D3PMUniformMatrixScheduler as _D3PMUniformMatrixScheduler
    from utils.meta_fitness import meta_arch_fitness as _meta_arch_fitness
    from utils.ofa_proxy import OFAProxyEvaluator as _OFAProxyEvaluator
    from utils.predictor import d3pm_step as _d3pm_step
    from utils.select import select_population as _select_population

    torch = _torch
    F = _F
    tqdm = _tqdm
    compute_uniqueness = _compute_uniqueness
    D3PMUniformMatrixScheduler = _D3PMUniformMatrixScheduler
    meta_arch_fitness = _meta_arch_fitness
    OFAProxyEvaluator = _OFAProxyEvaluator
    d3pm_step = _d3pm_step
    select_population = _select_population


VARIANTS = (
    "fitness_as_one",
    "diffusion_consistency_as_one",
    "no_posterior_recovery",
    "global_mutation",
)


def default_output_path(dataset: str) -> str:
    return str(Path(__file__).resolve().parent / "results" / f"core_modules_{dataset}.jsonl")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_token_mask():
    mask = torch.ones(num_edges, vocab_size, dtype=torch.bool)
    mask[:num_depth_tokens, 3:] = False
    return mask


def init_population(population_num: int):
    x = torch.empty(population_num, num_edges, dtype=torch.long)
    x[:, :num_depth_tokens] = torch.randint(
        low=0,
        high=3,
        size=(population_num, num_depth_tokens),
    )
    x[:, num_depth_tokens:] = torch.randint(
        low=0,
        high=vocab_size,
        size=(population_num, num_edges - num_depth_tokens),
    )
    return x


def random_mutation(x: torch.Tensor, mutation_rate: float):
    mask = torch.rand_like(x.float()) < mutation_rate
    rnd = torch.empty_like(x)
    rnd[:, :num_depth_tokens] = torch.randint(
        low=0,
        high=3,
        size=(x.shape[0], num_depth_tokens),
        device=x.device,
    )
    rnd[:, num_depth_tokens:] = torch.randint(
        low=0,
        high=vocab_size,
        size=(x.shape[0], x.shape[1] - num_depth_tokens),
        device=x.device,
    )
    x_new = x.clone()
    x_new[mask] = rnd[mask]
    return x_new


def build_ofa_proxy(dataset: str):
    return OFAProxyEvaluator(
        dataset=dataset,
        num_sample=ofa_proxy_num_sample,
        nvt=ofa_proxy_nvt,
        hs=ofa_proxy_hs,
        nz=ofa_proxy_nz,
        eval_repeats=ofa_proxy_eval_repeats,
        acc_mean=ofa_proxy_acc_mean,
        acc_std=ofa_proxy_acc_std,
    )


def evaluate_population(x: torch.Tensor, dataset: str, ofa_proxy):
    return meta_arch_fitness(
        operation_matrix=x,
        dataset=dataset,
        ofa_proxy=ofa_proxy,
    )


def fitness_base_weights(
    fitness: torch.Tensor,
    temperature: float,
    eps: float,
    *,
    use_fitness: bool,
) -> torch.Tensor:
    if not use_fitness:
        return torch.full_like(fitness.float(), 1.0 / max(fitness.numel(), 1))

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
    *,
    use_diffusion_consistency: bool,
) -> torch.Tensor:
    n, d = x.shape
    if not use_diffusion_consistency:
        return base.view(1, n).repeat(n, 1)

    k = qbar.shape[-1]
    qbar_t = qbar[int(t)].clamp_min(eps)
    x_t = x.long().clamp(0, k - 1)
    log_kernel = torch.zeros((n, n), device=x.device, dtype=torch.float32)

    for edge in range(d):
        from_token = x_t[:, edge]
        to_token = x_t[:, edge]
        qbar_td = qbar_t if qbar_t.dim() == 2 else qbar_t[edge]
        matrix = qbar_td[from_token.view(1, n), to_token.view(n, 1)].clamp_min(eps)
        log_kernel += torch.log(matrix)

    log_kernel = log_kernel / d
    kernel = torch.exp(log_kernel / max(float(sigma), eps)).clamp_min(eps)
    weights = kernel * base.view(1, n)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)


def estimate_x0(
    x: torch.Tensor,
    fitness: torch.Tensor,
    scheduler: D3PMUniformMatrixScheduler,
    t: int,
    temperature: float,
    sigma: float,
    estimator_eps: float,
    *,
    use_fitness: bool,
    use_diffusion_consistency: bool,
) -> torch.Tensor:
    n, d = x.shape
    k = scheduler.Qbar.shape[-1]
    eps = float(estimator_eps)

    base = fitness_base_weights(
        fitness=fitness,
        temperature=temperature,
        eps=eps,
        use_fitness=use_fitness,
    )
    weights = diffusion_consistent_weights(
        x=x,
        qbar=scheduler.Qbar,
        t=t,
        base=base,
        sigma=sigma,
        eps=eps,
        use_diffusion_consistency=use_diffusion_consistency,
    )

    onehot = F.one_hot(x.long().clamp(0, k - 1), num_classes=k).float()
    p0 = torch.einsum("ij,jdk->idk", weights, onehot).clamp_min(eps)
    p0 = p0 / p0.sum(dim=-1, keepdim=True).clamp_min(eps)

    x0_flat = torch.multinomial(p0.reshape(n * d, k), num_samples=1).squeeze(1)
    x0 = x0_flat.view(n, d)
    x0[:, :num_depth_tokens] = x0[:, :num_depth_tokens].clamp(0, 2)
    x0[:, num_depth_tokens:] = x0[:, num_depth_tokens:].clamp(0, vocab_size - 1)
    return x0


def generate_candidate(
    x: torch.Tensor,
    fitness: torch.Tensor,
    scheduler: D3PMUniformMatrixScheduler,
    t: int,
    variant: str,
    temperature: float,
    sigma: float,
    estimator_eps: float,
    global_mutation_rate: float,
) -> torch.Tensor:
    if variant == "global_mutation":
        return random_mutation(x, mutation_rate=global_mutation_rate)

    use_fitness = variant != "fitness_as_one"
    use_diffusion_consistency = variant != "diffusion_consistency_as_one"
    x0_est = estimate_x0(
        x=x,
        fitness=fitness,
        scheduler=scheduler,
        t=t,
        temperature=temperature,
        sigma=sigma,
        estimator_eps=estimator_eps,
        use_fitness=use_fitness,
        use_diffusion_consistency=use_diffusion_consistency,
    )

    if variant == "no_posterior_recovery":
        return x0_est

    return d3pm_step(
        xt=x,
        x0=x0_est,
        q=scheduler.Q,
        qbar=scheduler.Qbar,
        t=t,
    )


def run_variant(args, ofa_proxy, variant: str, seed: int):
    set_seed(seed)
    started = time.time()

    x = init_population(args.population)
    accuracies, fitness, valid_rate = evaluate_population(x, args.dataset, ofa_proxy)

    best_acc = accuracies.max().item()
    best_arch = x[accuracies.argmax()].clone()
    history = []

    scheduler = None
    if variant != "global_mutation":
        scheduler = D3PMUniformMatrixScheduler(
            num_steps=args.num_step,
            vocab_size=vocab_size,
            eps=args.d3pm_eps,
            schedule=args.d3pm_schedule,
            cosine_s=args.d3pm_cosine_s,
            token_mask=build_token_mask(),
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
            global_mutation_rate=args.global_mutation_rate,
        )
        cand_acc, cand_fitness, cand_valid_rate = evaluate_population(x_cand, args.dataset, ofa_proxy)
        uniq_rate = compute_uniqueness(x_cand)

        cand_best_acc = cand_acc.max().item()
        if cand_best_acc > best_acc:
            best_acc = cand_best_acc
            best_arch = x_cand[cand_acc.argmax()].clone()

        x, fitness = select_population(
            x,
            fitness,
            x_cand,
            cand_fitness,
            args.population,
            elite_frac=args.select_elite_frac,
            eps=args.select_eps,
            fill_uniform=args.select_fill_uniform,
            return_fitness=True,
        )
        accuracies, fitness, valid_rate = evaluate_population(x, args.dataset, ofa_proxy)

        pop_best_acc = accuracies.max().item()
        if pop_best_acc > best_acc:
            best_acc = pop_best_acc
            best_arch = x[accuracies.argmax()].clone()

        row = {
            "step": int(step),
            "t": int(t),
            "candidate_best_acc": float(cand_best_acc),
            "population_best_acc": float(pop_best_acc),
            "population_mean_acc": float(accuracies.mean().item()),
            "candidate_valid_rate": float(cand_valid_rate),
            "population_valid_rate": float(valid_rate),
            "uniq_rate": float(uniq_rate),
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
        "dataset": args.dataset,
        "variant": variant,
        "seed": seed,
        "num_step": args.num_step,
        "population": args.population,
        "global_mutation_rate": args.global_mutation_rate if variant == "global_mutation" else None,
        "best_acc": float(best_acc),
        "best_arch": best_arch.tolist(),
        "duration_sec": time.time() - started,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNetV3/OFA core-module ablations for DiffEvo-NAS"
    )
    parser.add_argument("--dataset", choices=meta_dataset_list, default="cifar10")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS) + ["all"],
        default="all",
        help=(
            "fitness_as_one, diffusion_consistency_as_one, "
            "no_posterior_recovery, global_mutation, or all"
        ),
    )
    parser.add_argument("--num_step", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=30)
    parser.add_argument("--gpu", type=str, default=os.environ.get("DDE_NAG_GPU", os.environ.get("EDNAG_GPU", "1")))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")

    parser.add_argument("--d3pm_eps", type=float, default=d3pm_eps)
    parser.add_argument("--d3pm_schedule", choices=["linear", "cosine"], default=d3pm_schedule)
    parser.add_argument("--d3pm_cosine_s", type=float, default=d3pm_cosine_s)
    parser.add_argument("--predictor_estimator_eps", type=float, default=predictor_estimator_eps)
    parser.add_argument("--predictor_temperature", type=float, default=predictor_temperature)
    parser.add_argument("--predictor_sigma", type=float, default=predictor_sigma)
    parser.add_argument("--global_mutation_rate", type=float, default=0.05)
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
    set_single_gpu(args.gpu)
    load_runtime_deps()
    output_path = Path(args.output or default_output_path(args.dataset))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    ofa_proxy = build_ofa_proxy(args.dataset)

    print(f"Running core-module ablations on {args.dataset}")
    print(f"variants: {variants}")
    print(f"seeds: {args.seed_start}-{args.seed_end}")
    print(f"output: {output_path}")

    results = []
    with output_path.open("a", encoding="utf-8") as f:
        for variant in variants:
            for seed in seeds:
                result = run_variant(args, ofa_proxy, variant, seed)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results.append(result)
                print(f"variant={variant}, seed={seed:02d}, best_acc={result['best_acc']:.4f}")

    summarize(results)


if __name__ == "__main__":
    main()
