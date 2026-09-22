from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    select_elite_frac,
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
    global tqdm
    global meta_arch_fitness
    global OFAProxyEvaluator
    global select_population

    import torch as _torch
    import tqdm as _tqdm

    from utils.meta_fitness import meta_arch_fitness as _meta_arch_fitness
    from utils.ofa_proxy import OFAProxyEvaluator as _OFAProxyEvaluator
    from utils.select import select_population as _select_population

    torch = _torch
    tqdm = _tqdm
    meta_arch_fitness = _meta_arch_fitness
    OFAProxyEvaluator = _OFAProxyEvaluator
    select_population = _select_population


def default_output_path(dataset: str) -> str:
    return str(Path(__file__).resolve().parent / "results" / f"evo_algorithms_{dataset}.jsonl")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def algorithm_random_mutation(
    dataset: str,
    ofa_proxy,
    num_generations: int,
    population_num: int,
    mutation_rate: float,
    elite_frac: float,
):
    x = init_population(population_num)
    accuracies, fitness, valid_rate = evaluate_population(x, dataset, ofa_proxy)
    best_acc = accuracies.max().item()
    best_arch = x[accuracies.argmax()].clone()
    history = []

    for generation in tqdm.tqdm(range(num_generations), desc="RandomMutation", ncols=120):
        cand = random_mutation(x, mutation_rate)
        cand_acc, cand_fit, cand_valid = evaluate_population(cand, dataset, ofa_proxy)

        cand_best_acc = cand_acc.max().item()
        if cand_best_acc > best_acc:
            best_acc = cand_best_acc
            best_arch = cand[cand_acc.argmax()].clone()

        x, fitness = select_population(
            x,
            fitness,
            cand,
            cand_fit,
            population_num,
            elite_frac=elite_frac,
            eps=1e-12,
            fill_uniform=True,
            return_fitness=True,
        )
        accuracies, fitness, valid_rate = evaluate_population(x, dataset, ofa_proxy)

        gen_best = accuracies.max().item()
        if gen_best > best_acc:
            best_acc = gen_best
            best_arch = x[accuracies.argmax()].clone()

        history.append(
            {
                "generation": int(generation),
                "candidate_best_acc": float(cand_best_acc),
                "candidate_valid_rate": float(cand_valid),
                "best_acc": float(gen_best),
                "mean_acc": float(accuracies.mean().item()),
                "valid_rate": float(valid_rate),
            }
        )

    return best_acc, best_arch, history


def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV3/OFA ablation: random mutation baseline")
    parser.add_argument("--dataset", choices=meta_dataset_list, default="cifar10")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--mutation_rate", type=float, default=0.05)
    parser.add_argument("--elite_frac", type=float, default=select_elite_frac)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=30)
    parser.add_argument("--gpu", type=str, default=os.environ.get("DDE_NAG_GPU", os.environ.get("EDNAG_GPU", "1")))
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def run_algorithm_for_seed(args, ofa_proxy, seed):
    set_seed(seed)
    return algorithm_random_mutation(
        dataset=args.dataset,
        ofa_proxy=ofa_proxy,
        num_generations=args.generations,
        population_num=args.population,
        mutation_rate=args.mutation_rate,
        elite_frac=args.elite_frac,
    )


def main():
    args = parse_args()
    set_single_gpu(args.gpu)
    load_runtime_deps()
    output_path = Path(args.output or default_output_path(args.dataset))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    ofa_proxy = build_ofa_proxy(args.dataset)

    print(f"Running random_mutation on dataset {args.dataset} for seeds {args.seed_start}-{args.seed_end}")
    print(f"mutation_rate: {args.mutation_rate}")
    print(f"output: {output_path}\n")

    all_best_acc = []
    with output_path.open("a", encoding="utf-8") as f:
        for seed in seeds:
            best_acc, best_arch, history = run_algorithm_for_seed(args, ofa_proxy, seed)
            all_best_acc.append(best_acc)
            result = {
                "dataset": args.dataset,
                "algorithm": "random_mutation",
                "seed": seed,
                "generations": args.generations,
                "population": args.population,
                "mutation_rate": args.mutation_rate,
                "elite_frac": args.elite_frac,
                "best_acc": float(best_acc),
                "best_arch": best_arch.tolist(),
                "history": history,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"algorithm=random_mutation, seed={seed:02d}, best_acc={best_acc:.4f}")

    results_tensor = torch.tensor(all_best_acc, dtype=torch.float32)
    print("\nSummary")
    print(
        f"random_mutation: mean={results_tensor.mean().item():.4f}, "
        f"std={results_tensor.std(unbiased=False).item():.4f}, runs={len(all_best_acc)}"
    )


if __name__ == "__main__":
    main()
