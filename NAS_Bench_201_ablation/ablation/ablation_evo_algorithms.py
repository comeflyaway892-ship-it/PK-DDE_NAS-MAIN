import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import tqdm

# Add the parent directory to sys.path first to avoid importing another nas_201_api package.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config.config import nb201_num_edges, nb201_vocab_size
from nas_201_api import NASBench201API
from utils.nb201_fitness import arch_fitness
from utils.select import select_population


DATASET = "ImageNet16-120"


def default_api_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    original_api = repo_root / "NAS_Bench_201" / "nas_201_api" / "NAS_Bench_201-v1_1-096897.pth"
    local_api = repo_root / "NAS_Bench_201_ablation" / "nas_201_api" / "NAS_Bench_201-v1_1-096897.pth"
    if original_api.exists():
        return str(original_api)
    return str(local_api)


def default_output_path() -> str:
    return str(Path(__file__).resolve().parent / "results" / "evo_algorithms_imagenet16_120.jsonl")


def load_nb201_api(path: str, verbose: bool = False):
    load_api_time = time.time()
    print(f'>>> Creating the API for NAS_Bench_201: {path}')
    api = NASBench201API(path, verbose=verbose)
    if verbose:
        print(f'>>> Running time of creating NAS_Bench_201 API: {time.time() - load_api_time:.2f} s')
    return api


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_mutation(x: torch.Tensor, mutation_rate: float, vocab_size: int):
    mask = torch.rand_like(x.float()) < mutation_rate
    rnd = torch.randint(low=0, high=vocab_size, size=x.shape, device=x.device)
    x_new = x.clone()
    x_new[mask] = rnd[mask]
    return x_new


def evaluate_population(x: torch.Tensor, api, dataset: str):
    accuracies, fitness, valid_rate = arch_fitness(operation_matrix=x, api=api, dataset=dataset)
    return accuracies, fitness, valid_rate


def algorithm_random_mutation(
    api,
    dataset: str,
    num_generations: int,
    population_num: int,
    mutation_rate: float,
    elite_frac: float,
    seed: int,
):
    x = torch.randint(low=0, high=nb201_vocab_size, size=(population_num, nb201_num_edges))
    accuracies, fitness, valid_rate = evaluate_population(x, api, dataset)
    best_acc = accuracies.max().item()
    best_arch = x[accuracies.argmax()].clone()
    history = []

    for generation in tqdm.tqdm(range(num_generations), desc="RandomMutation", ncols=120):
        cand = random_mutation(x, mutation_rate, nb201_vocab_size)
        cand_acc, cand_fit, cand_valid = evaluate_population(cand, api, dataset)

        x = select_population(
            x,
            fitness,
            cand,
            cand_fit,
            population_num,
            elite_frac=elite_frac,
            eps=1e-12,
            fill_uniform=True,
        )
        accuracies, fitness, valid_rate = evaluate_population(x, api, dataset)

        gen_best = accuracies.max().item()
        if gen_best > best_acc:
            best_acc = gen_best
            best_arch = x[accuracies.argmax()].clone()

        history.append((generation, gen_best, accuracies.mean().item(), valid_rate))

    return best_acc, best_arch, history


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-Bench-201 Ablation: random mutation baseline")
    parser.add_argument("--dataset", choices=[DATASET], default=DATASET)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--mutation_rate", type=float, default=0.05)
    parser.add_argument("--elite_frac", type=float, default=0.3)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=30)
    parser.add_argument("--api_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=default_output_path())
    return parser.parse_args()


def run_algorithm_for_seed(args, api, seed):
    set_seed(seed)
    return algorithm_random_mutation(
        api,
        args.dataset,
        args.generations,
        args.population,
        args.mutation_rate,
        args.elite_frac,
        seed,
    )


def main():
    args = parse_args()

    api_path = args.api_path
    if api_path is None:
        api_path = default_api_path()

    api = load_nb201_api(api_path, verbose=False)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running random_mutation on dataset {args.dataset} for seeds {args.seed_start}-{args.seed_end}")
    print(f"mutation_rate: {args.mutation_rate}")
    print(f"output: {output_path}\n")

    all_best_acc = []
    with output_path.open("a", encoding="utf-8") as f:
        for seed in seeds:
            best_acc, best_arch, history = run_algorithm_for_seed(args, api, seed)
            all_best_acc.append(best_acc)
            result = {
                "dataset": args.dataset,
                "algorithm": "random_mutation",
                "seed": seed,
                "generations": args.generations,
                "population": args.population,
                "mutation_rate": args.mutation_rate,
                "elite_frac": args.elite_frac,
                "best_acc": best_acc,
                "best_arch": best_arch.tolist(),
                "history": [
                    {
                        "generation": int(item[0]),
                        "best_acc": float(item[1]),
                        "mean_acc": float(item[2]),
                        "valid_rate": float(item[3]),
                    }
                    for item in history
                ],
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
