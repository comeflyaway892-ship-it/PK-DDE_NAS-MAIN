"""Discrete diffusion evolutionary search for NAS-Bench-301."""

import time

import torch
import tqdm

from config.config import nb301_vocab_sizes
from utils.analyse import compute_uniqueness
from utils.d3pm import D3PMVariableVocabScheduler
from utils.fitness import arch_fitness
from utils.plot import plot_denoise
from utils.predictor import BayesianGenerator
from utils.select import select_population
from prior_exp.prior_kernels import build_prior_kernels


def sample_valid_architectures(population_num, device=None):
    """Sample valid tokens independently using each position's vocabulary."""
    columns = [
        torch.randint(0, vocab_size, (population_num,), device=device)
        for vocab_size in nb301_vocab_sizes
    ]
    return torch.stack(columns, dim=1).long()


def evo_diff(
    predictor,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
    *,
    d3pm_eps,
    d3pm_schedule,
    d3pm_cosine_s,
    predictor_estimator_eps,
    predictor_temperature,
    predictor_sigma,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
    prior_type="uniform",
    prior_kwargs=None,
):
    start_time = time.time()
    x = sample_valid_architectures(population_num)

    average_accuracy_trace = []
    max_accuracy_trace = []
    valid_rate_trace = []
    uniqueness_trace = []

    _, fitness, _ = arch_fitness(x, predictor)

    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    dynamic_steps = int(prior_kwargs.pop("dynamic_steps", 0))
    if dynamic_steps < 0:
        raise ValueError("dynamic_steps must be non-negative")

    def make_scheduler(kernels):
        return D3PMVariableVocabScheduler(
            num_steps=num_step,
            vocab_sizes=nb301_vocab_sizes,
            eps=d3pm_eps,
            schedule=d3pm_schedule,
            cosine_s=d3pm_cosine_s,
            prior_kernels=kernels,
        )

    knowledge_archs = []
    knowledge_scores = []
    if str(prior_type).lower() in ("prior_knowledge", "prior_konwleage", "knowledge"):
        knowledge_archs.append(x.detach().clone())
        knowledge_scores.append(fitness.detach().clone())
        prior_kwargs.update({"initial_population": x, "initial_scores": fitness})
    scheduler = make_scheduler(build_prior_kernels(prior_type, **prior_kwargs))

    progress = tqdm.tqdm(range(num_step), ncols=120)
    for iteration in progress:
        generator = BayesianGenerator(
            x=x,
            fitness=fitness,
            scheduler=scheduler,
            t=num_step - iteration,
            estimator_eps=predictor_estimator_eps,
            temperature=predictor_temperature,
            sigma=predictor_sigma,
        )
        candidate = generator.generate()
        uniqueness = compute_uniqueness(candidate)
        accuracy, candidate_fitness, valid_rate = arch_fitness(candidate, predictor)

        if str(prior_type).lower() in ("prior_knowledge", "prior_konwleage", "knowledge") and iteration < dynamic_steps:
            knowledge_archs.append(candidate.detach().clone())
            knowledge_scores.append(candidate_fitness.detach().clone())
            prior_kwargs.update({
                "initial_population": torch.cat(knowledge_archs),
                "initial_scores": torch.cat(knowledge_scores),
            })
            scheduler = make_scheduler(build_prior_kernels(prior_type, **prior_kwargs))

        average_accuracy = accuracy.mean().item()
        max_accuracy = accuracy.max().item()
        average_accuracy_trace.append(average_accuracy)
        max_accuracy_trace.append(max_accuracy)
        valid_rate_trace.append(valid_rate)
        uniqueness_trace.append(uniqueness)
        progress.set_postfix(
            {
                "max_acc": "{:.2f}".format(max_accuracy),
                "avg_acc": "{:.2f}".format(average_accuracy),
                "valid_rate": "{:.2f}".format(valid_rate),
                "uniq_rate": "{:.2f}".format(uniqueness),
            }
        )

        # Keep fitness synchronized with the selected parent/child mixture.
        x, fitness = select_population(
            x,
            fitness,
            candidate,
            candidate_fitness,
            population_num,
            elite_frac=select_elite_frac,
            eps=select_eps,
            fill_uniform=select_fill_uniform,
            return_fitness=True,
        )

    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=average_accuracy_trace,
            max_acc_trace=max_accuracy_trace,
            seed=seed,
            dataset="cifar10",
        )

    return (
        max(max_accuracy_trace),
        time.time() - start_time,
        uniqueness_trace[-1],
        x,
    )
