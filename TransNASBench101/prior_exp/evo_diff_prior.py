import time

import torch
import tqdm

from config.config import d3pm_cosine_s, d3pm_eps, d3pm_schedule
from prior_exp.predictor_prior import BayesianGenerator
from prior_exp.prior_d3pm import PresetPriorMatrixScheduler
from prior_exp.prior_kernels import build_prior_kernel
from utils.analyse import compute_uniqueness
from utils.correct import select_population
from utils.plot import plot_denoise
from utils.transnasbench101_fitness import arch_fitness


VOCAB_SIZE = 4


def _geno_size(search_space):
    return 6 if search_space == "micro" else 7


def _init_population(population_num, search_space):
    return torch.randint(low=0, high=VOCAB_SIZE, size=(population_num, _geno_size(search_space)))


def _build_scheduler(
    num_step,
    search_space,
    prior_type,
    prior_kwargs,
    d3pm_eps,
    d3pm_schedule,
    d3pm_cosine_s,
):
    num_edges = _geno_size(search_space)
    prior_kernel = build_prior_kernel(
        prior_type,
        num_edges=num_edges,
        vocab_size=VOCAB_SIZE,
        **prior_kwargs,
    )
    return PresetPriorMatrixScheduler(
        num_steps=num_step,
        vocab_size=VOCAB_SIZE,
        prior_kernel=prior_kernel,
        eps=d3pm_eps,
        schedule=d3pm_schedule,
        cosine_s=d3pm_cosine_s,
    )


def _is_prior_knowledge(prior_type):
    return str(prior_type).lower() in ("prior_konwleage", "prior_knowledge", "knowledge")


def _select_topk_candidates(candidates, scores, topk):
    candidates = candidates.detach().cpu()
    scores = scores.detach().cpu().view(-1)
    topk = min(int(topk), candidates.shape[0])
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    top_indices = torch.topk(scores, k=topk, largest=True).indices
    return candidates[top_indices].clone(), scores[top_indices].clone()


def _build_knowledge_scheduler_kwargs(base_kwargs, knowledge_archs, knowledge_scores):
    archs = torch.cat(knowledge_archs, dim=0)
    scores = torch.cat(knowledge_scores, dim=0)
    scheduler_kwargs = dict(base_kwargs)
    scheduler_kwargs.update(
        {
            "initial_population": archs,
            "initial_scores": scores,
            "topk": archs.shape[0],
        }
    )
    return scheduler_kwargs


def evo_diff_with_prior(
    task,
    search_space,
    api,
    num_step,
    population_num,
    plot_results,
    save_dir,
    seed,
    *,
    prior_type="structural",
    prior_kwargs=None,
    d3pm_eps=d3pm_eps,
    d3pm_schedule=d3pm_schedule,
    d3pm_cosine_s=d3pm_cosine_s,
    predictor_estimator_eps=1e-12,
    predictor_temperature=0.4,
    predictor_sigma=1.0,
    select_elite_frac=0.3,
    select_eps=1e-20,
    select_fill_uniform=True,
):
    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    knowledge_dynamic_steps = int(prior_kwargs.pop("dynamic_steps", 0))
    if knowledge_dynamic_steps < 0:
        raise ValueError(f"dynamic_steps must be >= 0, got {knowledge_dynamic_steps}")
    start_time = time.time()
    x = _init_population(population_num, search_space)

    avg_acc_trace = []
    max_acc_trace = []
    uniq_rate_trace = []

    accuracy, fitness = arch_fitness(operation_matrix=x, api=api, task=task, search_space=search_space)
    knowledge_archs = []
    knowledge_scores = []
    if _is_prior_knowledge(prior_type):
        if search_space == "macro":
            prior_kwargs.setdefault("groups", [[0], list(range(1, _geno_size(search_space)))])
        top_x, top_scores = _select_topk_candidates(x, fitness, prior_kwargs.get("topk", 10))
        knowledge_archs.append(top_x)
        knowledge_scores.append(top_scores)
        scheduler_kwargs = _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores)
    else:
        scheduler_kwargs = prior_kwargs
    scheduler = _build_scheduler(
        num_step,
        search_space,
        prior_type,
        scheduler_kwargs,
        d3pm_eps,
        d3pm_schedule,
        d3pm_cosine_s,
    )

    bar = tqdm.tqdm(range(num_step), ncols=120)
    for t in bar:
        generator = BayesianGenerator(
            x=x,
            fitness=fitness,
            q=scheduler.Q,
            qbar=scheduler.Qbar,
            t=num_step - t,
            estimator_eps=predictor_estimator_eps,
            sigma=predictor_sigma,
            temperature=predictor_temperature,
        )
        x_next = generator.generate()
        uniq_rate = compute_uniqueness(arch_op_matrices=x_next)
        accuracy, fitness_next = arch_fitness(operation_matrix=x_next, api=api, task=task, search_space=search_space)

        if _is_prior_knowledge(prior_type) and t < knowledge_dynamic_steps:
            child_top_x, child_top_scores = _select_topk_candidates(
                x_next,
                fitness_next,
                prior_kwargs.get("topk", 10),
            )
            knowledge_archs.append(child_top_x)
            knowledge_scores.append(child_top_scores)
            scheduler = _build_scheduler(
                num_step,
                search_space,
                prior_type,
                _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores),
                d3pm_eps,
                d3pm_schedule,
                d3pm_cosine_s,
            )

        max_acc = accuracy.min().item() if task == "room_layout" else accuracy.max().item()
        avg_acc = accuracy.mean().item()
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        uniq_rate_trace.append(uniq_rate)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                "uniq_rate": f"{uniq_rate:.2f}",
                "prior": str(prior_type),
                "dyn": str(knowledge_dynamic_steps),
            }
        )

        x, fitness = select_population(
            x,
            fitness,
            x_next,
            fitness_next,
            population_num,
            elite_frac=select_elite_frac,
            eps=select_eps,
            fill_uniform=select_fill_uniform,
            return_fitness=True,
        )

    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            seed=seed,
            dataset=task,
        )
    duration = time.time() - start_time

    accuracy, fitness = arch_fitness(operation_matrix=x, api=api, task=task, search_space=search_space)
    if task == "room_layout":
        max_acc = min(accuracy.min().item(), min(max_acc_trace))
    else:
        max_acc = max(accuracy.max().item(), max(max_acc_trace))
    return max_acc, duration, uniq_rate, x
