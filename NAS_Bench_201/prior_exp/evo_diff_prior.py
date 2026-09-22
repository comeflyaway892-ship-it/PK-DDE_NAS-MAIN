import time

import torch
import tqdm

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    meta_predictor_ckpt_path,
    meta_predictor_hs,
    meta_predictor_nasbench201_pt_path,
    meta_predictor_num_sample,
    meta_predictor_nvt,
    meta_predictor_nz,
    nb201_num_edges,
    nb201_vocab_size,
)
from prior_exp.predictor_prior import BayesianGenerator
from prior_exp.prior_d3pm import PresetPriorMatrixScheduler
from prior_exp.prior_kernels import build_prior_kernel
from utils.analyse import compute_uniqueness
from utils.meta_d2a import FitnessRestorer, MetaSurrogateUnnoisedModel, load_graph_config, load_model
from utils.meta_fitness import meta_arch_fitness
from utils.nb201_fitness import arch_fitness
from utils.plot_cn import plot_denoise
from utils.select import select_population


def _init_population(population_num):
    return torch.randint(low=0, high=nb201_vocab_size, size=(population_num, nb201_num_edges))


def _build_scheduler(num_step, prior_type, prior_kwargs):
    prior_kernel = build_prior_kernel(
        prior_type,
        num_edges=nb201_num_edges,
        vocab_size=nb201_vocab_size,
        **prior_kwargs,
    )
    return PresetPriorMatrixScheduler(
        num_steps=num_step,
        vocab_size=nb201_vocab_size,
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
    dataset,
    api,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
    *,
    prior_type="structural",
    prior_kwargs=None,
    predictor_estimator_eps=1e-12,
    predictor_temperature=0.4,
    predictor_sigma=1.0,
    select_elite_frac=0.3,
    select_eps=1e-20,
    select_fill_uniform=True,
    return_trace=False,
):
    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    knowledge_dynamic_steps = int(prior_kwargs.pop("dynamic_steps", 0))
    if knowledge_dynamic_steps < 0:
        raise ValueError(f"dynamic_steps must be >= 0, got {knowledge_dynamic_steps}")
    x = _init_population(population_num)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []

    accuracy, fitness, valid_rate = arch_fitness(operation_matrix=x, api=api, dataset=dataset)
    knowledge_archs = []
    knowledge_scores = []
    if _is_prior_knowledge(prior_type):
        top_x, top_scores = _select_topk_candidates(x, accuracy, prior_kwargs.get("topk", 10))
        knowledge_archs.append(top_x)
        knowledge_scores.append(top_scores)
        scheduler_kwargs = _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores)
    else:
        scheduler_kwargs = prior_kwargs
    scheduler = _build_scheduler(num_step, prior_type, scheduler_kwargs)

    start_time = time.time()
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
        accuracy, fitness_next, valid_rate = arch_fitness(operation_matrix=x_next, api=api, dataset=dataset)
        if _is_prior_knowledge(prior_type) and t < knowledge_dynamic_steps:
            child_top_x, child_top_scores = _select_topk_candidates(
                x_next,
                accuracy,
                prior_kwargs.get("topk", 10),
            )
            knowledge_archs.append(child_top_x)
            knowledge_scores.append(child_top_scores)
            scheduler = _build_scheduler(
                num_step,
                prior_type,
                _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores),
            )
        max_acc = accuracy.max().item()
        avg_acc = accuracy.mean().item()
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        valid_rate_trace.append(valid_rate)
        uniq_rate_trace.append(uniq_rate)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                "valid_rate": f"{valid_rate:.2f}",
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

    duration = time.time() - start_time
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )
    result = (max(max_acc_trace), duration, uniq_rate, x)
    if return_trace:
        result += ({
            "step": list(range(1, num_step + 1)),
            "population_mean_accuracy": [float(v) for v in avg_acc_trace],
            "population_max_accuracy": [float(v) for v in max_acc_trace],
            "valid_rate": [float(v) for v in valid_rate_trace],
            "unique_rate": [float(v) for v in uniq_rate_trace],
        },)
    return result


def _build_meta_components(dataset, seed):
    nasbench201 = torch.load(meta_predictor_nasbench201_pt_path)
    graph_config = load_graph_config(
        graph_data_name="nasbench201",
        nvt=meta_predictor_nvt,
        data_path=meta_predictor_nasbench201_pt_path,
    )
    meta_surrogate_unnoised_model = MetaSurrogateUnnoisedModel(
        nvt=meta_predictor_nvt,
        hs=meta_predictor_hs,
        nz=meta_predictor_nz,
        num_sample=meta_predictor_num_sample,
        graph_config=graph_config,
    )
    meta_surrogate_unnoised_model = load_model(
        model=meta_surrogate_unnoised_model,
        ckpt_path=meta_predictor_ckpt_path,
    )
    fitness_restorer = FitnessRestorer(
        dataset_name=dataset,
        num_sample=meta_predictor_num_sample,
        seed=seed,
    )
    return nasbench201, meta_surrogate_unnoised_model, fitness_restorer


def evo_diff_meta_with_prior(
    dataset,
    api,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
    *,
    prior_type="structural",
    prior_kwargs=None,
    predictor_estimator_eps=1e-12,
    predictor_temperature=0.4,
    predictor_sigma=1.0,
    select_elite_frac=0.3,
    select_eps=1e-20,
    select_fill_uniform=True,
    return_trace=False,
):
    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    knowledge_dynamic_steps = int(prior_kwargs.pop("dynamic_steps", 0))
    if knowledge_dynamic_steps < 0:
        raise ValueError(f"dynamic_steps must be >= 0, got {knowledge_dynamic_steps}")
    x = _init_population(population_num)
    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    all_x = []

    nasbench201, meta_model, fitness_restorer = _build_meta_components(dataset, seed)
    accuracy, fitness, valid_rate = meta_arch_fitness(
        operation_matrix=x,
        api=api,
        dataset=dataset,
        test_dataset=None,
        meta_surrogate_unnoised_model=meta_model,
        nasbench201=nasbench201,
        fitness_restorer=fitness_restorer,
    )
    knowledge_archs = []
    knowledge_scores = []
    if _is_prior_knowledge(prior_type):
        top_x, top_scores = _select_topk_candidates(x, accuracy, prior_kwargs.get("topk", 10))
        knowledge_archs.append(top_x)
        knowledge_scores.append(top_scores)
        scheduler_kwargs = _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores)
    else:
        scheduler_kwargs = prior_kwargs
    scheduler = _build_scheduler(num_step, prior_type, scheduler_kwargs)

    start_time = time.time()
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
        accuracy, fitness_next, valid_rate = meta_arch_fitness(
            operation_matrix=x_next,
            api=api,
            dataset=dataset,
            test_dataset=None,
            meta_surrogate_unnoised_model=meta_model,
            nasbench201=nasbench201,
            fitness_restorer=fitness_restorer,
        )
        if _is_prior_knowledge(prior_type) and t < knowledge_dynamic_steps:
            child_top_x, child_top_scores = _select_topk_candidates(
                x_next,
                accuracy,
                prior_kwargs.get("topk", 10),
            )
            knowledge_archs.append(child_top_x)
            knowledge_scores.append(child_top_scores)
            scheduler = _build_scheduler(
                num_step,
                prior_type,
                _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores),
            )
        max_acc = accuracy.max().item()
        avg_acc = accuracy.mean().item()
        avg_acc_trace.append(avg_acc)
        max_acc_trace.append(max_acc)
        valid_rate_trace.append(valid_rate)
        uniq_rate_trace.append(uniq_rate)
        bar.set_postfix(
            {
                "max_acc": f"{max_acc:.2f}",
                "avg_acc": f"{avg_acc:.2f}",
                "valid_rate": f"{valid_rate:.2f}",
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
        all_x.append(x)

    duration = time.time() - start_time
    if plot_results:
        plot_denoise(
            save_dir=save_dir,
            avg_acc_trace=avg_acc_trace,
            max_acc_trace=max_acc_trace,
            valid_rate_trace=valid_rate_trace,
            uniq_rate_trace=uniq_rate_trace,
            seed=seed,
            dataset=dataset,
        )
    result = (max(max_acc_trace), duration, uniq_rate, torch.cat(all_x, dim=0))
    if return_trace:
        result += ({
            "step": list(range(1, num_step + 1)),
            "population_mean_accuracy": [float(v) for v in avg_acc_trace],
            "population_max_accuracy": [float(v) for v in max_acc_trace],
            "valid_rate": [float(v) for v in valid_rate_trace],
            "unique_rate": [float(v) for v in uniq_rate_trace],
        },)
    return result
