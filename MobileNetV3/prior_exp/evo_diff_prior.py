import time

import torch
import tqdm

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    num_depth_tokens,
    num_edges,
    ofa_proxy_acc_mean,
    ofa_proxy_acc_std,
    ofa_proxy_eval_repeats,
    ofa_proxy_hs,
    ofa_proxy_num_sample,
    ofa_proxy_nvt,
    ofa_proxy_nz,
    vocab_size,
)
from prior_exp.prior_d3pm import PresetPriorMatrixScheduler
from prior_exp.structural_prior import build_prior_kernel
from utils.analyse import compute_uniqueness
from utils.meta_fitness import meta_arch_fitness
from utils.ofa_proxy import OFAProxyEvaluator
from utils.plot import plot_denoise
from utils.predictor import BayesianGenerator
from utils.select import select_population


def _build_ofa_proxy(dataset):
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


def _evaluate_population(x, dataset, ofa_proxy):
    return meta_arch_fitness(
        operation_matrix=x,
        dataset=dataset,
        ofa_proxy=ofa_proxy,
    )


def _build_token_mask():
    mask = torch.ones(num_edges, vocab_size, dtype=torch.bool)
    mask[:num_depth_tokens, 3:] = False
    return mask


def _init_population(population_num):
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


def _append_ranked_candidates(candidate_log, candidates, accuracy):
    for candidate, score in zip(candidates.detach().cpu(), accuracy.detach().cpu()):
        candidate_log.append((float(score), candidate.clone()))


def _rank_unique_candidates(candidate_log):
    ranked = sorted(candidate_log, key=lambda item: item[0], reverse=True)
    unique = []
    seen = set()
    for _, candidate in ranked:
        key = tuple(candidate.view(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    if not unique:
        return torch.empty(0, num_edges, dtype=torch.long)
    return torch.stack(unique, dim=0)


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


def _build_scheduler(num_step, prior_type, prior_kwargs, d3pm_eps, d3pm_schedule, d3pm_cosine_s):
    prior_kernel = build_prior_kernel(prior_type, **prior_kwargs)
    return PresetPriorMatrixScheduler(
        num_steps=num_step,
        vocab_size=vocab_size,
        prior_kernel=prior_kernel,
        eps=d3pm_eps,
        schedule=d3pm_schedule,
        cosine_s=d3pm_cosine_s,
        token_mask=_build_token_mask(),
    )


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


def evo_diff_meta_with_prior(
    dataset,
    num_step,
    population_num,
    seed,
    plot_results,
    save_dir,
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
    x = _init_population(population_num)

    avg_acc_trace = []
    max_acc_trace = []
    valid_rate_trace = []
    uniq_rate_trace = []
    candidate_log = []

    ofa_proxy = _build_ofa_proxy(dataset)
    accuracy, fitness, valid_rate = _evaluate_population(x, dataset, ofa_proxy)
    _append_ranked_candidates(candidate_log, x, accuracy)

    knowledge_archs = []
    knowledge_scores = []
    if _is_prior_knowledge(prior_type):
        top_x, top_scores = _select_topk_candidates(x, accuracy, prior_kwargs.get("topk", 10))
        knowledge_archs.append(top_x)
        knowledge_scores.append(top_scores)
        scheduler_kwargs = _build_knowledge_scheduler_kwargs(prior_kwargs, knowledge_archs, knowledge_scores)
    else:
        scheduler_kwargs = prior_kwargs
    scheduler = _build_scheduler(
        num_step,
        prior_type,
        scheduler_kwargs,
        d3pm_eps,
        d3pm_schedule,
        d3pm_cosine_s,
    )

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

        accuracy, fitness_next, valid_rate = _evaluate_population(x_next, dataset, ofa_proxy)
        _append_ranked_candidates(candidate_log, x_next, accuracy)

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
                d3pm_eps,
                d3pm_schedule,
                d3pm_cosine_s,
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

    ranked_candidates = _rank_unique_candidates(candidate_log)
    return max(max_acc_trace), duration, uniq_rate, ranked_candidates
