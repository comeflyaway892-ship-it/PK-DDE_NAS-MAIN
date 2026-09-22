import os
import random

import numpy as np
import torch

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    hyper_params_setting,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from prior_exp.evo_diff_prior import evo_diff_with_prior
from TransNASBench101.api import TransNASBenchAPI as API


PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_dir(search_space, task, prior_name):
    return os.path.join(PRIOR_EXP_ROOT, "results", prior_name, search_space, task)


def _mean_std(values):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute mean and std from an empty result list")
    return float(values.mean()), float(values.std())


def main_exp_prior(
    task,
    search_space,
    *,
    prior_type,
    prior_name,
    prior_kwargs=None,
    seed_list=None,
    api=None,
    num_step=None,
    population_num=None,
    plot_results=True,
):
    api = API("transnas-bench_v10141024.pth") if api is None else api
    assert task in api.task_list, f"ERROR: invalid task {task}, expected {api.task_list}"
    assert search_space in api.search_spaces, f"ERROR: invalid search space {search_space}, expected {api.search_spaces}"
    args = dict(hyper_params_setting[search_space][task])
    if num_step is not None:
        args["num_step"] = int(num_step)
    if population_num is not None:
        args["population_num"] = int(population_num)
    args["save_dir"] = _save_dir(search_space, task, prior_name)
    seed_list = args["seed"] if seed_list is None else list(seed_list)

    print(f">>> TransNASBench101 prior experiment: task={task}, search_space={search_space}, prior={prior_name}")
    print(f">>> prior_type={prior_type}, prior_kwargs={prior_kwargs}, seeds={seed_list}")
    result_log = {}
    os.makedirs(os.path.join(args["save_dir"], "search_log"), exist_ok=True)
    log_path = os.path.join(args["save_dir"], "search_log", f"{task}_{search_space}_{prior_name}_search.pth")
    if os.path.exists(log_path):
        result_log = torch.load(log_path)

    for seed in seed_list:
        if seed in result_log:
            cached_max_acc = result_log[seed].get("max_acc", "unknown")
            print(f">>> Log already exists, pass seed {seed}: max_acc={cached_max_acc}")
            continue
        set_random_seed(seed)
        print(f"\n>>> Running on {task}/{search_space} with seed {seed}...")
        max_acc, duration, uniq_rate, x = evo_diff_with_prior(
            seed=seed,
            task=task,
            search_space=search_space,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            plot_results=plot_results,
            save_dir=args["save_dir"],
            prior_type=prior_type,
            prior_kwargs=prior_kwargs,
            d3pm_eps=d3pm_eps,
            d3pm_schedule=d3pm_schedule,
            d3pm_cosine_s=d3pm_cosine_s,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        result_log[seed] = {
            "prior_type": prior_type,
            "prior_name": prior_name,
            "prior_kwargs": prior_kwargs,
            "max_acc": max_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "searched_archs": x.cpu(),
        }
        torch.save(result_log, log_path)
        print(f">>> Task {task} seed {seed} search_space {search_space}, max accuracy: {max_acc:.2f} %")

    seed_results = [result_log[seed] for seed in seed_list if seed in result_log]
    max_acc_values = [result["max_acc"] for result in seed_results if "max_acc" in result]
    if max_acc_values:
        max_acc_mean, max_acc_std = _mean_std(max_acc_values)
        duration_mean = np.mean([float(result["duration"]) for result in seed_results])
        uniq_rate_values = [float(result["uniq_rate"]) for result in seed_results if "uniq_rate" in result]
        summary = (
            f">>> For {task}/{search_space}/{prior_name}, "
            f"max_acc mean+std={max_acc_mean:.2f}+/-{max_acc_std:.2f}, "
            f"average duration={duration_mean:.2f}s"
        )
        if uniq_rate_values:
            summary += f", average uniqueness={np.mean(uniq_rate_values):.2f}"
        print(summary + ".")
        return {
            "completed_seeds": len(max_acc_values),
            "max_acc_mean": max_acc_mean,
            "max_acc_std": max_acc_std,
            "duration_mean": float(duration_mean),
            "uniq_rate_mean": float(np.mean(uniq_rate_values)) if uniq_rate_values else None,
            "log_path": log_path,
        }
    else:
        print(f">>> For {task}/{search_space}/{prior_name}, no completed max_acc results are available.")
        return {
            "completed_seeds": 0,
            "max_acc_mean": None,
            "max_acc_std": None,
            "duration_mean": None,
            "uniq_rate_mean": None,
            "log_path": log_path,
        }
