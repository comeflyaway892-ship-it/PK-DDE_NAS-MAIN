import os
import random

import numpy as np
import torch

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    eval_repeat_times,
    meta_dataset_list,
    meta_hyper_params_setting,
    nb201_api_path,
    nb201_dataset_list,
    nb201_hyper_params_setting,
    nb201_total_archs,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from experiments import get_topk_archs
from prior_exp.evo_diff_prior import evo_diff_meta_with_prior, evo_diff_with_prior
from utils.eval_arch import eval_architectures
from utils.nb201_fitness import get_nb201_arch_str, load_nb201_api


PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_dir(kind, dataset, prior_name):
    return os.path.join(PRIOR_EXP_ROOT, "results", kind, prior_name, dataset) + os.sep


def _mean_std(values):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute mean and std from an empty result list")
    return float(values.mean()), float(values.std())


def _atomic_torch_save(value, path):
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _describe_architectures(x, api):
    descriptions = []
    for rank, operation_matrix in enumerate(x.view(x.shape[0], 6), start=1):
        arch_str = get_nb201_arch_str(operation_matrix)
        descriptions.append(
            {
                "rank": rank,
                "arch_index": int(api.query_index_by_arch(arch_str)),
                "arch_str": arch_str,
                "operation_matrix": operation_matrix.detach().cpu().tolist(),
            }
        )
    return descriptions


def _completed_meta_seed(result, repeat_times):
    if not isinstance(result, dict) or result.get("status") != "complete":
        return False
    top_architectures = result.get("top_architectures")
    retrain_accuracies = result.get("retrain_accuracies")
    return (
        bool(top_architectures)
        and isinstance(retrain_accuracies, list)
        and len(retrain_accuracies) == len(top_architectures)
        and all(len(values) == repeat_times for values in retrain_accuracies)
        and "max_acc" in result
    )


def exp_with_fixed_seed_in_nb201_prior(dataset, *, prior_type, prior_name, prior_kwargs=None, seed_list=None):
    assert dataset in nb201_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)
    args = dict(nb201_hyper_params_setting[dataset])
    args["save_dir"] = _save_dir("nb201", dataset, prior_name)
    seed_list = args["seed"] if seed_list is None else list(seed_list)

    print(f">>> NB201 prior experiment: dataset={dataset}, prior={prior_name}, seeds={seed_list}")
    total_acc = [api.query_test_acc_by_index(i, dataset) for i in range(nb201_total_archs)]
    print(f">>> Benchmark avg acc: {np.mean(total_acc)}, max acc: {np.max(total_acc)}")

    result_log_path = os.path.join(args["save_dir"], "search_log", f"{dataset}_{prior_name}_search.pth")
    os.makedirs(os.path.dirname(result_log_path), exist_ok=True)
    result_log = torch.load(result_log_path, map_location="cpu") if os.path.exists(result_log_path) else {}
    max_acc_values = []
    avg_duration = torch.tensor(0.0)
    avg_uniq_rate = torch.tensor(0.0)
    for seed in seed_list:
        existing = result_log.get(seed)
        if existing is not None:
            if existing.get("num_step") != args["num_step"] or existing.get("population_num") != args["population_num"] or existing.get("prior_kwargs") != prior_kwargs:
                raise ValueError("Existing search settings differ; use a new prior_name")
            max_acc_values.append(existing["max_acc"])
            avg_duration += existing["duration"]
            avg_uniq_rate += existing["uniq_rate"]
            print(f">>> Reuse completed search seed {seed}")
            continue
        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} with seed {seed}...")
        max_acc, duration, uniq_rate, _ = evo_diff_with_prior(
            dataset=dataset,
            api=api,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=os.path.join(args["save_dir"], "reproduce_exp"),
            prior_type=prior_type,
            prior_kwargs=prior_kwargs,
            predictor_estimator_eps=predictor_estimator_eps,
            predictor_temperature=predictor_temperature,
            predictor_sigma=predictor_sigma,
            select_elite_frac=select_elite_frac,
            select_eps=select_eps,
            select_fill_uniform=select_fill_uniform,
        )
        print(
            f">>> Seed {seed}: max_acc={max_acc:.2f}, duration={duration:.2f} seconds, "
            f"uniqueness={uniq_rate:.2f}."
        )
        result_log[seed] = dict(status="complete", max_acc=float(max_acc), duration=float(duration),
                                uniq_rate=float(uniq_rate), num_step=args["num_step"],
                                population_num=args["population_num"], prior_kwargs=prior_kwargs)
        _atomic_torch_save(result_log, result_log_path)
        max_acc_values.append(max_acc)
        avg_duration += duration
        avg_uniq_rate += uniq_rate
    max_acc_mean, max_acc_std = _mean_std(max_acc_values)
    print(
        f">>> For {dataset}/{prior_name}, max_acc mean+std={max_acc_mean:.2f}+/-{max_acc_std:.2f}, "
        f"average duration={avg_duration / len(seed_list):.2f} seconds, "
        f"average uniqueness={avg_uniq_rate / len(seed_list):.2f}."
    )


def exp_with_fixed_seed_in_meta_predictor_prior(dataset, *, prior_type, prior_name, prior_kwargs=None, seed_list=None, retrain_artifact_dir=None):
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"
    api = load_nb201_api(nb201_api_path, verbose=False)
    args = dict(meta_hyper_params_setting[dataset])
    args["save_dir"] = _save_dir("meta", dataset, prior_name)
    seed_list = args["seed"] if seed_list is None else list(seed_list)

    print(f">>> NB201 meta prior experiment: dataset={dataset}, prior={prior_name}, seeds={seed_list}")
    result_log_path = os.path.join(args["save_dir"], "search_log", f"{dataset}_{prior_name}_search.pth")
    os.makedirs(os.path.dirname(result_log_path), exist_ok=True)
    result_log = torch.load(result_log_path, map_location="cpu") if os.path.exists(result_log_path) else {}
    retrain_artifact_dir = retrain_artifact_dir or os.path.join(args["save_dir"], "retrain_artifacts")

    for seed in seed_list:
        existing_result = result_log.get(seed)
        if isinstance(existing_result, dict) and existing_result.get("top_architectures"):
            top_architectures = existing_result["top_architectures"]
            x = torch.tensor(
                [architecture["operation_matrix"] for architecture in top_architectures],
                dtype=torch.long,
            )
            max_pred_acc = existing_result["max_pred_acc"]
            duration = existing_result["duration"]
            uniq_rate = existing_result["uniq_rate"]
            print(f"\n>>> Checking/resuming retraining for {dataset} with search seed {seed}...")
        else:
            set_random_seed(seed)
            print(f"\n>>> Running search on {dataset} with seed {seed}...")
            max_pred_acc, duration, uniq_rate, x = evo_diff_meta_with_prior(
                dataset=dataset,
                api=api,
                num_step=args["num_step"],
                population_num=args["population_num"],
                seed=seed,
                plot_results=True,
                save_dir=os.path.join(args["save_dir"], "reproduce_exp"),
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                predictor_estimator_eps=predictor_estimator_eps,
                predictor_temperature=predictor_temperature,
                predictor_sigma=predictor_sigma,
                select_elite_frac=select_elite_frac,
                select_eps=select_eps,
                select_fill_uniform=select_fill_uniform,
            )
            print(f"max_pred_acc:{max_pred_acc:.2f}\n")
            x = get_topk_archs(x=x, dataset=dataset, k=args["topk"], api=api, seed=seed).cpu()
            top_architectures = _describe_architectures(x, api)
            result_log[seed] = {
                "status": "retraining",
                "prior_type": prior_type,
                "prior_name": prior_name,
                "prior_kwargs": prior_kwargs,
                "max_pred_acc": max_pred_acc,
                "duration": duration,
                "uniq_rate": uniq_rate,
                "top_architectures": top_architectures,
            }
            _atomic_torch_save(result_log, result_log_path)
            print(f">>> Saved top-{len(top_architectures)} architectures before retraining.")

        max_acc, acc_list, retrain_accuracies = eval_architectures(
            x=x.cpu(),
            api=api,
            dataset_name=dataset,
            image_cutout=args["image_cutout"],
            batch_size=args["batch_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=args["LR"],
            momentum=args["momentum"],
            decay=args["decay"],
            nesterov=args["nesterov"],
            train_epochs=args["epochs"],
            warmup_epoch=args["warmup"],
            eta_min=args["eta_min"],
            multi_thread=args["multi_thread"],
            early_stop=args["early_stop"],
            repeat_times=eval_repeat_times,
            artifact_dir=retrain_artifact_dir,
            return_repeat_accuracies=True,
        )
        result_log[seed] = {
            "status": "complete",
            "prior_type": prior_type,
            "prior_name": prior_name,
            "prior_kwargs": prior_kwargs,
            "max_pred_acc": max_pred_acc,
            "max_acc": max_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "top_architectures": top_architectures,
            "retrain_accuracies": retrain_accuracies,
            "acc_list": acc_list,
        }
        _atomic_torch_save(result_log, result_log_path)

    seed_results = [
        result_log[seed]
        for seed in seed_list
        if seed in result_log and _completed_meta_seed(result_log[seed], eval_repeat_times)
    ]
    max_acc_values = [result["max_acc"] for result in seed_results if "max_acc" in result]
    if max_acc_values:
        max_acc_mean, max_acc_std = _mean_std(max_acc_values)
        duration_mean = np.mean([float(result["duration"]) for result in seed_results])
        uniq_rate_mean = np.mean([float(result["uniq_rate"]) for result in seed_results])
        print(
            f">>> For {dataset}/{prior_name}, max_acc mean+std={max_acc_mean:.2f}+/-{max_acc_std:.2f}, "
            f"average duration={duration_mean:.2f} seconds, average uniqueness={uniq_rate_mean:.2f}."
        )
    else:
        print(f">>> For {dataset}/{prior_name}, no completed max_acc results are available.")
