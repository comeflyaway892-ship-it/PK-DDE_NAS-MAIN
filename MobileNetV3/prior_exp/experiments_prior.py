import os

import torch

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    meta_dataset_list,
    meta_hyper_params_setting,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from experiments import _atomic_torch_save, _retrain_mobilenet_candidates, set_random_seed
from prior_exp.evo_diff_prior import evo_diff_meta_with_prior


PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))


def _prior_save_dir(dataset, prior_name):
    return os.path.join(PRIOR_EXP_ROOT, "results", prior_name, dataset) + os.sep


def exp_with_fixed_seed_in_meta_predictor_prior(
    dataset,
    *,
    prior_type="structural",
    prior_name="structural_tau1",
    prior_kwargs=None,
    seed_list=None,
    retrain=True,
):
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"

    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    args = meta_hyper_params_setting[dataset]
    args["save_dir"] = _prior_save_dir(dataset, prior_name)
    seed_list = args["seed"] if seed_list is None else list(seed_list)

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} / {prior_name} "
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    print(f">>> prior_type={prior_type}, prior_kwargs={prior_kwargs}")
    print(f">>> seeds={seed_list}")

    log_dir = os.path.join(args["save_dir"], "search_log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset}_{prior_name}_search.pth")
    result_log = torch.load(log_path, map_location="cpu") if os.path.exists(log_path) else {}

    total_duration = 0.0
    total_uniq_rate = 0.0
    completed = 0

    for seed in seed_list:
        if seed in result_log:
            if retrain and "searched_archs" in result_log[seed]:
                print(f">>> Search log exists for seed {seed}; checking/resuming MobileNet retraining.")
                result_log[seed]["mobilenet_retrain"] = _retrain_mobilenet_candidates(
                    dataset=dataset,
                    searched_archs=result_log[seed]["searched_archs"],
                    save_dir=args["save_dir"],
                    search_seed=seed,
                )
                _atomic_torch_save(result_log, log_path)
            else:
                print(">>> Log already exists, pass this seed. Searching Log: ", result_log[seed])
            continue

        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed} and prior {prior_name}...")
        max_pred_acc, duration, uniq_rate, searched_archs = evo_diff_meta_with_prior(
            dataset=dataset,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=os.path.join(args["save_dir"], "reproduce_exp"),
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
            "max_pred_acc": max_pred_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "searched_archs": searched_archs.cpu(),
        }
        _atomic_torch_save(result_log, log_path)
        print(f">>> Search result for seed {seed} saved before MobileNet retraining.")
        if retrain:
            result_log[seed]["mobilenet_retrain"] = _retrain_mobilenet_candidates(
                dataset=dataset,
                searched_archs=searched_archs.cpu(),
                save_dir=args["save_dir"],
                search_seed=seed,
            )
        _atomic_torch_save(result_log, log_path)

        total_duration += duration
        total_uniq_rate += uniq_rate
        completed += 1
        print(f"max_pred_acc:{max_pred_acc:.2f}\n")

    if completed > 0:
        print(
            f">>> For dataset {dataset} and prior {prior_name}, average search duration is "
            f"{total_duration / completed:.2f} seconds, average uniqueness rate is "
            f"{total_uniq_rate / completed:.2f}.\n"
        )
    else:
        print(f">>> For dataset {dataset} and prior {prior_name}, all configured seeds already have search logs.\n")
