import hashlib
import json
import os
import random
import sys

from config.config import mobile_retrain_gpu

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(mobile_retrain_gpu))
os.environ.setdefault("DDE_NAG_GPU", str(mobile_retrain_gpu))
os.environ.setdefault("EDNAG_GPU", str(mobile_retrain_gpu))

import numpy as np
import torch

from config.config import (
    d3pm_cosine_s,
    d3pm_eps,
    d3pm_schedule,
    meta_dataset_list,
    meta_hyper_params_setting,
    mobile_retrain_autoaugment,
    mobile_retrain_batch_size,
    mobile_retrain_cutout,
    mobile_retrain_cutout_length,
    mobile_retrain_data_root,
    mobile_retrain_drop,
    mobile_retrain_drop_path,
    mobile_retrain_epochs,
    mobile_retrain_grad_clip,
    mobile_retrain_img_size,
    mobile_retrain_lr,
    mobile_retrain_momentum,
    mobile_retrain_report_freq,
    mobile_retrain_seeds,
    mobile_retrain_topk,
    mobile_retrain_weight_decay,
    mobile_retrain_workers,
    predictor_estimator_eps,
    predictor_sigma,
    predictor_temperature,
    select_elite_frac,
    select_eps,
    select_fill_uniform,
)
from evo_diff import evo_diff_meta

MOBILENETV3_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_EXP_ROOT = os.path.join(MOBILENETV3_ROOT, "main_exp")
if MAIN_EXP_ROOT not in sys.path:
    sys.path.insert(0, MAIN_EXP_ROOT)

from main_exp.transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.train import train_single_model
from utils.ofa_proxy import candidate_to_ofa_model_str


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _unique_arch_records(candidates, limit):
    arch_records = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.detach().cpu().view(-1)
        model_str = candidate_to_ofa_model_str(candidate)
        if model_str in seen:
            continue
        seen.add(model_str)
        arch_records.append(
            {
                "tokens": [int(v) for v in candidate.tolist()],
                "model_str": model_str,
            }
        )
        if len(arch_records) >= limit:
            break
    return arch_records


def _atomic_torch_save(value, path):
    temporary_path = path + ".tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _write_jsonl(path, records):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
    os.replace(temporary_path, path)


def _mobile_retrain_config():
    return {
        "workers": int(mobile_retrain_workers),
        "lr": float(mobile_retrain_lr),
        "momentum": float(mobile_retrain_momentum),
        "weight_decay": float(mobile_retrain_weight_decay),
        "report_freq": int(mobile_retrain_report_freq),
        "epochs": int(mobile_retrain_epochs),
        "grad_clip": float(mobile_retrain_grad_clip),
        "cutout": bool(mobile_retrain_cutout),
        "cutout_length": int(mobile_retrain_cutout_length),
        "autoaugment": bool(mobile_retrain_autoaugment),
        "drop": float(mobile_retrain_drop),
        "drop_path": float(mobile_retrain_drop_path),
        "img_size": int(mobile_retrain_img_size),
        "batch_size": int(mobile_retrain_batch_size),
    }


def _load_completed_retrain(checkpoint_path, dataset, model_str, retrain_seed):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:
        print(f"==> Ignore unreadable retrain checkpoint {checkpoint_path}: {error}")
        return None
    compatible = (
        isinstance(checkpoint, dict)
        and checkpoint.get("format_version") == 1
        and checkpoint.get("dataset") == dataset
        and checkpoint.get("model_str") == model_str
        and checkpoint.get("retrain_seed") == int(retrain_seed)
        and checkpoint.get("completed_epochs") == int(mobile_retrain_epochs)
        and checkpoint.get("training_config") == _mobile_retrain_config()
        and "model_state_dict" in checkpoint
        and "valid_acc" in checkpoint
        and "max_valid_acc" in checkpoint
    )
    if not compatible:
        print(f"==> Ignore incomplete or incompatible retrain checkpoint: {checkpoint_path}")
        return None
    return (
        float(checkpoint["valid_acc"]),
        float(checkpoint["max_valid_acc"]),
        float(checkpoint["params"]),
        float(checkpoint["flops"]),
    )


def _link_checkpoint(source_path, target_path):
    if os.path.abspath(source_path) == os.path.abspath(target_path):
        return
    temporary_path = target_path + ".tmp"
    if os.path.lexists(temporary_path):
        os.unlink(temporary_path)
    os.link(source_path, temporary_path)
    os.replace(temporary_path, target_path)


def _retrain_mobilenet_candidates(dataset, searched_archs, save_dir, search_seed=None):
    arch_records = _unique_arch_records(searched_archs, mobile_retrain_topk)
    if not arch_records:
        raise RuntimeError("No valid MobileNet/OFA candidates were produced for retraining.")

    dataset_retrain_dir = os.path.join(save_dir, "mobilenet_retrain", dataset)
    retrain_dir = dataset_retrain_dir
    if search_seed is not None:
        retrain_dir = os.path.join(retrain_dir, f"search_seed_{search_seed}")
    os.makedirs(retrain_dir, exist_ok=True)
    run_log_path = os.path.join(retrain_dir, "retrain_runs.jsonl")
    architecture_cache_dir = os.path.join(dataset_retrain_dir, "architecture_cache")

    results = {}
    all_run_records = []
    for arch_idx, arch_record in enumerate(arch_records):
        model_str = arch_record["model_str"]
        arch_dir = os.path.join(retrain_dir, f"arch_{arch_idx}")
        os.makedirs(arch_dir, exist_ok=True)
        architecture_key = hashlib.sha256(model_str.encode("utf-8")).hexdigest()[:16]
        cache_arch_dir = os.path.join(architecture_cache_dir, architecture_key)
        os.makedirs(cache_arch_dir, exist_ok=True)
        run_records = []
        net_info = None
        for seed in mobile_retrain_seeds:
            checkpoint_name = f"seed-{int(seed):04d}.pth"
            checkpoint_path = os.path.join(cache_arch_dir, checkpoint_name)
            cached_result = _load_completed_retrain(checkpoint_path, dataset, model_str, seed)
            was_cached = cached_result is not None
            if was_cached:
                print(f"==> Reuse MobileNet retrain for {dataset} arch_{arch_idx} seed {seed}")
                valid_acc, max_valid_acc, params, flops = cached_result
            else:
                print(f"==> MobileNet retrain for {dataset} arch_{arch_idx} seed {seed}")
                valid_acc, max_valid_acc, params, flops = train_single_model(
                    save_path=cache_arch_dir,
                    workers=mobile_retrain_workers,
                    datasets=dataset,
                    xpaths=os.path.join(mobile_retrain_data_root, dataset),
                    splits=[0],
                    use_less=False,
                    seed=seed,
                    model_str=model_str,
                    device="cuda",
                    lr=mobile_retrain_lr,
                    momentum=mobile_retrain_momentum,
                    weight_decay=mobile_retrain_weight_decay,
                    report_freq=mobile_retrain_report_freq,
                    epochs=mobile_retrain_epochs,
                    grad_clip=mobile_retrain_grad_clip,
                    cutout=mobile_retrain_cutout,
                    cutout_length=mobile_retrain_cutout_length,
                    autoaugment=mobile_retrain_autoaugment,
                    drop=mobile_retrain_drop,
                    drop_path=mobile_retrain_drop_path,
                    img_size=mobile_retrain_img_size,
                    batch_size=mobile_retrain_batch_size,
                )
            checkpoint_view_path = os.path.join(arch_dir, checkpoint_name)
            _link_checkpoint(checkpoint_path, checkpoint_view_path)
            run_record = {
                "dataset": dataset,
                "search_seed": search_seed,
                "arch_idx": arch_idx,
                "arch_tokens": arch_record["tokens"],
                "model_str": model_str,
                "retrain_seed": int(seed),
                "valid_acc": float(valid_acc),
                "max_valid_acc": float(max_valid_acc),
                "params": params,
                "flops": flops,
                "checkpoint_path": checkpoint_path,
                "cached": was_cached,
            }
            run_records.append(run_record)
            all_run_records.append(run_record)
            _write_jsonl(run_log_path, all_run_records)
            net_info = {"params": params, "flops": flops}

        acc_runs = [run["max_valid_acc"] for run in run_records]
        results[model_str] = {
            "arch_tokens": arch_record["tokens"],
            "model_str": model_str,
            "runs": run_records,
            "acc_runs": acc_runs,
            "best_acc": max(acc_runs),
            "mean_acc": float(np.mean(acc_runs)),
            "net_info": net_info,
        }

    _atomic_torch_save(results, os.path.join(retrain_dir, f"{dataset}_mobilenet_retrain.pth"))
    return results


def exp_with_fixed_seed_in_meta_predictor(dataset: str):
    assert dataset in meta_dataset_list, f"ERROR: invalid dataset {dataset}"

    args = meta_hyper_params_setting[dataset]
    seed_list = args["seed"]

    print(
        f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )

    log_dir = os.path.join(args["save_dir"], "search_log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset}_search.pth")
    result_log = torch.load(log_path) if os.path.exists(log_path) else {}

    total_duration = 0.0
    total_uniq_rate = 0.0
    completed = 0

    for seed in seed_list:
        if seed in result_log:
            if "mobilenet_retrain" not in result_log[seed] and "searched_archs" in result_log[seed]:
                print(f">>> Search log exists for seed {seed}; running missing MobileNet retraining.")
                result_log[seed]["mobilenet_retrain"] = _retrain_mobilenet_candidates(
                    dataset=dataset,
                    searched_archs=result_log[seed]["searched_archs"],
                    save_dir=args["save_dir"],
                    search_seed=seed,
                )
                torch.save(result_log, log_path)
            else:
                print(">>> Log already exists, pass this seed. Searching Log: ", result_log[seed])
            continue

        set_random_seed(seed)
        print(f"\n>>> Running on {dataset} dataset with seed {seed}...")
        max_pred_acc, duration, uniq_rate, searched_archs = evo_diff_meta(
            dataset=dataset,
            num_step=args["num_step"],
            population_num=args["population_num"],
            seed=seed,
            plot_results=True,
            save_dir=os.path.join(args["save_dir"], "reproduce_exp"),
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
            "max_pred_acc": max_pred_acc,
            "duration": duration,
            "uniq_rate": uniq_rate,
            "searched_archs": searched_archs.cpu(),
        }
        result_log[seed]["mobilenet_retrain"] = _retrain_mobilenet_candidates(
            dataset=dataset,
            searched_archs=searched_archs.cpu(),
            save_dir=args["save_dir"],
            search_seed=seed,
        )
        torch.save(result_log, log_path)

        total_duration += duration
        total_uniq_rate += uniq_rate
        completed += 1
        print(f"max_pred_acc:{max_pred_acc:.2f}\n")

    if completed > 0:
        print(
            f">>> For dataset {dataset}, average search duration is {total_duration / completed:.2f} seconds, "
            f"average uniqueness rate is {total_uniq_rate / completed:.2f}.\n"
        )
    else:
        print(f">>> For dataset {dataset}, all configured seeds already have search logs.\n")
