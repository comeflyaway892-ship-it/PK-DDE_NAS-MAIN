from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_DIR)
ORIGINAL_MOBILENETV3_ROOT = os.path.join(REPO_ROOT, "MobileNetV3")
MAIN_EXP_ROOT = os.path.join(PROJECT_DIR, "main_exp")
for path in (ORIGINAL_MOBILENETV3_ROOT, PROJECT_DIR, MAIN_EXP_ROOT):
    if path in sys.path:
        sys.path.remove(path)
for path in (ORIGINAL_MOBILENETV3_ROOT, PROJECT_DIR, MAIN_EXP_ROOT):
    sys.path.insert(0, path)

from config.config import (  # noqa: E402
    meta_dataset_list,
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
    mobile_retrain_weight_decay,
    mobile_retrain_workers,
)


def set_single_gpu(gpu: str):
    gpu = str(gpu).strip()
    if "," in gpu:
        raise ValueError(f"MobileNetV3 retraining expects one GPU id, got: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["DDE_NAG_GPU"] = gpu
    os.environ["EDNAG_GPU"] = gpu
    print(f"==> MobileNetV3 ablation retraining uses physical GPU {gpu}", flush=True)


def load_runtime_deps():
    global np
    global torch
    global candidate_to_ofa_model_str
    global train_single_model

    import numpy as _np
    import torch as _torch

    from main_exp.transfer_nag_lib.MetaD2A_mobilenetV3.evaluation.train import (
        train_single_model as _train_single_model,
    )
    from utils.ofa_proxy import candidate_to_ofa_model_str as _candidate_to_ofa_model_str

    np = _np
    torch = _torch
    candidate_to_ofa_model_str = _candidate_to_ofa_model_str
    train_single_model = _train_single_model


def parse_int_list(value: str | None):
    if value is None or value.strip() == "":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_str_list(value: str | None):
    if value is None or value.strip() == "":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def arch_hash(tokens):
    payload = ",".join(str(int(token)) for token in tokens)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def method_name(record):
    return str(record.get("variant") or record.get("algorithm") or "unknown")


def read_ablation_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip invalid json line {line_no}: {exc}", flush=True)
                continue
            record["_source_line"] = line_no
            yield record


def load_completed_runs(path: Path):
    completed = {}
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = run_key(record)
            completed[key] = record
    return completed


def run_key(record):
    return (
        record.get("source_file"),
        record.get("dataset"),
        record.get("method"),
        int(record.get("search_seed")),
        record.get("arch_hash"),
        int(record.get("retrain_seed")),
    )


def append_jsonl(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def write_summary(run_log_path: Path, summary_path: Path):
    groups = {}
    for record in load_completed_runs(run_log_path).values():
        key = (
            record["source_file"],
            record["dataset"],
            record["method"],
            record["search_seed"],
            record["arch_hash"],
        )
        groups.setdefault(key, []).append(record)

    summaries = []
    for (
        source_file,
        dataset,
        method,
        search_seed,
        hash_value,
    ), records in sorted(groups.items()):
        acc_runs = [float(item["max_valid_acc"]) for item in records]
        first = records[0]
        summaries.append(
            {
                "source_file": source_file,
                "dataset": dataset,
                "method": method,
                "search_seed": int(search_seed),
                "source_line": int(first["source_line"]),
                "search_best_acc": float(first["search_best_acc"]),
                "arch_hash": hash_value,
                "arch_tokens": first["arch_tokens"],
                "model_str": first["model_str"],
                "retrain_seeds": [int(item["retrain_seed"]) for item in records],
                "acc_runs": acc_runs,
                "best_acc": max(acc_runs),
                "mean_acc": float(np.mean(acc_runs)),
                "std_acc": float(np.std(acc_runs)),
                "params": first["params"],
                "flops": first["flops"],
                "run_count": len(records),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
    return summaries


def keep_record(record, dataset_filter, method_filter, seed_filter):
    if "best_arch" not in record:
        return False
    if dataset_filter is not None and record.get("dataset") not in dataset_filter:
        return False
    if method_filter is not None and method_name(record) not in method_filter:
        return False
    if seed_filter is not None and int(record.get("seed")) not in seed_filter:
        return False
    return True


def retrain_record(args, source_path: Path, record, completed_runs, run_log_path: Path):
    dataset = record["dataset"]
    method = method_name(record)
    search_seed = int(record["seed"])
    tokens = [int(value) for value in record["best_arch"]]
    hash_value = arch_hash(tokens)
    model_str = candidate_to_ofa_model_str(torch.tensor(tokens, dtype=torch.long))
    arch_dir = (
        Path(args.output_dir)
        / dataset
        / method
        / f"search_seed_{search_seed}"
        / f"arch_{hash_value}"
    )

    print(
        f"==> retrain {dataset}/{method}/search_seed_{search_seed}/arch_{hash_value}",
        flush=True,
    )
    print(f"    search best_acc={float(record['best_acc']):.4f}", flush=True)
    print(f"    model_str={model_str}", flush=True)

    run_records = []
    for retrain_seed in args.retrain_seeds:
        base_record = {
            "source_file": str(source_path),
            "source_line": int(record["_source_line"]),
            "dataset": dataset,
            "method": method,
            "search_seed": search_seed,
            "search_best_acc": float(record["best_acc"]),
            "arch_hash": hash_value,
            "arch_tokens": tokens,
            "model_str": model_str,
            "retrain_seed": int(retrain_seed),
        }
        key = run_key(base_record)
        if key in completed_runs and not args.overwrite:
            print(f"    skip logged retrain_seed={retrain_seed}", flush=True)
            run_records.append(completed_runs[key])
            continue

        if args.dry_run:
            print(f"    dry-run retrain_seed={retrain_seed}", flush=True)
            continue

        started = time.time()
        print(f"    run retrain_seed={retrain_seed}", flush=True)
        valid_acc, max_valid_acc, params, flops = train_single_model(
            save_path=str(arch_dir),
            workers=args.workers,
            datasets=dataset,
            xpaths=os.path.join(mobile_retrain_data_root, dataset),
            splits=[0],
            use_less=False,
            seed=int(retrain_seed),
            model_str=model_str,
            device="cuda",
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            report_freq=args.report_freq,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
            cutout=args.cutout,
            cutout_length=args.cutout_length,
            autoaugment=args.autoaugment,
            drop=args.drop,
            drop_path=args.drop_path,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
        run_record = {
            **base_record,
            "valid_acc": float(valid_acc),
            "max_valid_acc": float(max_valid_acc),
            "params": float(params),
            "flops": float(flops),
            "duration_sec": time.time() - started,
            "save_dir": str(arch_dir),
        }
        append_jsonl(run_log_path, run_record)
        completed_runs[key] = run_record
        run_records.append(run_record)
    return run_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain MobileNetV3/OFA architectures from ablation jsonl results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "core_modules_aircraft_3runs.jsonl"),
        help="Ablation jsonl file containing best_arch records.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_log", type=str, default=None)
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Comma-separated dataset filter.")
    parser.add_argument("--variant", type=str, default=None, help="Comma-separated variant/algorithm filter.")
    parser.add_argument("--seed", type=str, default=None, help="Comma-separated ablation search seed filter.")
    parser.add_argument("--retrain_seeds", type=str, default=",".join(str(seed) for seed in mobile_retrain_seeds))
    parser.add_argument("--gpu", type=str, default=os.environ.get("DDE_NAG_GPU", os.environ.get("EDNAG_GPU", "1")))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--workers", type=int, default=mobile_retrain_workers)
    parser.add_argument("--epochs", type=int, default=mobile_retrain_epochs)
    parser.add_argument("--batch_size", type=int, default=mobile_retrain_batch_size)
    parser.add_argument("--lr", type=float, default=mobile_retrain_lr)
    parser.add_argument("--momentum", type=float, default=mobile_retrain_momentum)
    parser.add_argument("--weight_decay", type=float, default=mobile_retrain_weight_decay)
    parser.add_argument("--report_freq", type=int, default=mobile_retrain_report_freq)
    parser.add_argument("--grad_clip", type=float, default=mobile_retrain_grad_clip)
    cutout_group = parser.add_mutually_exclusive_group()
    cutout_group.add_argument("--cutout", dest="cutout", action="store_true")
    cutout_group.add_argument("--no_cutout", dest="cutout", action="store_false")
    parser.set_defaults(cutout=mobile_retrain_cutout)
    parser.add_argument("--cutout_length", type=int, default=mobile_retrain_cutout_length)
    autoaugment_group = parser.add_mutually_exclusive_group()
    autoaugment_group.add_argument("--autoaugment", dest="autoaugment", action="store_true")
    autoaugment_group.add_argument("--no_autoaugment", dest="autoaugment", action="store_false")
    parser.set_defaults(autoaugment=mobile_retrain_autoaugment)
    parser.add_argument("--drop", type=float, default=mobile_retrain_drop)
    parser.add_argument("--drop_path", type=float, default=mobile_retrain_drop_path)
    parser.add_argument("--img_size", type=int, default=mobile_retrain_img_size)
    args = parser.parse_args()

    args.dataset = parse_str_list(args.dataset)
    if args.dataset is not None:
        invalid = sorted(args.dataset - set(meta_dataset_list))
        if invalid:
            raise ValueError(f"invalid dataset filter: {invalid}")
    args.variant = parse_str_list(args.variant)
    args.seed = parse_int_list(args.seed)
    args.retrain_seeds = sorted(parse_int_list(args.retrain_seeds) or set(mobile_retrain_seeds))

    input_path = Path(args.input)
    default_root = input_path.parent / "retrain" / input_path.stem
    args.output_dir = args.output_dir or str(default_root)
    args.run_log = args.run_log or str(default_root / "retrain_runs.jsonl")
    args.summary = args.summary or str(default_root / "retrain_summary.jsonl")
    return args


def main():
    args = parse_args()
    set_single_gpu(args.gpu)
    load_runtime_deps()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"ablation result file does not exist: {input_path}")

    run_log_path = Path(args.run_log)
    completed_runs = load_completed_runs(run_log_path)
    selected = [
        record
        for record in read_ablation_records(input_path)
        if keep_record(record, args.dataset, args.variant, args.seed)
    ]

    print(f"input: {input_path}", flush=True)
    print(f"selected records: {len(selected)}", flush=True)
    print(f"retrain seeds: {args.retrain_seeds}", flush=True)
    print(f"output dir: {args.output_dir}", flush=True)
    print(f"run log: {run_log_path}", flush=True)

    for record in selected:
        retrain_record(args, input_path, record, completed_runs, run_log_path)

    if not args.dry_run:
        summaries = write_summary(run_log_path, Path(args.summary))
        print(f"summary: {args.summary} ({len(summaries)} architectures)", flush=True)


if __name__ == "__main__":
    main()
