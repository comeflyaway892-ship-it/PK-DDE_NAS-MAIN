"""Sequential NAS-Bench-301 search and official DARTS retraining pipeline."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from config.config import nb301_hyper_params_setting, nb301_surrogate_path
from darts_retrain.protocol import get_protocol
from darts_retrain.training import train_official_protocol
from utils.NB301 import (
    genotype_to_dict,
    load_nb301_surrogate,
    tokens_to_genotype,
    validate_tokens,
)
from utils.seeds import DEFAULT_SEEDS, parse_seed_list


MODULE_DIR = Path(__file__).resolve().parent
# NB301 uses the machine-level datasets directly.  Do not infer data paths
# from the neighboring NAS-Bench-201 or TransNASBench101 copies.
DEFAULT_CIFAR_DATA = Path("/data/datasets/cifar-10")
DEFAULT_IMAGENET_DATA = Path("/fasterdatasets/imagenet2012")
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "results" / "nb301_retrain"
IMAGENET_DOWNLOAD_PAGE = "https://www.image-net.org/download.php"
IMAGENET_KAGGLE_PAGE = (
    "https://www.kaggle.com/c/imagenet-object-localization-challenge/data"
)


def _resolve_module_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = MODULE_DIR / path
    return path.resolve()


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_architecture_artifacts(output_root, records):
    """Persist the canonical architecture both as a manifest and per seed."""
    architecture_dir = Path(output_root) / "search" / "architectures"
    for record in records:
        seed = int(record["search_seed"])
        tokens = validate_tokens(record["tokens"]).tolist()
        artifact = {
            "search_seed": seed,
            "prior_type": record.get("prior_type"),
            "tokens": tokens,
            "genotype": genotype_to_dict(tokens_to_genotype(tokens)),
            "predicted_accuracy": record.get("predicted_accuracy"),
        }
        _atomic_write_json(
            architecture_dir / "search_seed_{}.json".format(seed), artifact
        )


def _load_architecture_artifacts(output_root, seeds):
    """Recover valid per-seed architecture files missing from the manifest."""
    architecture_dir = Path(output_root) / "search" / "architectures"
    recovered = {}
    for seed in seeds:
        artifact_path = architecture_dir / "search_seed_{}.json".format(seed)
        if not artifact_path.is_file():
            continue
        try:
            artifact = _read_json(artifact_path)
            if int(artifact["search_seed"]) != int(seed):
                continue
            tokens = validate_tokens(artifact["tokens"]).tolist()
        except (KeyError, TypeError, ValueError, OSError):
            print(">>> Ignoring invalid architecture artifact: {}".format(artifact_path))
            continue
        normalized = dict(artifact)
        normalized["search_seed"] = int(seed)
        normalized["tokens"] = tokens
        normalized["genotype"] = genotype_to_dict(tokens_to_genotype(tokens))
        recovered[int(seed)] = normalized
    return recovered


def _check_output_root(path):
    """Fail before searching when the result location cannot be created."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError("Output root is not a directory: {}".format(path))
        if not os.access(str(path), os.W_OK):
            raise PermissionError("Output root is not writable: {}".format(path))
        return path

    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not os.access(str(ancestor), os.W_OK):
        raise PermissionError(
            "Cannot create output root {!s}; nearest existing parent {!s} is not writable. "
            "Pass --output-root to a writable directory.".format(path, ancestor)
        )
    return path


def inspect_cifar10(path):
    path = Path(path).expanduser().resolve()
    batch_dir = path / "cifar-10-batches-py"
    expected = [batch_dir / "data_batch_{}".format(index) for index in range(1, 6)]
    expected.append(batch_dir / "test_batch")
    present = batch_dir.is_dir() and all(item.is_file() for item in expected)
    return {
        "dataset": "cifar10",
        "path": str(path),
        "ready": present,
        "detail": "ready" if present else "not found; torchvision can download it",
    }


def inspect_imagenet(path):
    path = Path(path).expanduser().resolve()
    train_dir = path / "train"
    valid_dir = path / "val"
    train_classes = (
        sorted(item.name for item in train_dir.iterdir() if item.is_dir())
        if train_dir.is_dir()
        else []
    )
    valid_classes = (
        sorted(item.name for item in valid_dir.iterdir() if item.is_dir())
        if valid_dir.is_dir()
        else []
    )
    ready = (
        len(train_classes) == 1000
        and len(valid_classes) == 1000
        and train_classes == valid_classes
    )
    return {
        "dataset": "imagenet",
        "path": str(path),
        "ready": ready,
        "train_class_directories": len(train_classes),
        "val_class_directories": len(valid_classes),
        "detail": (
            "ready"
            if ready
            else "requires train/<WNID>/ and val/<WNID>/ with 1000 matching classes"
        ),
        "official_download": IMAGENET_DOWNLOAD_PAGE,
        "kaggle_download": IMAGENET_KAGGLE_PAGE,
    }


def preflight(cifar_data, imagenet_data, require_training, require_imagenet=False):
    report = {
        "cifar10": inspect_cifar10(cifar_data),
        "imagenet": inspect_imagenet(imagenet_data),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if require_training:
        if not report["cuda_available"]:
            raise RuntimeError("No CUDA device is visible for real retraining")
    if require_imagenet:
        if not report["imagenet"]["ready"]:
            raise FileNotFoundError(
                "ImageNet-1k is not ready at {}. Download: {}".format(
                    imagenet_data, IMAGENET_DOWNLOAD_PAGE
                )
            )
    return report


def _manifest_records_by_seed(manifest):
    records = manifest.get("architectures")
    if not isinstance(records, list):
        raise ValueError("Architecture manifest must contain an 'architectures' list")
    by_seed = {}
    for record in records:
        seed = int(record["search_seed"])
        if seed in by_seed:
            raise ValueError("Duplicate search seed {} in manifest".format(seed))
        tokens = validate_tokens(record["tokens"]).tolist()
        normalized = dict(record)
        normalized["search_seed"] = seed
        normalized["tokens"] = tokens
        normalized["genotype"] = genotype_to_dict(tokens_to_genotype(tokens))
        by_seed[seed] = normalized
    return by_seed


def _validate_manifest(manifest, seeds):
    by_seed = _manifest_records_by_seed(manifest)
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(
            "Manifest is missing requested search seeds: {}; available seeds: {}".format(
                missing, sorted(by_seed)
            )
        )
    return [by_seed[seed] for seed in seeds]


def _checkpoint_epoch(checkpoint_path, protocol, tokens):
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("protocol") != protocol.to_dict():
        raise ValueError("Checkpoint protocol mismatch: {}".format(checkpoint_path))
    if checkpoint.get("tokens") != tokens:
        raise ValueError("Checkpoint tokens mismatch: {}".format(checkpoint_path))
    expected_genotype = genotype_to_dict(tokens_to_genotype(tokens))
    if checkpoint.get("genotype") != expected_genotype:
        raise ValueError("Checkpoint genotype mismatch: {}".format(checkpoint_path))
    return int(checkpoint["epoch"])


def _load_or_initialize_state(path, seeds, manifest_path):
    if path.is_file():
        state = _read_json(path)
        if state.get("seeds") != seeds:
            print(
                ">>> Existing pipeline state uses seeds {}; starting a new state "
                "for requested seeds {}".format(state.get("seeds"), seeds)
            )
            return {
                "seeds": seeds,
                "architecture_manifest": str(manifest_path),
                "runs": {},
            }
        return state
    return {
        "seeds": seeds,
        "architecture_manifest": str(manifest_path),
        "runs": {},
    }


def search_stage(args, output_root, manifest_path):
    # Keep search-only plotting dependencies out of preflight/retrain startup.
    from main import run_searches

    if manifest_path.is_file() and not args.rerun_search:
        manifest = _read_json(manifest_path)
        stored_prior = manifest.get("prior_type", "uniform")
        requested_kwargs = {
            "tau": args.prior_tau,
            "shuffle_seed": args.shuffle_seed,
            "topk": args.topk,
            "lumda": args.lumda,
            "dynamic_steps": args.dynamic_steps,
        }
        stored_kwargs = manifest.get("prior_kwargs", {})
        if stored_prior != args.prior_type:
            raise ValueError(
                "Existing manifest uses prior_type={!r}, requested {!r}; "
                "use --rerun-search or a new --output-root".format(
                    stored_prior, args.prior_type
                )
            )
        if stored_kwargs != requested_kwargs:
            raise ValueError(
                "Existing manifest uses prior_kwargs={!r}, requested {!r}; "
                "use --rerun-search or a new --output-root".format(
                    stored_kwargs, requested_kwargs
                )
        )
        existing_by_seed = _manifest_records_by_seed(manifest)
        recovered_by_seed = _load_architecture_artifacts(
            output_root,
            [seed for seed in args.seeds if seed not in existing_by_seed],
        )
        existing_by_seed.update(recovered_by_seed)
        missing_seeds = [
            seed for seed in args.seeds if seed not in existing_by_seed
        ]
        if not missing_seeds:
            if recovered_by_seed:
                manifest = dict(manifest)
                manifest["seeds"] = sorted(existing_by_seed)
                manifest["architectures"] = [
                    existing_by_seed[seed] for seed in sorted(existing_by_seed)
                ]
                _atomic_write_json(manifest_path, manifest)
                print(
                    ">>> Recovered seeds {} from per-seed architecture files".format(
                        sorted(recovered_by_seed)
                    )
                )
            records = [existing_by_seed[seed] for seed in args.seeds]
            _save_architecture_artifacts(output_root, records)
            print(">>> Reusing existing Top-1 manifest: {}".format(manifest_path))
            return manifest
        print(
            ">>> Existing manifest is missing seeds {}; searching only those seeds".format(
                missing_seeds
            )
        )
    else:
        existing_by_seed = _load_architecture_artifacts(output_root, args.seeds)
        missing_seeds = list(args.seeds)
        missing_seeds = [seed for seed in args.seeds if seed not in existing_by_seed]
        if existing_by_seed:
            print(
                ">>> Recovered existing per-seed architectures {}; searching only missing seeds".format(
                    sorted(existing_by_seed)
                )
            )

    surrogate_path = _resolve_module_path(args.surrogate_path)
    print(">>> Loading official NAS-Bench-301 surrogate: {}".format(surrogate_path))
    predictor = load_nb301_surrogate(
        surrogate_path, with_noise=args.with_noise
    )
    settings = nb301_hyper_params_setting
    num_step = settings["num_step"] if args.num_step is None else args.num_step
    population_num = (
        settings["population_num"]
        if args.population_num is None
        else args.population_num
    )
    records = run_searches(
        predictor=predictor,
        seeds=missing_seeds,
        num_step=num_step,
        population_num=population_num,
        plot=args.plot,
        save_dir=output_root / "search" / "plots",
        prior_type=args.prior_type,
        prior_kwargs={
            "tau": args.prior_tau,
            "shuffle_seed": args.shuffle_seed,
            "topk": args.topk,
            "lumda": args.lumda,
            "dynamic_steps": args.dynamic_steps,
        },
    )
    merged_by_seed = dict(existing_by_seed)
    for record in records:
        merged_by_seed[int(record["search_seed"])] = record
    manifest = {
        "seeds": sorted(merged_by_seed),
        "surrogate_path": str(surrogate_path),
        "with_noise": bool(args.with_noise),
        "num_step": num_step,
        "population_num": population_num,
        "prior_type": args.prior_type,
        "prior_kwargs": {
            "tau": args.prior_tau,
            "shuffle_seed": args.shuffle_seed,
            "topk": args.topk,
            "lumda": args.lumda,
            "dynamic_steps": args.dynamic_steps,
        },
        "architectures": [merged_by_seed[seed] for seed in sorted(merged_by_seed)],
    }
    _atomic_write_json(manifest_path, manifest)
    _save_architecture_artifacts(output_root, records)
    print(
        ">>> Saved {}-seed Top-1 manifest: {}".format(
            len(merged_by_seed), manifest_path
        )
    )
    selected_records = _validate_manifest(manifest, args.seeds)
    _save_architecture_artifacts(output_root, selected_records)
    return manifest


def retrain_stage(args, output_root, manifest_path, manifest):
    records = _validate_manifest(manifest, args.seeds)
    _save_architecture_artifacts(output_root, records)
    state_path = output_root / "pipeline_state.json"
    state = _load_or_initialize_state(state_path, args.seeds, manifest_path)
    train_imagenet = bool(getattr(args, "train_imagenet", False))
    dataset_specs = [("cifar10", Path(args.cifar_data), args.cifar_workers)]
    if train_imagenet:
        dataset_specs.append(
            ("imagenet", Path(args.imagenet_data), args.imagenet_workers)
        )

    for record in records:
        search_seed = record["search_seed"]
        tokens = record["tokens"]
        genotype = tokens_to_genotype(tokens)
        if not train_imagenet:
            run_key = "imagenet:search_seed_{}".format(search_seed)
            run_dir = (
                output_root
                / "retrain"
                / "imagenet"
                / "search_seed_{}".format(search_seed)
            )
            state["runs"][run_key] = {
                "status": "skipped",
                "reason": "ImageNet retraining disabled; pass --train-imagenet to enable",
                "output": str(run_dir),
                "search_seed": search_seed,
                "training_seed": search_seed,
            }
            _atomic_write_json(state_path, state)
            print(">>> Skipping ImageNet retraining for search seed {}".format(search_seed))
        for dataset, data_path, workers in dataset_specs:
            protocol = get_protocol(dataset)
            run_key = "{}:search_seed_{}".format(dataset, search_seed)
            run_dir = (
                output_root
                / "retrain"
                / dataset
                / "search_seed_{}".format(search_seed)
            )
            checkpoint_path = run_dir / "last.pt"
            restart = bool(getattr(args, "restart", False))
            completed_epoch = (
                None
                if restart
                else _checkpoint_epoch(checkpoint_path, protocol, tokens)
            )
            if completed_epoch is not None and completed_epoch >= protocol.epochs:
                state["runs"][run_key] = {
                    "status": "complete",
                    "output": str(run_dir),
                    "epoch": completed_epoch,
                    "search_seed": search_seed,
                    "training_seed": search_seed,
                }
                _atomic_write_json(state_path, state)
                print(">>> Skipping completed run {}".format(run_key))
                continue

            resume = str(checkpoint_path) if completed_epoch is not None else None
            state["runs"][run_key] = {
                "status": "running",
                "output": str(run_dir),
                "resume": resume,
                "search_seed": search_seed,
                "training_seed": search_seed,
            }
            _atomic_write_json(state_path, state)
            print(">>> Starting {} with training seed {}".format(run_key, search_seed))
            try:
                best_top1 = train_official_protocol(
                    protocol=protocol,
                    genotype=genotype,
                    tokens=tokens,
                    data_path=data_path,
                    output_dir=run_dir,
                    seed=search_seed,
                    gpu=args.gpu,
                    parallel=args.parallel,
                    workers=workers,
                    resume=resume,
                    allow_existing=True,
                    restart=restart,
                )
            except Exception as error:
                state["runs"][run_key] = {
                    "status": "failed",
                    "output": str(run_dir),
                    "search_seed": search_seed,
                    "training_seed": search_seed,
                    "error": repr(error),
                }
                _atomic_write_json(state_path, state)
                raise
            state["runs"][run_key] = {
                "status": "complete",
                "output": str(run_dir),
                "epoch": protocol.epochs,
                "best_top1": float(best_top1),
                "search_seed": search_seed,
                "training_seed": search_seed,
            }
            _atomic_write_json(state_path, state)
    if train_imagenet:
        print(">>> All requested CIFAR-10 and ImageNet retraining runs are complete")
    else:
        print(">>> CIFAR-10 retraining complete; ImageNet retraining was skipped")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially search every seed's Top-1 NAS-Bench-301 architecture, "
            "then retrain each one on CIFAR-10; ImageNet is opt-in"
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "search", "retrain", "preflight"),
        default="all",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list(DEFAULT_SEEDS),
        help=(
            "explicit search and corresponding training seed list separated "
            "by English commas (default: 0-9)"
        ),
    )
    parser.add_argument("--surrogate-path", default=nb301_surrogate_path)
    parser.add_argument("--num-step", type=int, default=None)
    parser.add_argument("--population-num", type=int, default=None)
    parser.add_argument("--with-noise", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--prior-type",
        choices=("uniform", "structural", "shuffled_structural", "prior_knowledge"),
        default="uniform",
    )
    parser.add_argument("--prior-tau", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lumda", type=float, default=1.0)
    parser.add_argument("--dynamic-steps", type=int, default=0)
    parser.add_argument("--rerun-search", action="store_true")
    parser.add_argument("--cifar-data", default=str(DEFAULT_CIFAR_DATA))
    parser.add_argument("--imagenet-data", default=str(DEFAULT_IMAGENET_DATA))
    parser.add_argument(
        "--train-imagenet",
        action="store_true",
        help="enable real ImageNet retraining; disabled by default",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--architectures-file",
        default=None,
        help="existing Top-1 manifest for --stage retrain",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--cifar-workers",
        type=int,
        default=0,
        help="CIFAR DataLoader workers; 0 avoids shared-memory worker failures",
    )
    parser.add_argument("--imagenet-workers", type=int, default=None)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart requested CIFAR/ImageNet runs from epoch 0 and ignore old checkpoints",
    )
    return parser


def main(args):
    output_root = Path(args.output_root).expanduser().resolve()
    if args.stage != "preflight":
        _check_output_root(output_root)
        print(">>> Output root: {}".format(output_root))
    manifest_path = (
        Path(args.architectures_file).expanduser().resolve()
        if args.architectures_file
        else output_root / "search" / "top1_architectures.json"
    )
    if args.stage == "preflight":
        preflight(args.cifar_data, args.imagenet_data, require_training=False)
        return
    if args.stage in ("all", "retrain"):
        preflight(
            args.cifar_data,
            args.imagenet_data,
            require_training=True,
            require_imagenet=args.train_imagenet,
        )

    if args.stage in ("all", "search", "retrain"):
        manifest = search_stage(args, output_root, manifest_path)

    if args.stage in ("all", "retrain"):
        retrain_stage(args, output_root, manifest_path, manifest)


if __name__ == "__main__":
    main(build_parser().parse_args())
