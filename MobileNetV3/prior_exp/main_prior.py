import argparse
import os
import sys

PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
MOBILENETV3_ROOT = os.path.dirname(PRIOR_EXP_ROOT)
REPO_ROOT = os.path.dirname(MOBILENETV3_ROOT)
NAS_BENCH_201_ROOT = os.path.join(REPO_ROOT, "NAS_Bench_201")
for path in (REPO_ROOT, NAS_BENCH_201_ROOT, MOBILENETV3_ROOT):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from config.config import mobile_retrain_gpu


def _set_single_gpu(gpu):
    requested_gpu = str(gpu).strip()
    if not requested_gpu or "," in requested_gpu:
        raise ValueError(
            f"MobileNetV3 retraining expects one GPU id, got: {requested_gpu!r}"
        )

    inherited_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited_visible is None:
        # Backward-compatible mode: without an external visibility mask,
        # --gpu names the physical device that should become logical cuda:0.
        physical_gpu = requested_gpu
        mapping_source = f"--gpu {requested_gpu} selected the physical device"
    else:
        visible_devices = [item.strip() for item in inherited_visible.split(",") if item.strip()]
        if not visible_devices or visible_devices == ["-1"]:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES exposes no GPU; cannot start MobileNetV3 retraining"
            )
        try:
            logical_gpu = int(requested_gpu)
        except ValueError as exc:
            raise ValueError(
                "When CUDA_VISIBLE_DEVICES is already set, --gpu must be a logical "
                f"integer index, got: {requested_gpu!r}"
            ) from exc
        if logical_gpu < 0 or logical_gpu >= len(visible_devices):
            raise ValueError(
                f"--gpu {logical_gpu} is outside the logical range 0.."
                f"{len(visible_devices) - 1} exposed by "
                f"CUDA_VISIBLE_DEVICES={inherited_visible!r}"
            )
        physical_gpu = visible_devices[logical_gpu]
        mapping_source = (
            f"CUDA_VISIBLE_DEVICES={inherited_visible!r}, requested logical cuda:{logical_gpu}"
        )

    # Expose only the selected physical device. All project code consistently
    # uses logical cuda:0 after this point.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ["DDE_NAG_GPU"] = "0"
    os.environ["EDNAG_GPU"] = "0"
    os.environ["DDE_NAG_PHYSICAL_GPU"] = physical_gpu
    print(
        f"==> MobileNetV3 prior experiment GPU mapping: {mapping_source} -> "
        f"physical GPU {physical_gpu}; runtime device cuda:0",
        flush=True,
    )


def _build_prior_kwargs(args):
    prior_type = args.prior_type.lower()
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        return {
            "topk": args.knowledge_topk,
            "lumda": args.lumda,
            "dynamic_steps": args.knowledge_dynamic_steps,
        }

    prior_kwargs = {
        "depth_tau": args.depth_tau,
        "op_tau": args.op_tau,
        "kernel_weight": args.kernel_weight,
        "expand_weight": args.expand_weight,
        "depth_weight": args.depth_weight,
    }
    if prior_type == "shuffled_structural":
        prior_kwargs.update(
            {
                "shuffle_seed": args.shuffle_seed,
                "shuffle_depth": args.shuffle_depth,
            }
        )
    return prior_kwargs


def _default_prior_name(args):
    prior_type = args.prior_type.lower()
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        name = f"prior_Konwleage_top{args.knowledge_topk}_l{args.lumda:g}"
        if args.knowledge_dynamic_steps > 0:
            name += f"_dyn{args.knowledge_dynamic_steps}"
        return name

    base = (
        f"dt{args.depth_tau:g}_ot{args.op_tau:g}_kw{args.kernel_weight:g}_ew{args.expand_weight:g}"
    )
    if prior_type == "shuffled_structural":
        return f"shuffled_s{args.shuffle_seed}_{base}"
    return f"structural_{base}"


def _seed_list_from_args(args):
    if args.seeds:
        return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if args.runs is not None:
        if args.runs <= 0:
            raise ValueError(f"--runs must be positive, got {args.runs}")
        return list(range(args.runs))
    return None


def main(args):
    _set_single_gpu(args.gpu)

    from prior_exp.experiments_prior import exp_with_fixed_seed_in_meta_predictor_prior

    dataset_name = {"cifar10": "cifar10", "cifar100": "cifar100", "aircraft": "aircraft", "pets": "pets"}
    dataset = args.dataset.lower()
    assert dataset in dataset_name, f"ERROR: invalid dataset {dataset}"

    prior_kwargs = _build_prior_kwargs(args)
    prior_name = args.prior_name or _default_prior_name(args)
    exp_with_fixed_seed_in_meta_predictor_prior(
        dataset=dataset_name[dataset],
        prior_type=args.prior_type,
        prior_name=prior_name,
        prior_kwargs=prior_kwargs,
        seed_list=_seed_list_from_args(args),
        retrain=not args.skip_retrain,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV3/OFA prior experiment")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, cifar100, aircraft, pets")
    parser.add_argument(
        "--gpu",
        type=str,
        default=mobile_retrain_gpu,
        help=(
            "logical GPU index when CUDA_VISIBLE_DEVICES is set; otherwise the "
            "physical GPU id to expose"
        ),
    )
    parser.add_argument(
        "--prior-type",
        type=str,
        default="structural",
        choices=["structural", "shuffled_structural", "prior_Konwleage"],
        help="prior kernel to use",
    )
    parser.add_argument("--prior-name", type=str, default=None, help="name used under prior_exp/results")
    parser.add_argument("--depth-tau", type=float, default=1.0, help="depth-token structural prior temperature")
    parser.add_argument("--op-tau", type=float, default=1.0, help="operation-token structural prior temperature")
    parser.add_argument("--kernel-weight", type=float, default=1.0, help="distance weight for kernel size")
    parser.add_argument("--expand-weight", type=float, default=1.0, help="distance weight for expansion ratio")
    parser.add_argument("--depth-weight", type=float, default=1.0, help="distance weight for depth value")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="permutation seed for shuffled_structural")
    parser.add_argument(
        "--shuffle-depth",
        action="store_true",
        help="also shuffle depth-token semantics for shuffled_structural",
    )
    parser.add_argument("--knowledge-topk", type=int, default=10, help="top-k first-generation candidates used by prior_Konwleage")
    parser.add_argument(
        "--knowledge-dynamic-steps",
        type=int,
        default=0,
        help="also add top-k generated children from the first N evolution steps to prior_Konwleage",
    )
    parser.add_argument(
        "--lumda",
        type=float,
        default=0.99,
        help="prior_Konwleage strength: 1 uses top-k frequencies, 0 recovers uniform",
    )
    parser.add_argument("--runs", type=int, default=None, help="run the first N search seeds, e.g. 3 -> seeds 0,1,2")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated search seeds, e.g. 0,1,2")
    parser.add_argument("--skip-retrain", action="store_true", help="only run search/proxy evaluation")
    main(parser.parse_args())
