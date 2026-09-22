import argparse
import os
import sys


PRIOR_EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
TRANS_ROOT = os.path.dirname(PRIOR_EXP_ROOT)
REPO_ROOT = os.path.dirname(TRANS_ROOT)
for path in (REPO_ROOT, TRANS_ROOT):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
os.chdir(TRANS_ROOT)


def _seed_list(args):
    if args.seeds:
        return [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if args.runs is not None:
        if args.runs <= 0:
            raise ValueError(f"--runs must be positive, got {args.runs}")
        return list(range(args.runs))
    return None


def _prior_kwargs(args):
    prior_type = args.prior_type.lower()
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        return {
            "topk": args.knowledge_topk,
            "lumda": args.lumda,
            "dynamic_steps": args.knowledge_dynamic_steps,
        }
    if prior_type == "shuffled_structural":
        return {"tau": args.tau, "shuffle_seed": args.shuffle_seed}
    return {"tau": args.tau}


def _prior_name(args):
    if args.prior_name:
        return args.prior_name
    prior_type = args.prior_type.lower()
    if prior_type == "shuffled_structural":
        return f"shuffled_s{args.shuffle_seed}_tau{args.tau:g}"
    if prior_type in ("prior_konwleage", "prior_knowledge", "knowledge"):
        name = f"prior_Konwleage_top{args.knowledge_topk}_l{args.lumda:g}"
        if args.knowledge_dynamic_steps > 0:
            name += f"_dyn{args.knowledge_dynamic_steps}"
        return name
    return f"structural_tau{args.tau:g}"


def main(args):
    from prior_exp.experiments_prior import main_exp_prior

    main_exp_prior(
        task=args.task,
        search_space=args.search_space,
        prior_type=args.prior_type,
        prior_name=_prior_name(args),
        prior_kwargs=_prior_kwargs(args),
        seed_list=_seed_list(args),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransNASBench101 prior experiment")
    parser.add_argument("--task", type=str, default="class_scene")
    parser.add_argument("--search-space", type=str, default="macro", choices=["macro", "micro"])
    parser.add_argument(
        "--prior-type",
        type=str,
        default="structural",
        choices=["structural", "shuffled_structural", "prior_Konwleage"],
    )
    parser.add_argument("--prior-name", type=str, default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--knowledge-topk", type=int, default=10)
    parser.add_argument(
        "--knowledge-dynamic-steps",
        type=int,
        default=0,
        help="also add top-k generated children from the first N evolution steps to prior_Konwleage",
    )
    parser.add_argument("--lumda", type=float, default=0.99)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    main(parser.parse_args())
