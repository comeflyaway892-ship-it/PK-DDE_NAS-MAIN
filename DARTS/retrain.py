"""Retrain one NAS-Bench-301 architecture with official DARTS protocols."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from darts_retrain.protocol import get_protocol
from darts_retrain.training import (
    build_model,
    count_parameters,
    train_official_protocol,
)
from utils.NB301 import genotype_to_dict, tokens_to_genotype, validate_tokens


MODULE_DIR = Path(__file__).resolve().parent


def parse_tokens_text(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [int(item) for item in re.findall(r"-?\d+", text)]
    if isinstance(value, dict):
        value = value.get("tokens")
    if not isinstance(value, (list, tuple)):
        raise ValueError("Tokens must be a JSON list or a dict containing 'tokens'")
    return validate_tokens(value)


def load_tokens(args):
    if args.tokens is not None:
        return parse_tokens_text(args.tokens)
    token_path = Path(args.tokens_file).expanduser().resolve()
    return parse_tokens_text(token_path.read_text(encoding="utf-8"))


def default_output_dir(dataset, tokens, seed):
    architecture_id = hashlib.sha1(
        ",".join(str(int(value)) for value in tokens).encode("ascii")
    ).hexdigest()[:10]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return (
        MODULE_DIR
        / "retrain_runs"
        / dataset
        / "arch_{}_seed_{}_{}".format(architecture_id, seed, timestamp)
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a 22-token NAS-Bench-301 architecture using the locked "
            "official 2018 DARTS evaluation protocol"
        )
    )
    parser.add_argument("--dataset", required=True, choices=("cifar10", "imagenet"))
    architecture = parser.add_mutually_exclusive_group(required=True)
    architecture.add_argument(
        "--tokens", help="22 integers as JSON, comma-separated, or space-separated text"
    )
    architecture.add_argument(
        "--tokens-file", help="JSON/text file containing the 22 tokens"
    )
    parser.add_argument("--data", help="CIFAR root or ImageNet root with train/ and val/")
    parser.add_argument("--output", help="new run directory; a timestamped default is used")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--parallel", action="store_true", help="official DataParallel mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="data-loader workers only; does not change the learning protocol",
    )
    parser.add_argument("--resume", help="path to last.pt from the same run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate tokens and build the standard model without data or CUDA",
    )
    return parser


def main(args):
    protocol = get_protocol(args.dataset)
    tokens = load_tokens(args)
    genotype = tokens_to_genotype(tokens)
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_dir(args.dataset, tokens, args.seed)
    )

    if args.dry_run:
        model = build_model(protocol, genotype)
        summary = {
            "protocol": protocol.to_dict(),
            "tokens": tokens.tolist(),
            "genotype": genotype_to_dict(genotype),
            "parameters_no_auxiliary": count_parameters(model, True),
            "parameters_total": count_parameters(model, False),
            "output": str(output_dir),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.data is None:
        raise ValueError("--data is required for real training")
    best_top1 = train_official_protocol(
        protocol=protocol,
        genotype=genotype,
        tokens=tokens.tolist(),
        data_path=args.data,
        output_dir=output_dir,
        seed=args.seed,
        gpu=args.gpu,
        parallel=args.parallel,
        workers=args.workers,
        resume=args.resume,
    )
    print("training complete: best top-1 {:.4f}".format(best_top1))


if __name__ == "__main__":
    main(build_parser().parse_args())
