"""Quick parameter/FLOPs profiler for saved NAS-Bench-301 architectures.

The profile uses the full-size PC-DARTS/DARTS final evaluation network, not the
search-time partial-channel network. Auxiliary classifiers are disabled at
construction time, so both parameters and inference FLOPs exclude auxiliary
heads. FLOPs use the common multiply-add convention: one multiply and one add
count as two FLOPs. Convolution and Linear layers are counted; BatchNorm,
ReLU, tensor additions, concatenations, and pooling are intentionally omitted.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from darts_retrain.protocol import get_protocol
from darts_retrain.training import build_model, count_parameters
from utils.NB301 import tokens_to_genotype, validate_tokens
from utils.seeds import parse_seed_list


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = MODULE_DIR / "results/nb301_retrain/search/top1_architectures.json"


def _conv_flops(module, inputs, output):
    output_elements = int(output.numel())
    kernel_ops = (
        module.kernel_size[0]
        * module.kernel_size[1]
        * (module.in_channels // module.groups)
    )
    return 2 * output_elements * kernel_ops


def _linear_flops(module, inputs, output):
    input_tensor = inputs[0]
    batch_size = int(input_tensor.numel() // input_tensor.shape[-1])
    return 2 * batch_size * module.in_features * module.out_features


def count_inference_flops(model, input_shape):
    flops = {"value": 0}

    def hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            return
        if isinstance(module, nn.Conv2d):
            flops["value"] += _conv_flops(module, inputs, output)
        elif isinstance(module, nn.Linear):
            flops["value"] += _linear_flops(module, inputs, output)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(input_shape, dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    return int(flops["value"])


def _load_records(path, seeds):
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_seed = {int(record["search_seed"]): record for record in manifest["architectures"]}
    selected = list(by_seed) if seeds is None else seeds
    missing = [seed for seed in selected if seed not in by_seed]
    if missing:
        raise ValueError(
            "Manifest is missing seeds {}; available seeds are {}".format(
                missing, sorted(by_seed)
            )
        )
    return [by_seed[seed] for seed in selected]


def profile_record(record, dataset):
    # The retraining protocol enables the auxiliary loss for optimization, but
    # the requested model-complexity report is for the deployable main network.
    # Disable the head before constructing the model; this avoids accidentally
    # including its parameters in the reported count.
    profile_protocol = replace(get_protocol(dataset), auxiliary=False)
    tokens = validate_tokens(record["tokens"])
    model = build_model(profile_protocol, tokens_to_genotype(tokens))
    if dataset == "cifar10":
        input_shape = (1, 3, 32, 32)
    else:
        input_shape = (1, 3, 224, 224)
    flops = count_inference_flops(model, input_shape)
    parameters = count_parameters(model)
    return {
        "search_seed": int(record["search_seed"]),
        "dataset": dataset,
        "input_shape": list(input_shape),
        "protocol": {
            "init_channels": profile_protocol.init_channels,
            "layers": profile_protocol.layers,
            "auxiliary": False,
        },
        "parameters": parameters,
        "parameters_millions": parameters / 1e6,
        # PC-DARTS-style complexity tables commonly report one multiply-add
        # as one MAC. Keep this alongside the strict 2-FLOP value below.
        "inference_macs": flops // 2,
        "inference_mmacs": flops / 2e6,
        "inference_flops": flops,
        "inference_gflops": flops / 1e9,
        "flops_convention": "2 per multiply-add; Conv2d and Linear only",
        "tokens": tokens.tolist(),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset", choices=("cifar10", "imagenet"), default="cifar10")
    parser.add_argument("--seeds", type=parse_seed_list, default=None)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser


def main(args):
    records = _load_records(args.manifest, args.seeds)
    results = [profile_record(record, args.dataset) for record in records]
    results.sort(key=lambda item: item["search_seed"])
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(build_parser().parse_args())
