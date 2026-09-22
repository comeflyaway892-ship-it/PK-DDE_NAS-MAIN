# DARTS / NAS-Bench-301

This repository contains the source code for a 22-token NAS-Bench-301
architecture representation, discrete diffusion evolutionary search, prior-aware
search experiments, and DARTS-style CIFAR-10/ImageNet retraining.

## Repository contents

- `main.py`, `evo_diff.py`: surrogate-guided architecture search.
- `prior_exp/`: structural and knowledge-prior search experiments.
- `darts_retrain/`: DARTS-style model, data, protocol, and training code.
- `pipeline.py`: resumable search and retraining pipeline.
- `profile_architectures.py`: parameter and inference-complexity profiling.
- `tests/`: local unit tests.
- `results/search/`: compact architecture/profile outputs.
- `results/experiment_summary.json`: final-only experiment summary.

Datasets, surrogate model weights, checkpoints, training logs, per-epoch
metrics, plots, and other generated artifacts are intentionally excluded from
this GitHub export. The official NAS-Bench-301 surrogate ensemble must be
obtained separately and passed with `--surrogate-path` (or placed under
`nb_models/xgb_v1.0/` locally).

## Architecture encoding

Each architecture has 22 tokens: 11 normal-cell tokens followed by 11
reduction-cell tokens. Operation tokens are:

```text
0 avg_pool_3x3       1 max_pool_3x3       2 skip_connect
3 sep_conv_3x3       4 sep_conv_5x5       5 dil_conv_3x3
6 dil_conv_5x5
```

The topology tokens encode the predecessor pair for each of the last three
intermediate nodes. See `config/config.py` and `utils/NB301.py` for the full
mapping.

## Search

Install the required Python dependencies and provide the official surrogate
ensemble separately, then run for example:

```bash
python main.py --seeds 0,1,2 --surrogate-path /path/to/xgb_v1.0
python prior_exp/main_prior.py --prior-type prior_knowledge --topk 10
```

## Retraining

CIFAR-10 uses the locked 600-epoch, 36-channel, 20-cell protocol. ImageNet
uses the locked 250-epoch, 48-channel, 14-cell protocol with AMP and effective
batch size 1024. Dataset paths must be supplied by the user:

```bash
python retrain.py \
  --dataset cifar10 \
  --tokens-file /path/to/architecture.json \
  --data /path/to/cifar10 \
  --output /path/to/output \
  --seed 0
```

Run the tests without the surrogate weights or datasets:

```bash
python -m unittest discover -s tests -v
```

## License and attribution

DARTS-derived files retain the accompanying attribution and license notices in
`darts_retrain/LICENSE-DARTS` and `darts_retrain/NOTICE.md`.
