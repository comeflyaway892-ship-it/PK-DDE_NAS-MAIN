# PK-DDE-NAS

PK-DDE-NAS is a structured discrete diffusion evolution framework for neural architecture search. Architectures are represented as categorical token sequences. The search uses a D3PM-inspired posterior recovery step, fitness-guided target-state estimation, diffusion consistency, and a dynamic prior estimated from high-fitness architectures.

This GitHub release contains the source code, configuration, experiment entry points, and the final summary files needed to inspect the reported experiments. It deliberately excludes benchmark datasets, checkpoints, generated populations, raw search logs, and figures. `NAS_Bench_201_EDNAG` and `NAS_Bench_301` are not included.

## Environment

Use Python 3.9 or newer and install the common dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The benchmark files and pretrained assets are intentionally external to this repository. Before running an experiment, provide the files expected by the selected project configuration:

- NAS-Bench-201 API files for the NAS-Bench-201 experiments.
- Meta-predictor datasets and weights for the Pets and Aircraft experiments.
- The TransNASBench-101 benchmark file for TransNASBench-101 experiments.
- OFA/MobileNetV3 data and pretrained assets for MobileNetV3 experiments.

These files are not redistributed here. Set the paths in the corresponding project configuration or use the command-line path options where provided.

## Reproducing the main NAS-Bench-201 prior experiments

The released configuration uses a population of 20, 30 search steps, top-6 prior knowledge, 20 dynamic-prior updates, temperature 7, and prior strength 0.99:

```bash
python -u NAS_Bench_201/prior_exp/run_four_datasets_30steps.py \
  --datasets aircraft pets \
  --runs 100 \
  --steps 30 \
  --population 20 \
  --knowledge-topk 6 \
  --dynamic-steps 20 \
  --temperature 7
```

The runner does not edit `config/config.py`. It writes a new result directory and a final `summary.json` after all requested seeds finish.

## Component ablation and transition-kernel comparison

For the NAS-Bench-201 ImageNet16-120 ablations:

```bash
bash NAS_Bench_201/ablation/run_table6.sh
bash NAS_Bench_201/ablation/run_table7.sh
```

The default ablation setting is 20 individuals, 15 generations, dynamic-prior updates during the first 10 generations, temperature 7, and 100 seeds. Use the `--dry-run` option of the Python entry points to inspect the resolved settings without loading the benchmark.

## TransNASBench-101

Run the 100-seed temperature-7 experiment with population 20 and 30 generations:

```bash
python -u TransNASBench101/prior_exp/run_knowledge_100seeds.py
```

For a configuration preview:

```bash
python TransNASBench101/prior_exp/run_knowledge_100seeds.py --dry-run
```

## MobileNetV3

A search-only smoke run can be started with:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset cifar10 \
  --gpu 0 \
  --skip-retrain
```

The full prior experiment entry point is documented in `MobileNetV3/prior_exp/README.md`.

## Retained result summaries

Only the final summary files from the latest recorded runs are retained. They include the NAS-Bench-201 prior searches, the latest component and transition-kernel ablations, the TransNASBench-101 run, the MobileNetV3 table summary, and the latest population-trace summaries. Per-seed logs, architecture files, checkpoints, images, and intermediate traces were omitted.

The retained summaries are kept under the original experiment paths so that their parameter settings remain clear. They are result snapshots only; rerunning an experiment writes new files after the required external benchmark assets are installed.

## Scope and exclusions

The following are intentionally absent from this release:

- `NAS_Bench_201_EDNAG` and `NAS_Bench_301` source trees;
- all datasets and benchmark API databases;
- all pretrained and retraining checkpoints;
- experiment logs, status files, timing files, and per-seed search records;
- generated architecture populations and raw per-step traces;
- PNG, JPEG, SVG, and other generated figures.
