# NAS-Bench-201 Prior Experiments

This folder mirrors the MobileNetV3 `prior_exp` workflow without modifying the
original NAS-Bench-201 entry points.

Run from the repository root:

```bash
python NAS_Bench_201/prior_exp/main_prior.py --dataset cifar10 --prior-type structural --runs 3
python NAS_Bench_201/prior_exp/main_prior.py --dataset cifar10 --prior-type shuffled_structural --runs 3
python NAS_Bench_201/prior_exp/main_prior.py --dataset cifar10 --prior-type prior_Konwleage --runs 3
```

Priors:

- `structural`: token-distance kernel over tokens `0..4`.
- `shuffled_structural`: `P M P^T` negative control.
- `prior_Konwleage`: first generation stays uniform, then top-k initial
  candidates define a runtime marginal prior with `lumda` strength (default 0.99: 99% knowledge frequencies + 1% uniform). With
  `--knowledge-dynamic-steps N`, the top-k generated children from each of the
  first N evolution steps are retained, accumulated with all earlier knowledge,
  and used to rebuild the prior for the next step.

Run the dynamic knowledge prior with:

```bash
python NAS_Bench_201/prior_exp/main_prior.py \
  --dataset cifar10 \
  --prior-type prior_Konwleage \
  --knowledge-topk 10 \
  --knowledge-dynamic-steps 5 \
  --runs 3
```

Only the generated child population `x_next` is counted during the dynamic
steps, matching the MobileNetV3 implementation; the selected/merged population
is not counted. The result name receives a `_dynN` suffix automatically.

Results are saved under `NAS_Bench_201/prior_exp/results/`.
After all configured seeds finish, each experiment prints the final `max_acc`
as `mean+std` across those seeds.

## Aircraft and Pets: temperature 7, 30 steps / 20 candidates

```bash
python -u NAS_Bench_201/prior_exp/run_four_datasets_30steps.py \
  --datasets aircraft pets --runs 100 --steps 30 --population 20 \
  --knowledge-topk 6 --dynamic-steps 20 --temperature 7
```

The batch runner defaults to prior strength 0.99 and temperature 7 and overrides it inside each worker
before importing the experiment module. It does not edit `config/config.py`.
The temperature is included in `search_settings.json`, `summary.json`, and the
result directory name (`_l0.99_` and `_temp7`), so earlier search results are not reused.
Matching architecture retraining checkpoints are still reused; missing repeats
are trained. Rerunning the same configuration resumes its saved search results.
Use `--dry-run` to inspect the datasets and output directories without searching.

## Pets and Aircraft Full Training

Run seeds `0..99` for both meta datasets with 20 retained candidates and 40
dynamic knowledge steps:

```bash
python NAS_Bench_201/prior_exp/run_pets_aircraft_knowledge.py --gpu 0
```

The datasets run sequentially, in `pets`, `aircraft` order. Use `--datasets`
to run or resume only one of them, for example:

```bash
python NAS_Bench_201/prior_exp/run_pets_aircraft_knowledge.py --gpu 0 --datasets pets
```

Each completed seed is stored in the dataset's search log. When the command is
restarted, its saved top architectures and checkpoints are verified instead of
repeating the search. Each search seed trains the configured top architectures
(`pets`: 2, `aircraft`: 3) for 200 configured epochs plus 10 warmup epochs,
with three evaluation repeats per architecture.

For every search seed, the search log records the ranked top architectures,
the three retraining accuracies for each architecture, and their averages. The
log is written immediately after top-k selection, so an interrupted retraining
does not repeat the search. Final model, optimizer, and scheduler checkpoints
are stored by dataset, architecture index, and repeat under:

```text
prior_exp/results/meta/prior_Konwleage_top20_l0.99_dyn40/<dataset>/retrain_artifacts/
```

When an architecture is found again for the same dataset, each compatible
final checkpoint is reused and only missing repeats are trained. An
architecture with all three final checkpoints is not retrained.
