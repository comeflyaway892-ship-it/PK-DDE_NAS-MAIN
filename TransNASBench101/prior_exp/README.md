# TransNASBench101 Prior Experiments

This folder mirrors the MobileNetV3 `prior_exp` workflow without modifying the
original TransNASBench101 entry points.

Run from the repository root:

```bash
python TransNASBench101/prior_exp/main_prior.py --task class_scene --search-space macro --prior-type structural --runs 3
python TransNASBench101/prior_exp/main_prior.py --task class_scene --search-space macro --prior-type shuffled_structural --runs 3
python TransNASBench101/prior_exp/main_prior.py --task class_scene --search-space macro --prior-type prior_Konwleage --runs 3
```

Priors:

- `structural`: token-distance kernel over tokens `0..3`.
- `shuffled_structural`: `P M P^T` negative control.
- `prior_Konwleage`: first generation stays uniform, then top-k initial
  candidates define a runtime marginal prior with `lumda` strength (default 0.99: 99% knowledge frequencies + 1% uniform).
  For macro search, position `0` and positions `1..6` are counted separately;
  for micro search, all six positions are counted together.

To retain and accumulate the top-k generated children from the first N search
steps, use the dynamic knowledge prior:

```bash
python TransNASBench101/prior_exp/main_prior.py \
  --task class_scene \
  --search-space macro \
  --prior-type prior_Konwleage \
  --knowledge-topk 10 \
  --knowledge-dynamic-steps 10 \
  --runs 3
```

Only `x_next` is added during the dynamic steps. The selected/merged population
is not counted, and the result name receives a `_dynN` suffix automatically.

Results are saved under `TransNASBench101/prior_exp/results/`.
After all configured seeds finish, each experiment prints `max_acc` as
`mean+std` across those seeds, including seeds loaded from existing logs.

## Full Sweep

For a single configuration with temperature 7, population 20, 30 generations,
prior strength 0.99, top-6 knowledge and updates after each of the first 20 generations, run:

```bash
python -u TransNASBench101/prior_exp/run_knowledge_100seeds.py
```

This runs all seven tasks in both macro and micro spaces with seeds 0..99
(1,400 searches). Parameters are applied in the process before importing the
experiment code, without editing the shared config. A new timestamped directory
under `prior_exp/results/` contains `settings.json`, `effective_settings.json`,
`run.log`, `status.json`, incremental `summary.csv`/`summary.json`, and the original
per-seed search logs. Use `--dry-run` to inspect settings, or `--run-name NAME`
to resume the corresponding interrupted run. Summary `max_acc_std` is population
SD; `max_acc_std_sample` is sample SD. The existing task metrics are preserved;
room-layout loss is minimized. This is a benchmark lookup search, not retraining.

`run_prior_sweep.py` runs all seven tasks in both macro and micro spaces. Its
default phase uses the configured step/population settings and the knowledge
grid `topk={5,10,15,20}` x `dynamic_steps={10,20,30,40}`. The reduced phase
uses `num_step=30`, `population_num=20`, and the grid `topk={5,6,8,10}` x
`dynamic_steps={10,20,30}`. Structural and shuffled-structural priors use their
defaults in both phases, and every configuration runs seeds `0..99`.

Preview the 448 configurations without running them:

```bash
python TransNASBench101/prior_exp/run_prior_sweep.py --dry-run
```

Run the complete resumable sweep:

```bash
python TransNASBench101/prior_exp/run_prior_sweep.py
```

Per-seed checkpoints stay under `prior_exp/results/`. Completed configuration
summaries are appended to `prior_exp/results/prior_sweep_summary.txt` with only
the varying parameters (including `lumda` for knowledge priors) and `max_acc mean+std`.
New knowledge runs use `_l0.99_` in their directory names and do not reuse lambda-1 search results.
