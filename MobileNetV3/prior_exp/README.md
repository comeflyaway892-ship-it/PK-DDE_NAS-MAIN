# MobileNetV3 Prior Experiments

This folder keeps prior-based diffusion experiments separate from the original
MobileNetV3 pipeline.

## Structural Prior

The structural prior uses the MobileNetV3/OFA token semantics only:

- depth tokens: `0,1,2 -> depth 2,3,4`
- operation tokens: `0..8 -> (kernel size, expansion ratio)`

For operation tokens, the prior kernel is:

```text
d(a,b) = kernel_weight * |k_a - k_b| + expand_weight * |e_a - e_b|
M_d(a,b) = softmax_b(-d(a,b) / op_tau)
Q_t,d = (1 - beta_t) I + beta_t M_d
```

Depth tokens use the same distance-softmax idea over depth values. Invalid
depth-token outputs are masked out, so the first five positions still only
produce tokens `0,1,2`.

## Run

From the repository root:

```bash
python MobileNetV3/prior_exp/main_prior.py --dataset cifar10 --gpu 0
```

For a search-only smoke run without MobileNet retraining:

```bash
python MobileNetV3/prior_exp/main_prior.py --dataset cifar10 --gpu 0 --skip-retrain
```

To run only three search seeds while keeping MobileNet retraining enabled:

```bash
python MobileNetV3/prior_exp/main_prior.py --dataset aircraft --gpu 0 --runs 3
```

Run three search seeds (`0,1,2`) serially on all four datasets while retaining
all other `main_prior.py` defaults:

```bash
python MobileNetV3/prior_exp/run_all_datasets.py --gpu 0
```

When CUDA visibility is selected by the shell, `--gpu` is the logical index
inside that visible list. For example, this runs on physical GPU 1 through its
logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 python MobileNetV3/prior_exp/run_all_datasets.py --gpu 0
```

Additional prior arguments are forwarded unchanged. To run the knowledge prior
on all four datasets, retaining the top 20 candidates while dynamically
updating the prior for the first 30 generations, use:

```bash
python MobileNetV3/prior_exp/run_all_datasets.py \
  --gpu 0 \
  --prior-type prior_Konwleage \
  --knowledge-topk 20 \
  --knowledge-dynamic-steps 30 \
  --lumda 0.99
```

This configuration is saved under
`prior_exp/results/prior_Konwleage_top20_l0.99_dyn30/<dataset>/`, separately from
earlier `top10`/`dyn10` runs.

To wait for physical GPU 0 and keep at least 90% of its memory reserved as a
reusable PyTorch cache for this complete four-dataset run:

```bash
python gpu_memory_launcher.py \
  --gpu 0 \
  --start-below 50 \
  --reserve-at-least 90 \
  --interval 10 \
  -- MobileNetV3/prior_exp/run_all_datasets.py \
     --gpu 0 \
     --prior-type prior_Konwleage \
     --knowledge-topk 20 \
     --knowledge-dynamic-steps 30 \
     --lumda 0.99
```

The launcher's `--gpu` selects a physical device. Because the launcher then
exposes only that device, the target's `--gpu 0` selects its logical `cuda:0`.
When this batch runner detects the launcher, it executes all dataset jobs in the
shared process so they can reuse the reservation.

The order is `cifar10`, `cifar100`, `aircraft`, `pets`. A subset can be run or
resumed with `--datasets`, such as `--datasets aircraft pets`.

Useful ablation parameters:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset cifar10 \
  --gpu 0 \
  --prior-name structural_tau05 \
  --op-tau 0.5 \
  --kernel-weight 1.0 \
  --expand-weight 1.0
```

Results are saved under:

```text
MobileNetV3/prior_exp/results/<prior-name>/<dataset>/
```

Each searched architecture is retrained with the configured seeds
`777,888,999`. Completed final checkpoints are shared across search seeds for
the same dataset and architecture; compatible runs are reused and only missing
retraining seeds are executed. Search-specific `arch_N/seed-XXXX.pth` files are
hard links to the shared checkpoint, and `retrain_runs.jsonl` records every
accuracy and checkpoint path. Each checkpoint contains the final model,
optimizer and scheduler states, training configuration, and final/best
validation accuracies.

## Shuffled Structural Prior

The shuffled prior is a negative control for the structural prior. It first
builds `M^Struct`, then randomly permutes the operation-token rows and columns:

```text
M^Shuffle = P M^Struct P^T
```

This keeps the probability shape and concentration of the structural kernel,
but breaks the semantic relationship between operation tokens.

Run it with the same structural hyperparameters:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset cifar10 \
  --gpu 0 \
  --prior-type shuffled_structural \
  --shuffle-seed 0 \
  --op-tau 0.5 \
  --runs 3
```

By default, only operation-token semantics are shuffled. Depth-token semantics
stay structural so the control focuses on the MBv3 operation prior. To shuffle
depth tokens too:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset cifar10 \
  --gpu 0 \
  --prior-type shuffled_structural \
  --shuffle-seed 0 \
  --shuffle-depth
```

## Prior Knowledge From First Generation

`prior_Konwleage` keeps the first generation uniform, evaluates it, then uses
the top-k initial candidates to build a runtime preset prior:

```text
pi_depth = frequency of tokens 0,1,2 in the first 5 positions of top-k candidates
pi_op = frequency of tokens 0..8 in the last 20 positions of top-k candidates
pi = lumda * pi_topk + (1 - lumda) * pi_uniform
M_d = 1 pi^T
```

`lumda=1` uses the top-k frequency prior completely. `lumda=0` recovers the
uniform corruption prior.

The default is `lumda=0.99`: 99% top-k frequency prior and 1% uniform.

Run with the default top-k 10 and prior strength 0.99:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset aircraft \
  --gpu 0 \
  --prior-type prior_Konwleage \
  --runs 3
```

Set the strength explicitly:

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset aircraft \
  --gpu 0 \
  --prior-type prior_Konwleage \
  --lumda 0.99 \
  --runs 3
```

To update the prior dynamically, add top-k generated children from the first N
evolution steps. Only the generated child population `x_next` is counted; the
selected/merged next generation is not counted.

```bash
python MobileNetV3/prior_exp/main_prior.py \
  --dataset aircraft \
  --gpu 0 \
  --prior-type prior_Konwleage \
  --knowledge-dynamic-steps 5 \
  --runs 3
```

## Marginal Preset Prior

`structural_prior.py` also includes `build_marginal_prior_kernel`, which builds
the rank-one preset prior:

```text
M_d = 1 pi_d^T
Q_t,d = (1 - beta_t) I + beta_t 1 pi_d^T
```

That path is available for later experiments, but the current CLI runs the
structural prior by default.
