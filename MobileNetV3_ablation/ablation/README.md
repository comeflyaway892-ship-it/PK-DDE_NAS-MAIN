# MobileNetV3 Ablation

This ablation package intentionally keeps data and checkpoint files in their
original locations. Paths in `../all_path.py` point to `../MobileNetV3/data`,
`../MobileNetV3/checkpoints`, and `../MobileNetV3/configs`.

Example runs:

```bash
python MobileNetV3_ablation/ablation/ablation_core_modules.py --dataset cifar10 --seed_start 0 --seed_end 0
python MobileNetV3_ablation/ablation/ablation_evo_algorithms.py --dataset cifar10 --seed_start 0 --seed_end 0
```

Core-module variants:

- `fitness_as_one`: replace fitness weights with 1/uniform weights.
- `diffusion_consistency_as_one`: replace diffusion-consistency weights with 1.
- `no_posterior_recovery`: skip posterior recovery and evaluate the sampled x0 directly.
- `global_mutation`: use global random mutation with rate 0.05 and no discrete diffusion.
