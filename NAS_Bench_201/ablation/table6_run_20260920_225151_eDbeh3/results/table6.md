| Variant | Dynamic prior | Fitness guidance | Diffusion consistency | Posterior sampling | Performance (%) | Validity (%) |
|---|---|---|---|---|---|---|
| Full PK-DDE-NAS | Yes | Yes | Yes | Yes | 47.08 ± 0.31 | 100.00 ± 0.00 |
| w/o dynamic prior | No | Yes | Yes | Yes | 46.87 ± 0.34 | 100.00 ± 0.00 |
| w/o fitness guidance | Yes | No | Yes | Yes | 47.01 ± 0.32 | 100.00 ± 0.00 |
| w/o diffusion consistency | Yes | Yes | No | Yes | 47.07 ± 0.30 | 100.00 ± 0.00 |
| w/o posterior sampling | Yes | Yes | Yes | No | 45.06 ± 0.82 | 100.00 ± 0.00 |
| Random mutation | No | No | No | No | 46.61 ± 0.38 | 100.00 ± 0.00 |

Performance: highest test accuracy among evaluated architectures (x-test search and selection); sample SD across search seeds.
Validity: API-queryable candidates / all evaluated candidates, including duplicates.
See settings.json for ablation definitions and per-run evaluation budget.
