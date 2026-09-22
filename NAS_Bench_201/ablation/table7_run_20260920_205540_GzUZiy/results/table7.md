| Kernel | Applicable token semantics | Mean performance (%) | Standard deviation | Validity (%) |
|---|---|---:|---:|---:|
| Uniform | Nominal or ordinal | 46.87 | 0.34 | 100.00 |
| Absorbing / mask | Mask-aware encoding | 44.29 | 1.08 | 100.00 |
| Discrete Gaussian | Ordered decisions | 46.98 | 0.34 | 100.00 |
| Distance-aware | Ordered decisions | 46.95 | 0.35 | 100.00 |
| Fixed marginal | Prespecified frequencies | 46.95 | 0.32 | 100.00 |
| Dynamic marginal | Frequencies learned from high-fitness architectures | 47.08 | 0.30 | 100.00 |

Proposal mode: reverse_only. SD is sample SD across search seeds (ddof=1).
Performance uses x-test for both search and reporting. Validity is API-queryable evaluated candidates / all evaluated rows (duplicates included).
NB201 operations are nominal; Gaussian/distance kernels use the declared index order as a control, not a natural ordinal semantics.
Absorbing/mask has no forward corruption in this protocol: unmasked tokens cannot change under its reverse transition; it is a degenerate initial-population control.
