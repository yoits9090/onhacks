# Accepted Fusion V2 Checkpoint

Frozen before any further experiments. Do not overwrite this checkpoint.
Controller source: commit `be418d4`; protocol: `FUSION_EXPERIMENT.md`.

## Parameters

- SO400M weight: **1.0**
- GOPT weight: **2.0**
- DINOv3 weight: **0.5**
- Geographic consensus depth: **5**
- Per-tower retrieval: 200; fused candidate pool: 64.
- All three original encoders, projection heads and galleries remain frozen.

`prediction_config.json` is the deployable calibration checkpoint. This is
not a newly trained neural-weight file. The original multi-GB model assets
are not copied into Git; their numeric identities and actual runtime
versions are preserved in the three `tower-identity-*.json` files and
`environment.txt`. The pinned Chipoint model revision is in the config.

## Verified Result

On 2,000 fresh OSV-5M archive-01 evaluation sequences, excluding all 4,000
sequences in the first experiment:

| Metric | Equal-weight / top-10 baseline | This Checkpoint |
|---|---:|---:|
| GeoScore | 3963.36 | 4082.39 |
| Accuracy within 25km | 32.80% | 33.50% |
| Mean error | 1119.87km | 890.43km |

Paired GeoScore difference: **+119.02**, 95% sequence-bootstrap interval
**[+77.14, +161.11]**. The checkpoint passes the predefined acceptance gate.
The 97.5% two-attempt Bonferroni sensitivity interval also stays above zero;
see `lead-verification.json`.

This is a development-subset comparison against the public street-view
demo baseline, not the published full pipeline or an official benchmark.
Not all metrics improve: accuracy within 1km declines from 0.50% to 0.35%.
Read `report.json` for all metrics, without rounding.

## Integrity

`SHA256SUMS` covers the 11 frozen parameter, report, selection, provenance
and environment files. The selection uses only the original tuning cohort;
the fresh evaluation is not used to choose this checkpoint.

Future attempts must use separate directories and new untouched evaluation
sequences. This checkpoint remains the fallback unless a replacement passes
its own preregistered comparison against it.
