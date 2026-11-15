# Second, Bounded Calibration Attempt

Declared November 15 before inspecting any new evaluation predictions.
The first attempt did NOT pass its acceptance gate and is preserved unchanged.

## Why Another Attempt

Depth 5 had +4.55 GeoScore on the original 2,000 evaluation sequences, but its
paired interval included zero and Acc@25km declined by 1.10 percentage points.
That is not a demonstrated improvement. The baseline remains deployed.

This attempt uses the SAME paid instance and frozen encoders/galleries.
It adds relative tower weights, rather than inventing the original reranker's
features or changing encoder weights.

## Frozen Selection

- Use ONLY the original 2,000 tuning sequences from experiment-v1.
- Never use the original 2,000 evaluation labels for candidate scoring/selection.
- Fix SO400M weight at 1. Enumerate GOPT and DINOv3 weights in {1, 0.5, 2},
  crossed with depths {10, 5, 20, 64}: exactly 36 configurations.
- Multiply each tower's population-z-scored similarities by its weight before
  the unchanged sum-by-gallery-row fusion. Retain 64 candidates; use the
  unchanged cluster radii, seed, and fourth-power spherical-mean weighting.
- Select maximum tuning mean GeoScore ONLY among candidates whose tuning
  Acc@25km is at least the unchanged baseline's. Baseline is included.
- Exact ties prefer the baseline, then configuration enumeration order.
- Lock weights/depth and their input hashes BEFORE preparing the new cohort.
- If the baseline wins, stop; there is no new candidate to test.

## Fresh Evaluation

- Use pinned OSV-5M archive 01.zip and the same pinned test.csv.
- EXCLUDE every sequence in the original 4,000-query manifest, both splits.
- Also retain the original global hash assignment: evaluation sequences have
  sha256("chipoint-v1:" + sequence) modulo 2 equal to 1.
- Choose one image per remaining sequence by the frozen image hash, then rank
  sequences by the v2 evaluation salt. Use 2,000 sequences if available,
  otherwise 1,000; below 1,000 stop. No outcome-dependent sample size.
- Cache all three towers once for this fresh cohort.
- Compare the ONE locked candidate with the unchanged equal-weight/top-10
  baseline. Do not inspect other candidates on this cohort.
- Use 10,000 paired sequence-bootstrap replicates and the seed in
  fusion_config.json. The original acceptance gate remains unchanged:
  positive GeoScore difference, lower 95% bound > 0, no Acc@25km decline.
- No third attempt or selection based on this second holdout is part of this
  plan. Retain baseline if the gate fails.

## Provenance And Delivery

Keep experiment-v1 immutable. Write separate artifacts/experiment-v2/.
Any v2 demo uses that run directory explicitly; never overwrite v1 results
to make the first attempt appear successful. All reports must distinguish
the two cohorts. A development-subset gain is not a full official benchmark
result and is not encoder fine-tuning.
