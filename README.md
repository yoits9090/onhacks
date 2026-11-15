# Chipoint Calibration Lab

A reproducible street-photo geolocation experiment and presentation demo,
built on [Chiikabu Labs' Chipoint v2](https://huggingface.co/chiikabu-labs/chipoint-2).

**Status:** the single A100 experiment is starting. No measured accuracy gain
is claimed yet. Synthetic test results are software checks, not model results.

## What We Built

- Integrated the public demo's three frozen street-view towers with their
  matching projection heads and reference galleries.
- Added pinned asset retrieval, integrity checks, sequence-disjoint sampling,
  immutable retrieval caches, and a bounded calibration comparison.
- Added a one-command worker, interruption recovery, result export/sync,
  and a presentation UI with explicitly labeled recorded/live modes.
- Kept the baseline whenever the selected candidate fails the acceptance gate.

We did **not** train Chipoint's encoders or reproduce its unpublished full
reranker. The experiment changes geographic-consensus depth, not encoder
weights. The published full-pipeline model-card score is not our baseline.

## Pipeline

```text
street image
  -> SO400M SigLIP2 / GOPT SigLIP2 / DINOv3
  -> each tower's own 512-D projection and matching street gallery
  -> cached top-200 candidates per tower
  -> fixed z-score fusion, retaining 64 candidates
  -> baseline geographic consensus or tuning-selected consensus depth
  -> latitude / longitude
```

Different towers' embedding spaces are never mixed directly.

## Experiment

The source revisions and search are frozen in `experiment_config.json`.
One image per sequence is selected from one pinned OSV-5M test archive.
Whole sequences are assigned to tuning or final evaluation before encoding.
The search compares depths 10 (unchanged baseline), 5, 20 and 64.

Selection uses tuning GeoScore only. The selected configuration is locked
before the final comparison. Acceptance requires a positive mean GeoScore
difference, a paired 95% bootstrap interval above zero, and no decline in
accuracy within 25km. Otherwise the deployed configuration remains baseline.

This is a development-subset comparison, not an untouched official benchmark.
See [EVALUATION.md](EVALUATION.md) for metrics, separation, and limitations.

## Local Demo

With Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-demo.txt
.venv/bin/python scripts/serve_demo.py
```

Open http://127.0.0.1:8000. Recorded mode reads exported artifacts; it shows a
waiting state until real results exist, never fabricated examples or scores.
Map tiles/fonts use third-party services; coordinates remain available as
a fallback. `--live` enables real uploads only on a prepared CUDA host.

## GPU Run

Read [GPU_PREP.md](GPU_PREP.md) and [LAUNCH_PLAN.md](LAUNCH_PLAN.md) first.
These scripts do not rent or terminate instances and cannot cap cloud billing.

```bash
bash scripts/run_once.sh              # safe dry-run, no downloads/installs
bash scripts/run_once.sh --execute    # explicitly run on the prepared GPU host
```

The actual run downloads only the selected model/gallery assets plus OSV-5M
test.csv and archive 00.zip, not the full training corpus. It writes:

```text
artifacts/experiment-v1/
  manifest.json             fixed image/sequence/split membership
  cache/                    per-tower, per-batch retrieval results
  selected.json             tuning winner, locked before final evaluation
  report.json               paired final metrics and acceptance decision
  prediction_config.json    deployed baseline or accepted calibration
  predictions.csv           paired final-query predictions/errors
  demo_examples/            first three evaluation images, not best cases
```

Model blobs, credentials, datasets, and generated artifacts are excluded from
Git. Follow [SUBMISSION.md](SUBMISSION.md) for the five-minute pitch and Devpost
checklist. The repository remains private; judges need an explicitly approved
accessible code/project link before submission.

## Attribution And Scope

- Chipoint v2 and the reference street-view demo: **Chiikabu Labs**.
- Frozen vision backbones: SigLIP2 via OpenCLIP and Meta DINOv3.
- Evaluation imagery: **OSV-5M**. Image IDs and original split are retained.
- Pinned demo source and provenance: `upstream/DEMO_SOURCE.txt`.
- DINOv3's licence is preserved in its encoder directory; it is not an
  MIT-licensed model. Respect all upstream model, data, and software terms.

This is street-view geolocation, not a guarantee about arbitrary photographs.
A model guess is not proof of a person's location.
