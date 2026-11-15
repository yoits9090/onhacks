# Chipoint Improvement Protocol

Status: preparation only. No measured baseline or improvement yet.

## Scope

- Street-view arm only, with frozen encoders and frozen projection heads.
- Use every explicitly selected tower, each against its own matching gallery.
- Do not silently drop a failed tower or substitute a different encoder.
- Do not use the guessed 23-feature stack or the pretrained reranker.
- Do not claim reproduction of the model card's full-pipeline benchmark.
- No training, paid instance creation, or benchmark downloads during local prep.

## Reference And Inputs

- Model: chiikabu-labs/chipoint-2 at
  `a150120cd968d48310d092caf26854f109c90b51`.
- Demo: chiikabu-labs/chipoint-v2-geolocation at
  `65dd8df82f36608ad819f5ecd4371da16f73704e`.
- SO400M: OpenCLIP `ViT-SO400M-16-SigLIP2-256`, pretrained `webli`.
- GOPT: OpenCLIP `ViT-gopt-16-SigLIP2-256`, pretrained `webli`.
- DINOv3: `facebook/dinov3-vitl16-pretrain-lvd1689m`, mirrored on
  ModelScope at `bab39f7f4023c45534a5326e9378edc5b2016e48`.
- DINOv3 weights SHA-256:
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- Every result must record selected towers, exact input file hashes, source
  revisions, dependency versions, device, numerical precision, and image IDs.

## Baseline

Reproduce the demo's explicit Street-view mode, not its Auto/photo router:

1. Apply each tower's demo preprocessing; normalize its raw embedding.
2. Apply its matching projection head; normalize the resulting 512-D vector.
3. Retrieve top 200 by cosine similarity from that tower's own street gallery.
4. Z-score these 200 similarities within each tower, using population standard
   deviation plus 1e-9; sum contributions by gallery row across towers.
5. Keep the top 64 fused candidates, with equal tower weights.
6. Use the first 10 candidates, matching the demo's default display parameter.
7. Seed a geographic cluster at candidate 1. Try radii 250, 500, then 1000 km;
   stop when at least 3 candidates are present, otherwise use the last radius.
8. Predict the spherical mean of cluster coordinates with weights
   `max(fused_score, 0)^4`; use uniform weights if their sum is zero.

`chipoint/baseline.py` defines steps 4-8. Never average different towers'
embedding vectors or search one tower's queries against another tower's index.

## Data Separation

The first experiment will use labeled OSV-5M test images as a development
corpus, not claim an untouched official benchmark result. Retain image IDs,
sequence IDs, and original split in a manifest. Before inspecting predictions,
assign whole sequences to tuning or final evaluation with
`int(sha256(("chipoint-v1:" + sequence_id).encode()).hexdigest(), 16) % 2`:
0 is tuning, 1 is final evaluation. Fail if sequence IDs are missing.

The gallery must remain the published OSV-5M training gallery. Never add query
images to it. Never train or select configurations using final-evaluation
labels. Do not sample the first N zip entries as a representative dataset:
define a deterministic, sequence-aware sample manifest before encoding.
For this single run, sample one image per sequence using salted SHA-256
ordering. Archive membership is joined to metadata by image ID. Use 2,000
queries per side if available, otherwise 1,000, then 500. Select the largest
tier supported on BOTH sides before encoding, not based on observed accuracy.
The salts and tiers are frozen in experiment_config.json. Below the minimum
tier, stop rather than silently changing the experiment.

## First Improvement Experiment

Cache immutable per-query, per-tower top-200 IDs and cosine scores once.
Both baseline and tuning must read the same cache, not re-encode queries.
The initial search changes consensus depth only: top_n in {5, 10, 20, 64}.
Keep tower fusion, radius schedule, and weighting power fixed.

Choose maximum mean GeoScore on the tuning sequences, with ties preferring
top_n=10 and then the smallest depth. Preserve the unchanged baseline as an
option. Evaluate that one selected configuration and the baseline once on
final-evaluation sequences.

## Reporting

- Great-circle distance: haversine with Earth radius 6371.0088 km.
- Primary: mean `5000 * exp(-distance_km / 1492.7)` GeoScore.
- Secondary: percent within 1, 25, 200, 750, and 2500 km; median and mean km.
- Report identical query counts, failed images, and sequence counts.
- Failed encodes invalidate a run unless exclusions are fixed identically
  before comparing configurations. Never silently score only successes.
- A small positive difference alone does not establish a reliable gain.
  Report paired per-sequence differences and uncertainty before making that
  claim. This run uses 10,000 paired sequence-bootstrap replicates with the
  frozen seed in experiment_config.json and a percentile 95% interval.
  There is one query per sequence, so sequence and query resampling coincide.
- Accept the tuning-selected configuration only if final mean GeoScore
  improves, the lower bootstrap bound is above zero, and Acc@25km does not
  decline. Otherwise retain the baseline. Never select another candidate
  after seeing final-evaluation outcomes.

The prep smoke test checks shapes, finite outputs and retrieval plumbing.
It does not establish semantic accuracy or an improvement.
