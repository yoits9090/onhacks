# One Instance, One Run

The user approved the specific one-instance rental described here.
One cloud pod is provisioning. No improvement has been measured.
The local launcher, export/sync path and presentation demo are implemented.
The evaluation harness passes 12 synthetic tests; launcher/demo checks and
desktop/mobile browser checks pass. None of those checks measure model accuracy
or verify CUDA. The existing public SSH key is registered. Prime accepted one
DataCrunch A100 80GB spot instance with 200GB disk at $1.2478/hour on November
15 at 13:53 EST, within the approved $1.30/hour ceiling. No API key or private
SSH key goes onto the worker.

## Actual Submission Requirements

Sources checked November 15:

- Hacker package pages 7-10:
  https://drive.google.com/file/d/1CN7kNhDQsJM5UZt4y3W7-F_RTLJb0JI9/view
- Official submission requirements: https://onhacks.devpost.com/
- Official rules: https://onhacks.devpost.com/rules

Submit a project/code link and demonstration pictures, and mark the project
Submitted on Devpost BEFORE 17:00 EST. Drafts are not judged. All teammates
must join using their ONHacks application email. Video is optional.
The private pitch is five minutes; judging is 17:30-20:00. The package gives
Creativity, Technical, Pitch, and Impact equal 25% weights. The separate
detailed rubric linked by Devpost could not be accessed (HTTP 401).

This makes a reliable, clearly explained project more valuable than an
unfinished encoder-training run. The proposed scope is a working street-view
geolocation demo with proper upstream attribution and an honest measured
consensus-calibration comparison, not a claim to have created Chipoint.

Delivery milestones: 15:30 fallback decision if the experiment is not on track;
16:00 experiment cutoff; 16:15 local artifacts/demo evidence complete;
16:30 target Submitted status; 16:45 final link/team/status verification.
Use 17:00-17:30 for pitch practice, not unfinished submission uploads.

## Objective And Honest Deliverable

Improve the public Chipoint street-view demo's mean GeoScore on a fresh
sequence-disjoint comparison, using the same frozen three encoders and
published gallery. We are not reproducing the unpublished full reranker or
promising to exceed the model card. The output is a prediction configuration
plus measurements, not a falsely labeled fine-tuned encoder checkpoint.

If the candidate does not pass the acceptance gate, retain the baseline and
report that no reliable gain was demonstrated.

## Before Any Billing

1. Validate the complete runner on local synthetic fixtures, including resume,
   corrupt-cache rejection, sequence separation, selection, and reporting.
2. Keep downloaded DINOv3, matching configs, and projection heads ready.
3. Confirm one SSH key works with Prime; do not copy API keys onto the worker.
4. Refresh the exact offer and wallet. Target one A100 80GB spot, DataCrunch
   Finland, 22 vCPU, 120GB RAM, 200GB disk. Last observed compute rate:
   $1.193/hour; disk 200 * $0.000274/hour; combined about $1.248/hour.
   Reject a changed offer above $1.30/hour without a new decision.
5. Ask once for approval of this specific rental. Also agree on termination
   after verified result export; stopping a Python process does not stop billing.

## Fixed Experiment

`experiment_config.json` freezes the source revisions and search before any
outcomes are seen. `approved_spend_usd: 7.5` records the approved budget target,
not an enforced billing cap.

- Download only OSV-5M test.csv and test archive 00.zip (about 2.25GB images),
  not the whole dataset, plus the three street galleries and encoder assets.
- Join archive membership to metadata by image ID. Select one image per
  sequence using a salted SHA-256 order. Assign entire sequences to tuning
  or final evaluation using the frozen split hash.
- Choose 2,000 queries per side if both sides have enough sequences; otherwise
  1,000, then 500. This choice uses membership only, never model performance.
  Below 500 on either side: stop, do not invent a smaller experiment.
- Cache each tower's top 200 gallery IDs and scores for exactly those queries.
  Each tower searches its own gallery. No encoder gradients or guessed features.
- Test only consensus depths 10 (baseline), 5, 20, 64 on tuning images.
  Equal tower weights, candidate fusion, weighting power, and radius schedule
  remain fixed. No open-ended hyperparameter search or second training run.
- Freeze the tuning winner before reading final-evaluation metrics.
- Report baseline and winner on the same final sequences. Accept only if
  mean GeoScore increases, the paired 95% bootstrap interval excludes zero,
  and accuracy within 25km does not decline. Otherwise ship baseline.

## Single Paid Session

| Phase | Planning allowance | Gate |
|---|---|---|
| Provision, transfer, CUDA/dependency check | 10-20 min | correct GPU, sufficient disk, finite DINO embedding |
| Asset downloads and integrity checks | 10-30 min | source revisions, hashes and gallery row alignment match |
| Cache three towers once | 15-90 min | all selected queries succeed; progress persisted per batch |
| Select and compare cached predictions | 5-15 min | same-query comparison and frozen selection |
| Export and verify locally | 5-10 min | report, config, manifests and caches present locally |
| Terminate approved pod | immediately after export | no active pod/continued compute charges |

These are allowances, not measured throughput promises. At the full three-hour
worker limit, compute plus disk is about $3.75; provisioning/setup time outside
that worker window and provider billing rules also count. Reserve $7.50 for the
entire experiment and leave $2.50 uncommitted. An A100 and an H100 must not run
concurrently. Do not automatically replace an interrupted spot rental.

## Resume, Export, And Stop Conditions

- Per-batch cache writes are atomic; completed batches are reused, not re-encoded.
- Cache identity binds the manifest, config, implementation, dependency versions,
  loaded encoder/projection state, and gallery content. Changed inputs stop resume.
- Copy completed artifacts back to the laptop periodically over SSH. A spot
  interruption can destroy local worker storage; unexported progress can be lost.
- The worker has a maximum 180-minute lifetime and an earlier absolute
  deadline of 16:00 EST on November 15. Leave an hour for export, demonstration
  evidence, and submission before 5pm. run_once.sh wraps setup, downloads and
  the complete experiment in one timeout; run_experiment.py also checks its
  deadline at download/encoding boundaries.
- Stop on failed integrity checks, invalid image/metadata, NaNs, missing tower,
  or impossible ETA. No silent exclusions, replacement encoders, or partial
  result presented as a complete experiment.
- After caching, all selection/reporting can run locally. Do not keep a GPU
  rented just to change charts or wording.
- A worker timeout does NOT terminate the provider pod or enforce a billing cap.
  Confirm termination separately after export, using the approved lifecycle.

## Artifacts

`manifest.json`, input/cache provenance, immutable per-tower retrieval batches,
`selected.json`, `report.json`, final paired predictions, and
`prediction_config.json`. Preserve failure logs even when no improvement wins.

SSH registration and the specific rental are approved and have been performed.
The first paid gates are actual CUDA visibility, asset verification, and
finite real-tower retrieval. Stop on failures rather than silently changing the
experiment. A local smoke test is not evidence those GPU gates have passed.
