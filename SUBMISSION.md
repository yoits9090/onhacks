# Submission And Five-Minute Pitch

This is a draft/checklist, not an already submitted project.
Deadline: November 15, 2026, 17:00 EST. Target Submitted status at 16:30.

## Project Description Draft

**Working title:** Chipoint Calibration Lab

A street-photo geolocation demo and reproducible experiment for testing whether
lightweight prediction calibration improves a frozen open-weight retrieval
pipeline. It compares a pinned public-demo baseline with tuning-selected fusion
weights and consensus settings on sequence-disjoint images. It exports an offline recorded
demo so the presentation does not depend on an interruptible GPU instance.

We build on Chiikabu Labs' Chipoint v2, rather than claiming to have trained
its encoders or reproduced its unpublished full reranker. Our contribution is
the bounded evaluation/calibration workflow, verified retrieval integration,
resume/export tooling, and demonstration interface.

## Verified Results - Ready To Use

"On 2,000 fresh OSV-5M sequences, our frozen-tower calibration improves
GeoScore from 3963.36 to 4082.39 against the public street-view demo baseline.
The paired difference is +119.02, with a 95% sequence-bootstrap interval
of [+77.14, +161.11]. Accuracy within 25km increases from 32.80% to 33.50%,
and mean error falls from 1119.87km to 890.43km."

The selected parameters are SO400M:GOPT:DINOv3 weights 1:2:0.5 and consensus
depth 5. Selection used only the original tuning cohort; all 4,000 sequences
from the first attempt were excluded from this comparison.

Not all metrics improve: accuracy within 1km declines from 0.50% to 0.35%.
The earlier depth-only attempt was rejected. A later 48-setting refinement
was compared against the accepted checkpoint on another 4,000 new sequences:
its +6.24 GeoScore interval crosses zero, so it was also rejected. Do not add
gains across these different cohorts or call the third attempt an improvement.

Frozen evidence: `results/attempt-2/report.json` and
`checkpoints/fusion-v2/lead-verification.json`; later rejection:
`results/attempt-3/report.json`. Tag: `checkpoint-fusion-v2`.

These reports cover OSV-5M development subsets, with one image per sequence.
It is not the official full benchmark and is not a general-photo evaluation.
The output is a calibration configuration, not newly fine-tuned encoder weights.

## What We Learned

Most of the hard engineering was making a comparison trustworthy: matching
each encoder to its gallery, verifying assets, caching the GPU work, keeping
whole driving sequences apart, locking the candidate, and exporting real
examples without selecting only successes. Keeping rejected attempts and the
accepted checkpoint prevents a later noisy result from silently replacing it.

## Demonstration Assets

- `PITCH.md`: timed five-minute script and limitations.
- `docs/demo-recorded.webp`: compact real screenshot for the project README.
- `artifacts/screenshots/demo-desktop.png`: full-resolution demonstration
  picture for manual Devpost upload; mobile/hero captures are beside it.
- Local full recorded fallback: http://127.0.0.1:8000.
- Local live CUDA upload demo, while the SSH tunnel/GPU remains on:
  http://127.0.0.1:8001. Initial three-model load is slow; the demo is warmed.
- Localhost URLs are for this laptop, NOT remotely accessible submission links.
- The original repository is private. Arrange judge access or approve a
  separate sanitized public project snapshot; do not assume a private GitHub
  link works for judges. Nothing has been published publicly or submitted.

## Pitch Timing

Use [PITCH.md](PITCH.md) as the canonical timed script: problem, system,
reproducibility, rejected first attempt, accepted result, demo, then limits.
Rehearse with a five-minute timer. If short on time, omit optional follow-up
details rather than mislabeling recorded output as live.

## Demonstration Fallback

- Default recorded mode must be visibly labeled, with no simulated live upload.
- The three recorded examples are the first three evaluation queries in the
  deterministic manifest, not the three smallest localization errors.
- Keep aggregate numbers next to the example; one attractive result is not
  evidence of overall accuracy.
- A short screen recording is optional but useful if venue Wi-Fi fails.
- Rehearse locally after the GPU is gone. Check image, coordinates, comparison
  table and fallback map/coordinate display still work.

## Devpost Checklist

- Join https://onhacks.devpost.com/ and create the project draft early.
- Add all teammates using the email on their ONHacks application.
- Provide the repository/project link and demonstration pictures.
- Credit Chipoint v2 / Chiikabu Labs, the three encoder sources, and OSV-5M.
- State exactly what we built during the event and what is reused.
- Follow each dependency/model/dataset's licence; preserve the DINOv3 licence.
- Optional: add a short video. It is not worth missing the submission deadline.
- Before 16:30, mark **Submitted**, not Draft.
- Before 16:45, open every submitted link in a logged-out/private window and
  verify the project status, teammates, images, and repository accessibility.
- Find the assigned judging room/time on the participant pass/dashboard.

Do not publish API keys, .env files, private SSH keys, downloaded model blobs,
or any unrelated capture files. Do not create a public repository or submit
on the user's behalf without approval.

## Sources

- Hacker package: https://drive.google.com/file/d/1CN7kNhDQsJM5UZt4y3W7-F_RTLJb0JI9/view
- Requirements: https://onhacks.devpost.com/
- Rules: https://onhacks.devpost.com/rules
- Upstream model: https://huggingface.co/chiikabu-labs/chipoint-2
- Reference demo: https://huggingface.co/spaces/chiikabu-labs/chipoint-v2-geolocation
