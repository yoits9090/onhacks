# Five-Minute Pitch: Chipoint Calibration Lab

Use the saved **v2 checkpoint** for every numerical claim here. A later
experiment is not a replacement until its fresh comparison is verified.
Do not add gains measured on different cohorts.

## 0:00-0:30 - The Problem

"A street photograph can reveal geographic information even without a GPS
tag. But an impressive location guess on three hand-picked photos does not
tell us whether a model really improved.

We built a geolocation demo with an auditable answer to that second question:
can a small, inexpensive calibration make an existing open-weight pipeline
better on images it did not tune on?"

## 0:30-1:15 - The System

"We build on Chiikabu Labs' public Chipoint v2 street-view demo. Three frozen
vision towers encode the image. Each searches its own matching reference
gallery, and we combine their candidates with geographic consensus.

We do not claim to have trained these encoders or reproduced the unpublished
full reranker. Our contribution is the calibration, evaluation and
presentation layer around that public baseline."

Show the pipeline diagram in README.md. Explain separate embedding spaces,
top-200 retrieval per tower, fusion to 64 candidates, then spherical consensus.

## 1:15-2:00 - What Makes The Comparison Credible

"The expensive retrieval runs once and is cached. Model and data assets have
pinned sources and integrity checks; failed towers never disappear silently.

We split by whole driving sequences, not adjacent random frames. We lock the
calibration before the final comparison and preserve the baseline when an
experiment fails. The results, input identities and checksums are saved."

Mention one A100 instance at $1.2478/hour. Do not call that the total cost.

## 2:00-2:45 - The Failure We Did Not Hide

"Our first depth-only attempt did not pass. Its GeoScore gain was small,
the interval included zero, and 25km accuracy declined. We kept the baseline.

The next bounded search adjusted the towers' relative weights using only the
original tuning sequences. Then we tested its locked choice on 2,000 new
sequences, excluding all 4,000 from the first experiment."

The accepted calibration is SO400M:GOPT:DINOv3 = **1:2:0.5**, consensus depth 5.

## 2:45-3:30 - The Result

| Fresh 2,000-Sequence Comparison | Original Demo Baseline | Saved V2 |
|---|---:|---:|
| GeoScore | 3963.36 | 4082.39 |
| Accuracy within 25km | 32.80% | 33.50% |
| Mean error | 1119.87km | 890.43km |

"That is +119.02 GeoScore, with a paired 95% interval from +77.14 to +161.11.
Fourteen more images out of 2,000 land within 25km. This passes our
predefined deployment gate."

These are development-subset results, not an official benchmark or a gain
over the published full Chipoint pipeline. Not every metric improves:
accuracy within 1km changes from 0.50% to 0.35%.

## 3:30-4:20 - Demo

1. Open the local recorded demo and identify the **Recorded** badge.
2. Show the real report and deployed weights.
3. Select an exported image: orange is known truth, olive is baseline,
   teal is deployed prediction. The three examples are the first three
   evaluation entries, NOT selected successes.
4. Say: "A mean improvement does not guarantee a better guess on every image.
   Some of these predetermined examples get worse."
5. If the live CUDA server is available, upload one JPEG/PNG/WebP. Identify
   **ground truth unknown**; do not score it by eye or invent confidence.
6. If spot hardware/network fails, stay with the clearly labeled recorded
   results. Never present a recorded response as live inference.

## 4:20-5:00 - Impact And Limits

"This can support public street-image indexing and awareness of geographic
privacy. It is not proof of a person's location or a GPS replacement.

Uploads are bounded and processed in memory, not saved. Map tiles and fonts
use third-party services. Our next research step would be broader spatial
holdouts and further validation, rather than claiming a universal gain from
one development subset.

The main lesson is that a small change can help, but measuring it honestly
is part of the engineering."

## Before Judging

- Keep recorded mode available independently of the GPU.
- Check the private repository's judge access; do not assume a private URL
  works for logged-out judges.
- Include demonstration pictures and the code/project link on Devpost.
- Every teammate joins with their ONHacks application email.
- Mark the project **Submitted before 17:00 EST**; a draft is not judged.
- Use the same cohort's before/after values in any slide or submission text.
- Credit Chiikabu Labs, OSV-5M, OpenCLIP/SigLIP2 and Meta DINOv3.

## If Asked About Further Tuning

We also tried a preregistered 48-setting refinement against the saved v2
checkpoint, using another 4,000 untouched sequences. Its +6.24 GeoScore
difference has a 95% interval of [-4.59, +17.19]; the stricter three-attempt
adjusted interval also crosses zero. We keep v2 rather than treating that
extra positive point estimate as a reliable improvement.
