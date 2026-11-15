# Submission And Five-Minute Pitch

This is a draft/checklist, not an already submitted project.
Deadline: November 15, 2026, 17:00 EST. Target Submitted status at 16:30.

## Project Description Draft

**Working title:** Chipoint Calibration Lab

A street-photo geolocation demo and reproducible experiment for testing whether
lightweight prediction calibration improves a frozen open-weight retrieval
pipeline. It compares a pinned public-demo baseline with a tuning-selected
consensus setting on sequence-disjoint images. It exports an offline recorded
demo so the presentation does not depend on an interruptible GPU instance.

We build on Chiikabu Labs' Chipoint v2, rather than claiming to have trained
its encoders or reproduced its unpublished full reranker. Our contribution is
the bounded evaluation/calibration workflow, verified retrieval integration,
resume/export tooling, and demonstration interface.

## Results: Fill From Frozen Artifacts Only

Read `artifacts/experiment-v1/report.json` for measured values and the decision.
Read `prediction_config.json` for the deployed setting. Do not substitute test
fixture values, published model-card numbers, or the best individual example.

Until that report exists, say: **"The evaluation has not completed yet."**
If `accept_candidate` is false, say: **"Our test did not establish a reliable
improvement, so the demo retains the baseline."** Do not imply that a
non-significant positive difference proves an improvement.

The report is for one OSV-5M development subset, with one image per sequence.
It is not the official full benchmark and is not a general-photo evaluation.
The output is a calibration configuration, not newly fine-tuned encoder weights.

## Pitch Timing

- 0:00-0:35: problem and intended use. Locating an unfamiliar street photo is
  difficult; a point on a map needs an honest evaluation, not just a confident UI.
- 0:35-1:45: show one example. Identify recorded versus live mode, display the
  image, prediction, known location, and baseline/candidate comparison.
- 1:45-2:45: explain the pipeline: frozen encoders, matching image galleries,
  candidate fusion, and geographic consensus. Explain our exact contribution.
- 2:45-3:45: show the frozen aggregate comparison, sequence-disjoint split, and
  uncertainty interval. State the actual pass/fail decision plainly.
- 3:45-4:30: impact and limitations. This is street-view scoped; location errors
  remain possible. Do not use model guesses as proof of a person's location.
- 4:30-5:00: what we learned, a concrete next step, and upstream acknowledgments.

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
