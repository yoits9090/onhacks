# GPU Handoff — Chipoint v2 Street Arm (prep only)

Status: **preparation complete. NO baseline has been measured and no
accuracy is claimed.** This repo is ready to rsync to a GPU instance;
nothing here rents, terminates, or bills anything by itself.

## Verified locally — actually RUN (see artifacts/prep-verification.log)

- `chipoint.towers.Tower('dinov3', device='cpu')` — loads the real local
  DINOv3 checkpoint (415 tensors), demo preprocessing, fp32 unit-norm
  output (max |norm-1| ≈ 6e-8), non-finite raises.
- `chipoint.gallery.Gallery.topk` — chunked top-k on CPU equals dense
  reference (tests), returns fp32 sims / int64 ids sorted desc.
- `chipoint.pipeline.StreetArm` — CPU end-to-end retrieve+predict on
  deterministic SYNTHETIC galleries with the real dinov3 tower + real
  proj_head_dinov3.pt (labeled plumbing-only, no semantic meaning).
- `chipoint.nn.ProjHead` — loads all three local proj_head_*.pt
  (checkpoint dims verified: 1152/1536/1024 → 512); forward normalizes
  in fp32 even when the MLP runs fp16.
- `scripts/fetch_assets.py` — `--dry-run` plan correct; real
  `--towers dinov3` run verified the existing proj head + 4 DINO small
  files against authoritative hashes and skipped download; 8/8 mocked
  downloader tests pass (hard-fail metadata, missing-hash rejection,
  preexisting-file verify-before-hub).
- `scripts/setup_instance.sh`, `scripts/fetch_data.sh` — `bash -n` clean.
- OpenCLIP **registry/config only** (no weights, no model instantiation):
  `open_clip_torch==4.2.2` `list_pretrained()` contains
  `('ViT-SO400M-16-SigLIP2-256','webli')` and
  `('ViT-gopt-16-SigLIP2-256','webli')`; `get_model_config` embed_dim =
  1152 / 1536 matching the checkpoints' tower dims.

## Implemented but NOT yet run (requires the GPU instance)

- CUDA paths everywhere: bf16 DINO / fp16 OpenCLIP encoders, fp16 gallery
  matmul, fp16 proj-head MLP (CPU fp16-head smoke may be attempted but is
  not equivalent evidence).
- OpenCLIP towers end-to-end (`so400m256`, `gopt`) — weights are fetched
  lazily by `open_clip` at first `Tower(...)` construction.
- `--real-gallery` retrieval against the downloaded ~5 GB-per-tower index.
- `scripts/setup_instance.sh` execution (Linux+NVIDIA only; it hard-fails
  elsewhere by design).
- Non-dry `fetch_assets.py --galleries` / `--dinov3-weights` bulk downloads.

## Pinned sources (authoritative — do not substitute)

| Asset | Source | Revision |
|---|---|---|
| chipoint-2 weights/galleries | HF `chiikabu-labs/chipoint-2` | `a150120cd968d48310d092caf26854f109c90b51` |
| DINOv3 encoder | ModelScope `facebook/dinov3-vitl16-pretrain-lvd1689m` (ungated mirror) | `bab39f7f4023c45534a5326e9378edc5b2016e48` |
| Demo reference (read-only) | HF Space `chiikabu-labs/chipoint-v2-geolocation` | `65dd8df82f36608ad819f5ecd4371da16f73704e` → `upstream/demo_app.py` |

DINOv3 `model.safetensors`: 1,212,559,808 bytes,
sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`
(per-file hashes in `encoders/dinov3-vitl16-pretrain-lvd1689m/provenance.json`).

Local `weights/proj_head_{dinov3,so400m256,gopt}.pt` verified byte-identical
(LFS sha256) to the pinned HF revision. `fetch_assets.py` re-verifies
whatever is present, so transferring them is optional but saves ~360 MB.

## Transfer to the instance

Apple's stock `rsync` has no `--info=progress2`; use `-av --progress`.
Weights are already dense — do NOT use `-z`.

```bash
rsync -av --progress \
  --exclude '.venv/' --exclude '.git/' --exclude 'data/' \
  --exclude 'artifacts/' --exclude 'capture/' --exclude '.cache/' \
  --exclude 'features/' \
  --exclude '*.part' \
  --exclude 'weights/ghead*' --exclude 'weights/router*' \
  --exclude 'weights/streetview*' --exclude 'weights/photoarm*' \
  --exclude '.env' --exclude '.env.*' --exclude 'secrets*' \
  /Users/ace/projects/onhacks/  USER@INSTANCE:~/onhacks/
```

Included: `chipoint/ scripts/ tests/ upstream/ requirements-gpu.txt
.gitignore EVALUATION.md GPU_PREP.md encoders/` (verified 1.2 GB DINOv3 —
include to skip re-download) and `weights/proj_head_{dinov3,so400m256,
gopt}.pt` only (the ghead/router/streetview/photoarm weights are excluded —
they belong to the disabled reranker/router paths).

Excluded deliberately: `features/` (~698 MB `gallery_merged_cells.npz`,
reranker-only), `data/`, `artifacts/`, `capture/` (local deletion state),
`.venv/`, caches, secrets.

## On the instance (Linux + NVIDIA GPU)

```bash
cd ~/onhacks
bash scripts/setup_instance.sh                     # venv + CUDA 12.8 torch; exits nonzero w/o GPU
.venv/bin/python scripts/fetch_assets.py --galleries --dinov3-weights
#   ^ HF proj heads (verified) + ~15 GB street galleries + 1.2 GB DINOv3
.venv/bin/python scripts/smoke_test.py --device cuda --towers dinov3
.venv/bin/python scripts/smoke_test.py --device cuda \
    --towers so400m256 gopt dinov3 --real-gallery   # after open_clip weights resolve
```

`--galleries` is required for `--real-gallery`; each tower's
`street_gallery/proj_head_gallery_512_{tower}.npy` is ~5 GB fp16 plus the
shared `gallery_all_C_so400m256.npy` coords (~39 MB). OpenCLIP tower weights
are downloaded lazily by `open_clip` on first `Tower(...)` construction —
not hash-pinned by this repo.

## One-shot worker (optional wrapper — see LAUNCH_PLAN.md)

`LAUNCH_PLAN.md` is the authoritative plan; `experiment_config.json` freezes
the experiment (deadline **2026-11-15T21:00Z** = 16:00 EST,
`max_worker_minutes` 180, `approved_spend_usd` 0). `scripts/run_once.sh` is
operational glue only — it chooses no models, samples no data, scores
nothing, and never creates/rents/terminates cloud resources.

```bash
bash scripts/run_once.sh                 # dry-run (default): prints the 3 stage commands, zero side effects
bash scripts/run_once.sh --execute       # ON the Linux CUDA instance only
bash scripts/sync_results.sh <SSH_ALIAS>          # one pull of artifacts/experiment-v1/
bash scripts/sync_results.sh <SSH_ALIAS> --watch  # re-pull every 60s (Ctrl-C stops)
```

`--execute` gates, in order: flock present → Linux-only → nvidia-smi →
config present → python3/GNU-timeout → wall-clock budget =
min(180 min, deadline−now) refusing if elapsed → ≥65 GiB free →
single-worker flock on `artifacts/experiment-v1/.worker.lock` → then ONE GNU
`timeout --signal=TERM --kill-after=30` wraps **all three** stages (setup →
fetch → `run_experiment.py all`, which ends with the `export` stage writing
`demo_examples/`). Everything appends to `artifacts/experiment-v1/worker.log`.

Completion is defined as **both** `report.json` and
`demo_examples/examples.json` present. If only `report.json` exists (run
interrupted after evaluate, before export), `--execute` runs **only**
`.venv/bin/python scripts/run_experiment.py export --root … --config …
--run-dir …` — a CPU-only, offline repair needing `.venv` + `data/` but no
GPU/install/download; dry-run prints that repair command instead of the
full plan. Missing `.venv` is refused with instructions (never auto-installs).

**Worker timeout ≠ billing stop.** The timeout kills the worker process
tree only; provider billing continues until the pod is terminated
separately (approved lifecycle only — this repo never does it). Spot
interruptions can destroy worker storage at any time, so pull results
periodically with `sync_results.sh --watch`; unexported progress can be
lost. SSH aliases (including non-standard ports) live in `~/.ssh/config`.
Sync pulls into local `artifacts/experiment-v1/` — the same path the demo
server reads — uses BatchMode + `StrictHostKeyChecking=yes` (the host key
must already be trusted; verify the fingerprint on a first manual `ssh`
login), treats ANY nonzero rsync exit as failure (rc 23 = partial transfer,
not success), and never deletes local files.

## Demo server (local only)

```bash
uv pip install --python .venv/bin/python -r requirements-demo.txt
.venv/bin/python scripts/serve_demo.py            # recorded mode, 127.0.0.1:8000
.venv/bin/python scripts/serve_demo.py --live     # + live upload (CUDA host only)
```

Recorded mode serves `artifacts/experiment-v1/report.json`,
`prediction_config.json`, and `demo_examples/examples.json` (+images under
`/examples`) exactly as exported — missing files show an explicit waiting
state, nothing is fabricated. `--live` lazily loads the frozen three-tower
StreetArm on first upload (JPEG/PNG/WebP ≤10 MB, decoded in memory only,
never persisted); without CUDA it fails explicitly rather than pretending.
The output is a **consensus calibration** comparison, not encoder training,
and claims no accuracy beyond the measured report.

## Asset / storage budget

- Street galleries: ~15 GB (3 × ~5 GB fp16 + ~39 MB coords).
- OpenCLIP checkpoints (webli, fetched lazily): ~12 GB for the two towers.
- DINOv3: 1.2 GB (rsynced or `--dinov3-weights`).
- Proj heads: ~0.36 GB.
- **Total model assets ≈ 29 GB**; add OS + software + eval scratch —
  **~100 GB storage recommended**.
- Budget: **$10 total**. Preferred shape: 80 GB A100 spot at roughly
  $1.25/GPU-hr list price — **price not guaranteed**; verify current spot
  pricing before launching.
- This repo does **not** auto-terminate instances and does **not** enforce a
  hard spending cap — monitor billing manually.
- CPU smoke (`--device cpu --towers dinov3`) works without a GPU for
  plumbing checks only.

## Explicit non-goals

- No router, photo arm, ghead, reranker, or 23-feature stack.
- `scripts/fetch_assets.py`, `fetch_data.sh`, and the smoke tests never
  download benchmark images/datasets. The experiment runner
  (`scripts/run_experiment.py`, lead-authored) is the ONLY component that
  fetches benchmark data — on the instance it downloads the pinned
  OSV-5M `test.csv` + `images/test/00.zip` (~2.25 GB) at the frozen
  revision and builds the sequence-disjoint manifest itself.
  `scripts/prep_queries.py` remains disabled (superseded by that manifest).
- Smoke tests verify plumbing only; they make no semantic or accuracy claims.
