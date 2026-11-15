# Offgrid — five-minute demo

**A photo says where.** Explore visual location exposure, not safety certification or a precise GPS replacement.

## Open the verified app

**Already running: http://127.0.0.1:8765/**. `/lab` remains the original calibration interface. The running app uses:

```sh
.venv/bin/python scripts/serve_demo.py --host 127.0.0.1 --port 8765 --run-dir artifacts/experiment-v2 --inference-url http://127.0.0.1:8001
```

Use the existing process; do not launch a duplicate. **Leave the existing port-8001 SSH tunnel to the GPU worker up.** No restart of port 8000 is needed.

**Recorded-only alternative, no GPU:** launch separately on a free port (8766 below; change `--port` and the URL if occupied):

```sh
.venv/bin/python scripts/serve_demo.py --host 127.0.0.1 --port 8766 --run-dir artifacts/experiment-v2
```

Open http://127.0.0.1:8766/. This still supports editing and PNG export, but not edited-photo inference.

## Before the clock

- Confirm **`Live inference`** at 8765; fallback mode reads **`Recorded examples`**. Rehearsal passed the real two-request comparison, recorded fallback, mobile crop/export, and browser-console checks with no errors.
- Click **`Reset story`**: returns to the first forest-highway example and clears edits, truth, results, and live-original cache. Toggle the header **`Presentation mode`** icon on; the stage fits **1366×768**. Reset the clock separately with **`Reset timer`**.
- The cue bar is **Photo 0:00 → Reveal 0:30 → Edit 1:30 → Evidence 2:45 → Takeaway 3:45**. **`Start timer`**, **`Pause`**, and **`Resume`** control the clock; cues do not auto-reveal results. Evidence opens a dialog; Takeaway opens the closing-purpose dialog.
- Keep `artifacts/offgrid-final/live-presenter.png` and `artifacts/offgrid-final/report.json` available as verified rehearsal backups. These are generated, ignored local assets; copy them with the demo bundle.

**Useful controls:** `Undo last edit` is an icon; `Reset edits` preserves the original estimate; `Download edited PNG` is the download-arrow tooltip. `Crop` → drag the retained rectangle → `Apply crop`. Sample thumbnails are **hidden in presentation mode**: toggle it **off**, select another sample, then toggle **on**; the timer is preserved.

The labeled local world map works offline. **`Enable street map`** is optional and was tested (six tile responses, HTTP 200); it contacts OpenStreetMap and exposes your IP/map area. Leave it off for a network-independent stage map.

## Exactly five minutes

Quoted text is the spoken script. Read the actual live outcome, not a promised rehearsal result.

| Time | Screen / action | Say |
| --- | --- | --- |
| **0:00–0:30** | `Start timer`; **Photo** cue. First forest-highway image; point to **`No GPS metadata found`**. Keep the estimate hidden. | “Offgrid. A photo says where. This street photo has no GPS metadata, but the road and landscape still contain visual clues. Before sharing a photo, what could someone infer from the pixels alone? We let you explore that question, rather than promise protection.” |
| **0:30–1:30** | **Reveal** cue → `Reveal estimate` → `Reveal known location`. Show **6.63 km** error; pause for the map reveal. | “This first answer is recorded, not a live prediction. It is about 6.6 kilometres from the known location—not an address. These are the first three deterministic evaluation examples, not selected successes. The original baseline was actually closer here, at 5.4 kilometres. An aggregate improvement does not make every answer better.” |
| **1:30–2:45** | **Edit** cue. Ask for one cover/crop suggestion. If indecisive: `Cover` → **`Cover lower center`** → `Apply cover`. Then **`Analyze edited photo`**. Watch original request, then edited request; compare **`Live original`** / **`Live edited`**. Read actual movement. At **20 seconds**, use **`Continue to evidence`** and the fallback line below. | “Which detail would you cover or crop? You choose the rectangle; this is not automatic protection. The app freshly analyzes the normalized original, then the edited pixels, using matching model settings. That avoids confusing a re-export difference with an editing effect.” **After both results:** “Here is how far the estimate moved. That is distance between predictions, not accuracy or a privacy score. This chosen edit is an illustration, not evidence that editing always helps.” |
| **2:45–3:45** | Close any open dialog. Presentation **off** → second thumbnail, straight road/clouds → presentation **on**. Reveal estimate and truth: **1,617 km** miss. Click **Evidence** for the frozen report. Timer keeps running. | “The next example misses by about 1,617 kilometres. Individual failures and aggregate results are different. On 2,000 fresh evaluation sequences, calibration raised GeoScore from 3,963.36 to 4,082.39: a gain of 119.02, with a paired 95-percent interval from 77.14 to 161.11. Within-25-kilometre accuracy rose from 32.8 to 33.5 percent; within-one-kilometre accuracy fell from 0.50 to 0.35 percent. This is a development subset, not the official benchmark.” |
| **3:45–4:40** | Close Evidence; **Takeaway** cue. Explain the system and your contribution alongside the purpose dialog. | “The encoders were pretrained upstream; we did not train them. Three frozen street-view towers retrieve gallery matches, then weighted fusion and geographic consensus produce an estimate. Our contribution is the exposure-exploration workflow, reproducible evaluation, and bounded calibration: SO400M, GOPT, and DINOv3 weights one, two, and one-half, with top-five consensus. We compare against the original public-demo baseline, not the unpublished full system. A later v3 candidate was rejected; v2 stays deployed.” |
| **4:40–5:00** | `Back to the photo`; return to headline. Finish by **4:55**, leaving **five seconds of buffer**. | “Offgrid explores exposure; it does not certify safety. Credit Chiikabu Labs’ Chipoint, OSV-5M, OpenCLIP and SigLIP2, and Meta DINOv3. Removing GPS does not remove visual clues.” |

## Verified live illustration — presenter reference

The rehearsed lower-center cover produced **two sequential real browser requests**, matching v2 **top-5 / SO400M:GOPT:DINOv3 = 1:2:0.5**:

| Result | Latitude, longitude | Error from known location |
| --- | --- | --- |
| Live original | 55.11123066086147, 59.787187858399065 | **8.11 km** |
| Live edited | 56.602911617164416, 41.72018208353963 | **1,144.3 km** |

Displayed **movement: 1,136.3 km**, between those predictions—not either error. Total browser action: **11.754816 seconds**; request timings approximately **6.115346 + 5.430014 seconds**, plus browser overhead. This is measured rehearsal timing, not a latency guarantee. Another audience-chosen rectangle may give a different result.

Raw submitted PNGs were identical outside the exact opaque rectangle **x=318…591, y=204…511** (inclusive). Evidence: `artifacts/offgrid-final/report.json`, `original-input.png`, `edited-input.png`, and both response JSON files in that directory. `live-presenter.png` captures the real result.

**Why the historical 6.63 km differs:** the saved JPEG is a quality-90 display re-export of the evaluation source (`scripts/run_experiment.py:435–490`). The app freshly infers its normalized original before its edit and checks matching model settings. The historical saved prediction is separate and **not used in the live comparison**. A subsequent edit can reuse the matched live-original cache; `Reset story` clears it.

## Frozen evidence, not extra stage time

- The first three saved examples are deterministic manifest entries, not cherry-picked successes. Forest highway: saved estimate **55.1488428, 59.8467519**, truth **55.1032542, 59.9138770**, error **6.6264939 km**, original-baseline error **5.3861824 km**. Straight road/clouds: **1617.116 km**. Winding road: **130.588 km**.
- Keep `artifacts/experiment-v2/report.json` and `prediction_config.json` unchanged. The locked calibration versus original equal-weight/top-10 baseline uses **2,000 fresh OSV-5M archive-01 sequences**, excluding all original 4,000 sequences. The **paired sequence-bootstrap interval** describes GeoScore difference, not a location radius. The selected cover is not another evaluation.
- `artifacts/experiment-v3/report.json` retains accepted v2; do not mix its different cohort/reference into the v2 gain.

## Stage fallback and 90-second minimum

**Slow request:** at 20 seconds click **`Continue to evidence`**. It cancels the client wait and opens evidence; the server may still finish. **Error/unavailable:** go straight to **Evidence**. Do not retry repeatedly or show unfinished output as a comparison. For recorded-only mode, demonstrate a cover/crop and **`Download edited PNG`**, then say:

> “We can edit and download, but this model run is unavailable. We have not measured a changed estimate for this edit. Let’s use the recorded evidence.”

If the app fails, show the verified screenshot explicitly as **rehearsal evidence**, never as today's live output. Keep the failure case and scope; cut extra editing first.

**Interrupted? Skip live inference and audience editing:**

1. **0:00–0:15:** headline, forest photo, GPS absent.
2. **0:15–0:35:** recorded reveal → truth; **6.6 km, not an address**.
3. **0:35–0:55:** presentation off → second sample → on; reveal **1,617 km** miss. First three are deterministic.
4. **0:55–1:15:** Evidence: **+119.02 GeoScore / 2,000 fresh sequences**; subset-only, 25-km accuracy up, 1-km accuracy down. Frozen upstream encoders; our workflow, evaluation, calibration.
5. **1:15–1:30:** Takeaway: v2 retained after v3 rejection; exposure exploration, not safety certification; credits and stop.
