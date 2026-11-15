---
license: mit
tags:
- image-geolocation
- street-view
- osv-5m
pipeline_tag: image-feature-extraction
datasets:
- osv5m/osv5m
- Jia-py/MP16-Pro
model-index:
- name: Chipoint v2
  results:
  - task:
      type: image-feature-extraction
      name: Image Geolocation
    dataset:
      type: osv5m
      name: OSV-5M
    metrics:
    - name: Acc@1km
      type: accuracy
      value: 2.66
    - name: Acc@25km
      type: accuracy
      value: 44.83
    - name: Acc@200km
      type: accuracy
      value: 81.10
    - name: Acc@750km
      type: accuracy
      value: 92.11
    - name: Acc@2500km
      type: accuracy
      value: 96.82
    - name: GeoScore
      type: geoscore
      value: 4485
    - name: Median error (km)
      type: distance-error
      value: 32
    - name: Mean error (km)
      type: distance-error
      value: 383
  - task:
      type: image-feature-extraction
      name: Image Geolocation (cross-dataset generalization)
    dataset:
      type: im2gps3k
      name: im2gps3k (street-view rerank)
    metrics:
    - name: Acc@25km
      type: accuracy
      value: 11.68
    - name: GeoScore
      type: geoscore
      value: 2018.5
- name: Chipoint v2 (general-photo arm)
  results:
  - task:
      type: image-feature-extraction
      name: Image Geolocation
    dataset:
      type: im2gps3k
      name: im2gps3k (photo)
    metrics:
    - name: Acc@1km
      type: accuracy
      value: 16.5
    - name: Acc@25km
      type: accuracy
      value: 41.2
    - name: Acc@200km
      type: accuracy
      value: 54.1
    - name: Acc@750km
      type: accuracy
      value: 70.7
    - name: Acc@2500km
      type: accuracy
      value: 84.8
  - task:
      type: image-feature-extraction
      name: Image Geolocation
    dataset:
      type: yfcc4k
      name: YFCC4k (photo)
    metrics:
    - name: Acc@1km
      type: accuracy
      value: 19.1
    - name: Acc@25km
      type: accuracy
      value: 34.5
    - name: Acc@200km
      type: accuracy
      value: 44.2
    - name: Acc@750km
      type: accuracy
      value: 60.2
    - name: Acc@2500km
      type: accuracy
      value: 76.0
---

<p align="center"><img src="wordmark.svg" alt="chiikabu labs" width="340"></p>

<p align="center"><a href="https://chiikabu.cc">chiikabu.cc</a> &nbsp;·&nbsp; <a href="https://chiikabu.cc/discord">discord</a> &nbsp;·&nbsp; <a href="https://x.com/Chiikabu_Labs">twitter / x</a></p>

Chipoint v2 is a SOTA geolocation model that is open-weight and beats the
published frontier closed-weight model (Pinpoint, June 2026), its predecessor (Chipoint v1, July 2026), and all models
on the OSV-5M street-view benchmark. This repo hosts both arms:

- **street-view arm** — the 4.89M-frame OSV-5M gallery index, the
  projection heads, and the 23-feature reranker pipeline
  (`streetview_k512_joint.pt`, K512, per-threshold confidence head)
- **general-photo arm** — the unified 14.5M photo gallery (MP14 10.9M +
  MP16-Pro 4.1M) and the 23-feature rerankers
  (`photoarm_k768_thr.pt` / `photoarm_k768_sr.pt` / `photoarm_k512_thr.pt`)
- `router_osv_flickr.pt` routes queries between the two arms for the best accuracy on a provided image.

## Street-view arm

![chipoint_v2_mean_error](images/chipoint_v2_mean_error.png)
![chipoint_v2_geoscore](images/chipoint_v2_geoscore.png)

### OSV-5M benchmark

OSV-5M test set (210,122 images). Acc@Xkm is the share of predictions
within X km. GeoScore is `5000·exp(-d/1492.7)`, higher is better.

| Model | @25km | @200km | @750km | @2500km | GeoScore | Mean err |
|---|---|---|---|---|---|---|
| OSV-5M Baseline | – | – | – | – | 3361 | 1814 km |
| GeoCLIP | 21.5 | 52.1 | 72.1 | – | – | – |
| GRE | 9.7 | 35.6 | 72.5 | 91.1 | – | 1192 km |
| LocDiff | 11.0 | 46.3 | 77.0 | 88.2 | – | – |
| RFM (S2S) | – | – | – | – | 3767 | 1069 km |
| HierLoc | – | – | – | – | 3963 | 861 km |
| Pinpoint (retrieval only) | 32.1 | 65.6 | 82.8 | 92.8 | 4035 | 784 km |
| Pinpoint (full) | 35.6 | 67.5 | 83.7 | 93.2 | 4114 | 743 km |
| Chipoint v1 | 37.63 | 70.56 | 84.94 | 93.36 | 4174 | 716 km |
| **Chipoint v2 · street (full pipeline)** | **44.8** | **81.1** | **92.1** | **96.8** | **4485** | **383 km** |

Chipoint v2's median error is **32 km** (v1: 51 km). The full-pipeline
line is retrieval (K=64 pool) + the K512 23-feature reranker with a
multiscale blend on the full 210,122-image test split. For reference, raw
retrieval without reranking scores 34.89 / 61.96 / 74.60 / 85.34,
GeoScore 3742, mean 1427 km.

## General-photo arm

![chipoint_v2_im2gps3k_acc1km](images/chipoint_v2_im2gps3k_acc1km.png)
![chipoint_v2_yfcc4k_acc1km](images/chipoint_v2_yfcc4k_acc1km.png)

# Running the model

Instructions on how to run Chipoint v2 locally are in the RUN_GUIDE.md
file. Quick check:

```bash
python run_chipoint.py weights/streetview_k512_joint.pt
python run_chipoint.py weights/photoarm_k768_thr.pt
```