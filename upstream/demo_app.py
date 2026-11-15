"""Chipoint v2 — Image Geolocation Demo (full two-arm pipeline).

Street-view arm: three encoder towers (SigLIP2 SO400M, GOPT ViT-g, DINOv3 ViT-L),
each with a trained projection head retrieving against its own 4.89M-row OSV-5M
gallery; per-tower z-scored similarities are fused into one candidate pool.
General-photo arm: one unified 14.5M-row projected gallery; the projection space
is identified empirically by a startup self-check.
A router on the DINOv3-projected 512-d query picks the arm (logit > 0 => photo).
Galleries stay on disk as numpy memmaps and are scored in GPU chunks, so VRAM
stays flat regardless of gallery size.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # MUST be imported before any CUDA-touching import (ZeroGPU)

import html as html_module
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = "chiikabu-labs/chipoint-2"
TOP_K_DEFAULT = 10
POOL_M = 200            # retrieval depth per tower / shard pass (reference value)
FUSE_K = 64             # fused candidate pool kept after merging towers/shards
CHUNK_ROWS = 1_000_000  # gallery rows scored per GPU matmul chunk (1M×512 fp16 ≈ 1.0 GB VRAM; safe on 24GB A10G; photo shard ~4M → 4 chunks not 16, street ~4.89M → 5 chunks not 20; ~15+15 matmuls vs 60+58 before)
ENABLE_RERANKER = False  # the 23-feature reranker is wired but OFF: exact
# training-time feature formulas are not published (see README). Fused
# retrieval + spherical weighted consensus is used instead.

# ---------------------------------------------------------------------------
# Networks (architectures from run_chipoint.py)
# ---------------------------------------------------------------------------


class Blk(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.h = h
        self.dk = d // h
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dr = nn.Dropout(p)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, x):
        B, K, D = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = (t.view(B, K, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.dr(self.proj(a.transpose(1, 2).reshape(B, K, D)))
        return x + self.ff(self.n2(x))


class SR(nn.Module):
    def __init__(self, nf, d=256, h=4, l=3, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.LayerNorm(d))
        self.b = nn.ModuleList([Blk(d, h, p) for _ in range(l)])
        self.n = nn.LayerNorm(d)
        self.o = nn.Linear(d, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.b:
            h = b(h)
        return self.o(self.n(h)).squeeze(-1)


class ThrNet(nn.Module):
    def __init__(self, nf, d=256, h=4, l=3, p=0.1, n_thr=5, per_bin=False):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.LayerNorm(d))
        self.b = nn.ModuleList([Blk(d, h, p) for _ in range(l)])
        if per_bin:
            self.ns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_thr)])
            self.os = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_thr)])
        else:
            self.n = nn.LayerNorm(d)
            self.o = nn.Linear(d, n_thr)

    def forward(self, x):
        h = self.inp(x)
        for b in self.b:
            h = b(h)
        if hasattr(self, "ns"):
            return torch.cat([o(n(h)) for n, o in zip(self.ns, self.os)], -1)
        return self.o(self.n(h))


class ProjHead(nn.Module):
    """Linear(in,1024) -> GELU -> Dropout(0.1) -> Linear(1024,512).

    Matches the reference Head exactly (Dropout is identity at eval time; this
    was validated empirically against the bundled examples). Checkpoints also
    carry an unused 'W' geocell-codebook matrix which is filtered out.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load_proj_head(tag: str, input_dim: int) -> ProjHead:
    path = hf_hub_download(MODEL_REPO, f"weights/proj_head_{tag}.pt", repo_type="model")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = ProjHead(input_dim)
    state = {k: v for k, v in ckpt.items() if k.startswith("net.")}
    head.load_state_dict(state)
    return head.to("cuda").half().eval()


# ---------------------------------------------------------------------------
# Data-driven tower registry. A tower whose encoder cannot be loaded is skipped
# loudly and its street gallery file is never downloaded.
# ---------------------------------------------------------------------------

TOWER_REGISTRY = {
    "so400m256": {
        "open_clip_name": "ViT-SO400M-16-SigLIP2-256",
        "input_dim": 1152,
        "gallery": "street_gallery/proj_head_gallery_512_so400m256.npy",
    },
    "gopt": {
        "open_clip_name": "ViT-gopt-16-SigLIP2-256",
        "input_dim": 1536,
        "gallery": "street_gallery/proj_head_gallery_512_gopt.npy",
    },
    "dinov3": {
        # facebook/dinov3-vitl16-pretrain-lvd1689m on the Hub is gated; the
        # ModelScope mirror of the same checkpoint is ungated (see the
        # chiikabu-labs/chipoint RUN_GUIDE). Set DINOV3_PATH to use a local copy.
        "modelscope_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "input_dim": 1024,
        "gallery": "street_gallery/proj_head_gallery_512_dinov3.npy",
    },
}

def _make_dinov3_preprocess(proc):
    """DINOv3 preprocessing matching the reference script: Resize(256)/CenterCrop(224)/normalize."""
    import torchvision.transforms as T

    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    mean = torch.tensor(proc.image_mean).view(3, 1, 1)
    std = torch.tensor(proc.image_std).view(3, 1, 1)

    def preprocess(img):
        return (tf(img.convert("RGB")) - mean) / std

    return preprocess

print("[chipoint-2] Loading encoder towers...", flush=True)
TOWERS = {}  # tag -> dict(preprocess, encode, head)
for tag, spec in TOWER_REGISTRY.items():
    try:
        if "open_clip_name" in spec:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                spec["open_clip_name"], pretrained="webli"
            )
            model = model.to("cuda").half().eval()

            def make_openclip_encode(_m=model):
                @torch.no_grad()
                def encode(pixels):
                    return F.normalize(_m.encode_image(pixels.half()).float(), dim=-1)

                return encode

            encode = make_openclip_encode()
        else:
            from modelscope import snapshot_download

            src = os.environ.get("DINOV3_PATH") or snapshot_download(spec["modelscope_id"])
            from transformers import AutoModel, AutoImageProcessor

            # DINOv3 in fp16 produces NaNs on A10G; the reference script runs
            # encoders under bf16 autocast, so use bf16 weights here too.
            dm = AutoModel.from_pretrained(src, dtype=torch.bfloat16).to("cuda").eval()
            proc = AutoImageProcessor.from_pretrained(src)

            def make_dinov3_encode(_m=dm):
                @torch.no_grad()
                def encode(pixels):
                    o = _m(pixel_values=pixels.to(torch.bfloat16))
                    e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
                    return F.normalize(e.float(), dim=-1)

                return encode

            preprocess = _make_dinov3_preprocess(proc)
            encode = make_dinov3_encode()

        head = _load_proj_head(tag, spec["input_dim"])
        TOWERS[tag] = {"preprocess": preprocess, "encode": encode, "head": head}
        print(f"[chipoint-2] Tower '{tag}' loaded.", flush=True)
    except Exception as e:  # noqa: BLE001 - a missing tower must not kill startup
        print(f"[chipoint-2] WARNING: tower '{tag}' unavailable ({e!r}); skipping it and its gallery.", flush=True)

if not TOWERS:
    raise RuntimeError("No encoder tower could be loaded; cannot serve.")

if "dinov3" not in TOWERS:
    print("[chipoint-2] WARNING: DINOv3 tower missing -> router disabled; Auto mode falls back to Street-view.", flush=True)

# ---------------------------------------------------------------------------
# Downloads: router, street coords + galleries (enabled towers only), photo gallery
# ---------------------------------------------------------------------------

print("[chipoint-2] Downloading router and galleries...", flush=True)

router_path = hf_hub_download(MODEL_REPO, "weights/router_osv_flickr.pt", repo_type="model")
router_ckpt = torch.load(router_path, map_location="cpu", weights_only=False)
router = Router()
router.load_state_dict(router_ckpt["state"])
router = router.to("cuda").float().eval()
ROUTER_MU = torch.from_numpy(np.asarray(router_ckpt["mu"])).to("cuda").float()
ROUTER_SD = torch.from_numpy(np.asarray(router_ckpt["sd"])).to("cuda").float()
print("[chipoint-2] Router loaded (input = DINOv3-projected 512-d query, >0 => photo).", flush=True)

coords_path = hf_hub_download(MODEL_REPO, "street_gallery/gallery_all_C_so400m256.npy", repo_type="model")
STREET_COORDS = np.load(coords_path, mmap_mode="r")

STREET_GALLERIES = {}  # tag -> memmap (enabled towers only)
for tag, spec in TOWER_REGISTRY.items():
    if tag not in TOWERS:
        continue
    print(f"[chipoint-2] Downloading street gallery for '{tag}' (~5 GB)...", flush=True)
    p = hf_hub_download(MODEL_REPO, spec["gallery"], repo_type="model")
    STREET_GALLERIES[tag] = np.load(p, mmap_mode="r")
    print(f"[chipoint-2] Street gallery '{tag}': {STREET_GALLERIES[tag].shape[0]} rows (memmapped).", flush=True)

PHOTO_SHARDS: list = []
PHOTO_LL: list = []
print("[chipoint-2] Downloading unified photo gallery (~14.6 GB across 4 shards)...", flush=True)
for i in range(4):
    emb_p = hf_hub_download(MODEL_REPO, f"photo_gallery/emb_{i:04d}.npy", repo_type="model")
    ll_p = hf_hub_download(MODEL_REPO, f"photo_gallery/ll_{i:04d}.npy", repo_type="model")
    PHOTO_SHARDS.append(np.load(emb_p, mmap_mode="r"))
    PHOTO_LL.append(np.load(ll_p, mmap_mode="r"))
    print(f"[chipoint-2] Photo shard {i}: {PHOTO_SHARDS[-1].shape[0]} rows (memmapped).", flush=True)
PHOTO_TOTAL = sum(mm.shape[0] for mm in PHOTO_SHARDS)

# ---------------------------------------------------------------------------
# Startup self-check: identify which tower's projection space the unified photo
# gallery lives in. The correct space yields clearly higher top-sims (~0.4-0.8).
# ---------------------------------------------------------------------------


def _encode_tower(tag: str, image: Image.Image) -> torch.Tensor:
    """image -> (1,512) unit-norm projected query in tower `tag`'s space (GPU fp32)."""
    t = TOWERS[tag]
    pixels = t["preprocess"](image.convert("RGB")).unsqueeze(0).to("cuda")
    emb = t["encode"](pixels)
    q = t["head"](emb.half()).float()
    return F.normalize(q, dim=-1)


def _probe_photo_space() -> str:
    try:
        probe_img = Image.open("examples/city_skyline_toronto.jpg")
    except Exception:  # noqa: BLE001
        probe_img = None
    scores = {}
    if probe_img is not None:
        slice_n = min(100_000, PHOTO_SHARDS[0].shape[0])
        G = torch.from_numpy(np.asarray(PHOTO_SHARDS[0][:slice_n], dtype=np.float16)).to("cuda").float()
        for tag in TOWERS:
            try:
                q = _encode_tower(tag, probe_img)
                sims = (G @ q.T).squeeze(-1)
                top = torch.topk(sims, 10).values
                mean = float(top.mean())
                if np.isfinite(mean):
                    scores[tag] = mean
                else:
                    print(f"[chipoint-2] photo-space probe: '{tag}' produced non-finite sims; ignoring.", flush=True)
                del q, sims, top
            except Exception as e:  # noqa: BLE001
                print(f"[chipoint-2] photo-space probe failed for '{tag}': {e!r}", flush=True)
        del G
        torch.cuda.empty_cache()
    if not scores:
        tag = next(iter(TOWERS))
        print(f"[chipoint-2] Photo-space probe skipped; defaulting to '{tag}'.", flush=True)
        return tag
    best = max(scores, key=scores.get)
    print(f"[chipoint-2] Photo-gallery space self-check: mean-top10 {scores} -> using '{best}'.", flush=True)
    return best


PHOTO_SPACE_TOWER = _probe_photo_space()

# ---------------------------------------------------------------------------
# Startup warmup: trigger CUDA kernel compilation/cache for each tower so the
# first user query doesn't pay JIT + cold paging. Uses a dummy image under
# no_grad and synchronizes once; failures are non-fatal.
# ---------------------------------------------------------------------------
try:
    _warmup_img = Image.new("RGB", (224, 224), color=(114, 114, 114))
    with torch.no_grad():
        for _tag in list(TOWERS):
            _ = _encode_tower(_tag, _warmup_img)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    print("[chipoint-2] Encoder warmup done.", flush=True)
except Exception as _e:  # noqa: BLE001 - warmup must never kill startup
    print(f"[chipoint-2] Encoder warmup skipped ({_e!r}).", flush=True)
# Optional reranker arms: only materialized when ENABLE_RERANKER is turned on.
RERANKERS = {}


def _maybe_load_rerankers():
    """Load SR/ThrNet arms. Returns quietly when disabled."""
    if not ENABLE_RERANKER:
        return
    for arm, fname in [("street", "weights/streetview_k512_joint.pt"), ("photo", "weights/photoarm_k768_thr.pt")]:
        p = hf_hub_download(MODEL_REPO, fname, repo_type="model")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = SR(ck["nf"])
        net.load_state_dict(ck["state"])
        net = net.to("cuda").half().eval()
        thr = None
        if "thr_state" in ck:
            per_bin = any(k.startswith("ns.") for k in ck["thr_state"])
            thr = ThrNet(ck["nf"], per_bin=per_bin)
            thr.load_state_dict(ck["thr_state"])
            thr = thr.to("cuda").half().eval()
        RERANKERS[arm] = {
            "net": net,
            "thr": thr,
            "names": list(ck["names"]),
            "mu": torch.tensor(np.asarray(ck["mu"]), device="cuda").half(),
            "sd": torch.tensor(np.asarray(ck["sd"]), device="cuda").half(),
            "thr_km": ck.get("thr_km"),
        }


_maybe_load_rerankers()


def _rerank_status(arm: str) -> str:
    """Status note for the 23-feature SR/ThrNet reranker.

    The rerankers are wired (SR/ThrNet load fine, see _maybe_load_rerankers) but
    deliberately not applied: the exact training-time formulas for several of
    the 23 features (rank/agreement semantics, merged-head cell mapping for
    street candidates) were never published, and gallery_merged_cells.npz is
    row-aligned to the photo gallery only. Until the feature stack can be
    validated against reference outputs we refuse to guess scores and serve the
    honest fused-retrieval pipeline instead. This function never raises.
    """
    try:
        if not ENABLE_RERANKER or arm not in RERANKERS:
            return ""
        raise NotImplementedError("23-feature contract not yet validated against reference outputs")
    except Exception as e:  # noqa: BLE001
        print(f"[chipoint-2] reranker unavailable ({e}); serving fused retrieval.", flush=True)
        return " -- reranker requested but unavailable"


# ---------------------------------------------------------------------------
# Retrieval: chunked GPU scoring over disk-backed memmaps. VRAM stays flat.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _scan_topk(mmaps: list, query: torch.Tensor, m: int = POOL_M):
    """Top-m cosine sims of a (1,512) unit-norm CUDA query over concatenated memmaps.

    Returns global row indices (across all mmaps, in order) and sims, CPU numpy.
    """
    # heap stays fp32 for stable merge; query and gallery stay fp16 until topk
    best_sims = torch.full((m,), -1e9, device="cuda", dtype=torch.float32)
    best_idx = torch.zeros((m,), dtype=torch.long, device="cuda")
    offset = 0
    for mm in mmaps:
        n = mm.shape[0]
        for i in range(0, n, CHUNK_ROWS):
            block = mm[i : i + CHUNK_ROWS]  # already float16 memmap — no dtype copy
            if not block.flags["C_CONTIGUOUS"]:
                block = np.ascontiguousarray(block)  # ensure contiguous for from_numpy, no dtype recast
            gb = torch.from_numpy(block).to("cuda", non_blocking=True)  # fp16 on GPU
            # keep matmul in half (1M×512 @ 512×1) → half sims; cast only selected top-m to float for heap merge
            sims = (gb @ query.t()).squeeze(-1)  # still half
            k = m if sims.shape[0] >= m else sims.shape[0]
            cs, ci = torch.topk(sims, k)
            ci = ci + (offset + i)  # global row = shard base + chunk start + within-chunk rank
            cs_f = cs.float()  # cast only the m winners to fp32 for heap merge
            cat_s = torch.cat([best_sims, cs_f])
            cat_i = torch.cat([best_idx, ci])
            order = torch.topk(cat_s, m).indices
            best_sims = cat_s[order]
            best_idx = cat_i[order]
        offset += n
    return best_idx.cpu().numpy(), best_sims.cpu().numpy()


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(np.asarray(lat2, dtype=np.float64) - np.asarray(lat1, dtype=np.float64))
    dlon = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _zc(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def _spherical_weighted_mean(lats, lons, sims):
    """Inverse-sim^4 weighted spherical mean of candidate coordinates."""
    la = np.radians(np.asarray(lats, dtype=np.float64))
    lo = np.radians(np.asarray(lons, dtype=np.float64))
    w = np.clip(np.asarray(sims, dtype=np.float64), 0.0, None) ** 4
    if w.sum() <= 0:
        w = np.ones_like(w)
    w /= w.sum()
    x = np.sum(w * np.cos(la) * np.cos(lo))
    y = np.sum(w * np.cos(la) * np.sin(lo))
    z = np.sum(w * np.sin(la))
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return float(lat), float(lon)


def _route(query_dino: torch.Tensor) -> float:
    with torch.no_grad():
        x = (query_dino.float() - ROUTER_MU) / ROUTER_SD
        return float(router(x).item())


def _street_retrieve(queries: dict):
    """Fuse per-tower retrievals. Returns (fused_idx, fused_score, per_tower)."""
    per_tower = {}
    sc: dict = {}
    for tag in STREET_GALLERIES:
        idx, sim = _scan_topk([STREET_GALLERIES[tag]], queries[tag].half())
        per_tower[tag] = (idx, sim)
        for j, s in zip(idx, _zc(sim)):
            sc[int(j)] = sc.get(int(j), 0.0) + float(s)
    fused_idx = np.array(sorted(sc, key=lambda k: -sc[k])[:FUSE_K], dtype=np.int64)
    fused_score = np.array([sc[int(i)] for i in fused_idx], dtype=np.float32)
    return fused_idx, fused_score, per_tower


def _photo_retrieve(queries: dict):
    q = queries[PHOTO_SPACE_TOWER]
    idx, sim = _scan_topk(PHOTO_SHARDS, q.half())
    return idx, sim


def _coords_for(idx: np.ndarray, source: str) -> np.ndarray:
    if source == "street":
        return np.asarray(STREET_COORDS[idx])
    out = []
    for i in idx:
        i = int(i)
        for mm, start in zip(PHOTO_LL, np.cumsum([0] + [mm.shape[0] for mm in PHOTO_SHARDS[:-1]])):
            if i < start + mm.shape[0]:
                out.append(mm[i - start])
                break
    return np.asarray(out)


def _cluster_consensus(coords: np.ndarray, scores: np.ndarray):
    """Cluster-consensus fallback prediction.

    A global weighted mean over candidates spanning multiple continents is
    meaningless, so instead: seed a cluster at the top-1 candidate, take all
    pool candidates within 250 km great-circle of it (progressively relaxed to
    500 / 1000 km while fewer than 3 candidates fall inside), then return the
    inverse-sim^4 weighted spherical mean over that cluster only.
    """
    members = np.array([0])
    for radius_km in (250.0, 500.0, 1000.0):
        dists = _haversine_km(coords[0, 0], coords[0, 1], coords[:, 0], coords[:, 1])
        members = np.where(dists <= radius_km)[0]
        if members.size >= 3:
            break
    lat, lon = _spherical_weighted_mean(coords[members, 0], coords[members, 1], scores[members])
    return lat, lon, members


def _build_map_html(pred_lat, pred_lon, top_coords, top_sims):
    markers_js = ""
    for i in range(min(len(top_coords), 10)):
        lat = float(top_coords[i, 0])
        lon = float(top_coords[i, 1])
        sim = float(top_sims[i])
        markers_js += f"""
            L.circleMarker([{lat}, {lon}], {{
                radius: 5,
                color: '#3b82f6',
                fillColor: '#3b82f6',
                fillOpacity: 0.5
            }}).addTo(map).bindPopup('#{i+1} sim={sim:.4f}<br>{lat:.4f}, {lon:.4f}');"""

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{pred_lat}, {pred_lon}], 5);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 18,
            attribution: '&copy; OpenStreetMap'
        }}).addTo(map);

        var predIcon = L.divIcon({{
            html: '<div style="font-size: 28px; color: #ef4444; text-align: center; line-height: 28px;">&#9733;</div>',
            iconSize: [28, 28],
            iconAnchor: [14, 14]
        }});
        L.marker([{pred_lat}, {pred_lon}], {{icon: predIcon}})
            .addTo(map)
            .bindPopup('<b>Predicted Location</b><br>{pred_lat:.4f}, {pred_lon:.4f}');

        {markers_js}
    </script>
</body>
</html>"""

    srcdoc = html_module.escape(inner_html)
    return f'<iframe srcdoc="{srcdoc}" style="width:100%; height:400px; border:none; border-radius:10px;"></iframe>'


@spaces.GPU(duration=120)
def geolocate(image: Image.Image, top_k: int = TOP_K_DEFAULT, mode: str = "Auto (router)"):
    """Geolocate an image with the chipoint-2 two-arm pipeline.

    Args:
        image: Input image to geolocate.
        top_k: Number of nearest candidates to display (default 10).
        mode: "Auto (router)", "Street-view", or "General-photo".
    """
    if image is None:
        return "Please upload an image.", None

    try:
        t0 = time.perf_counter()
        top_k = int(max(1, min(int(top_k), FUSE_K)))

        # ---- lazy encode + arm selection (route-first for Auto) --------------
        # Before: queries = {tag: _encode_tower(tag, image) for tag in TOWERS}  # 3 encodes always
        # Now: encode only what the decided arm needs. Photo needs 1 tower,
        # street needs all enabled towers, Auto encodes dinov3 first to route.
        queries: dict = {}
        router_logit = None
        if mode == "Street-view":
            arm = "street"
            queries = {tag: _encode_tower(tag, image) for tag in TOWERS}
        elif mode == "General-photo":
            arm = "photo"
            queries = {PHOTO_SPACE_TOWER: _encode_tower(PHOTO_SPACE_TOWER, image)}
        else:  # Auto (router)
            if "dinov3" in TOWERS:
                dino_q = _encode_tower("dinov3", image)
                queries["dinov3"] = dino_q
                router_logit = _route(dino_q)
                arm = "photo" if router_logit > 0 else "street"
                if arm == "street":
                    for tag in TOWERS:
                        if tag not in queries:
                            queries[tag] = _encode_tower(tag, image)
                else:  # photo: only PHOTO_SPACE_TOWER needed (often dinov3 itself)
                    if PHOTO_SPACE_TOWER not in queries:
                        queries[PHOTO_SPACE_TOWER] = _encode_tower(PHOTO_SPACE_TOWER, image)
            else:
                arm = "street"
                queries = {tag: _encode_tower(tag, image) for tag in TOWERS}

        # ---- retrieval -----------------------------------------------------
        per_tower_lines = []
        rerank_note = _rerank_status(arm)
        if arm == "street":
            fused_idx, fused_score, per_tower = _street_retrieve(queries)
            top_coords_all = np.asarray(STREET_COORDS[fused_idx])
            top_scores = fused_score
            for tag, (idx, sim) in per_tower.items():
                c = np.asarray(STREET_COORDS[idx[0]])
                per_tower_lines.append(f"| {tag} | {sim[0]:.4f} | {c[0]:.4f}°, {c[1]:.4f}° |")
            pool_label = f"{sum(len(v) for v in STREET_GALLERIES.values()):,}"
        else:
            fused_idx, sim = _photo_retrieve(queries)
            top_coords_all = _coords_for(fused_idx, "photo")
            top_scores = sim
            c = top_coords_all[0]
            per_tower_lines.append(
                f"| {PHOTO_SPACE_TOWER} (unified gallery) | {sim[0]:.4f} | {c[0]:.4f}°, {c[1]:.4f}° |"
            )
            pool_label = f"{PHOTO_TOTAL:,}"

        # ---- prediction: cluster-consensus fallback ------------------------
        # Seed at the top-1 candidate, mean over its 250 km cluster (relaxed
        # to 500/1000 km when sparse); also surface the raw top-1 candidate.
        k_disp = min(top_k, len(fused_idx))
        pred_lat, pred_lon, members = _cluster_consensus(
            top_coords_all[:k_disp], top_scores[:k_disp]
        )
        top1_lat, top1_lon = float(top_coords_all[0, 0]), float(top_coords_all[0, 1])

        elapsed = time.perf_counter() - t0
        arm_label = "Street-view" if arm == "street" else "General-photo"
        router_str = f"{router_logit:+.4f}" if router_logit is not None else ("n/a (router disabled)" if mode == "Auto (router)" else "not used")

        lines = [
            "## Predicted Location",
            "",
            f"**Latitude:** {pred_lat:.4f}°  |  **Longitude:** {pred_lon:.4f}°",
            "",
            f"**Raw top-1 candidate:** {top1_lat:.4f}°, {top1_lon:.4f}° (sim {float(top_scores[0]):.4f})",
            "",
            f"**Arm used:** {arm_label}  |  **Router logit:** {router_str} (>0 ⇒ photo)",
            f"**Mode:** {mode}  |  **Inference time:** {elapsed:.2f}s",
            f"**Candidate pool:** {pool_label} rows  |  reranker: {'ON' if ENABLE_RERANKER else 'off (fused retrieval)'}{rerank_note}",
            "",
            "### Per-tower top match",
            "",
            "| Tower | Sim | Coords |",
            "|---|---|---|",
            *per_tower_lines,
            "",
            f"### Top-{k_disp} Fused Candidates",
            "",
            "| # | Score/Sim | Latitude | Longitude | Dist from prediction |",
            "|---|-----------|----------|-----------|----------------------|",
        ]
        for i in range(k_disp):
            lat = float(top_coords_all[i, 0])
            lon = float(top_coords_all[i, 1])
            dist = _haversine_km(pred_lat, pred_lon, lat, lon)
            lines.append(f"| {i+1} | {float(top_scores[i]):.4f} | {lat:.4f}° | {lon:.4f}° | {dist:.1f} km |")

        map_html = _build_map_html(pred_lat, pred_lon, top_coords_all[:k_disp], top_scores[:k_disp])
        return "\n".join(lines), map_html

    except Exception as e:  # noqa: BLE001 - never crash a request
        tb = traceback.format_exc()
        return f"Error: {e}\n\n```\n{tb}\n```", None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

MODES = ["Auto (router)", "Street-view", "General-photo"]

with gr.Blocks(css=CSS, title="Chipoint v2 — Image Geolocation") as demo:
    gr.Markdown(
        "# 🌍 Chipoint v2 — Image Geolocation\n"
        "Upload a photo and the model will predict where in the world it was taken.\n\n"
        "Powered by the full [chiikabu-labs/chipoint-2](https://huggingface.co/chiikabu-labs/chipoint-2) model:\n"
        "**street-view arm** = DINOv3 + SigLIP2 SO400M + GOPT towers × 4.89M OSV-5M gallery each,\n"
        "**general-photo arm** = unified 14.5M-photo gallery, with a trained router choosing between them."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", height=350)
            mode_radio = gr.Radio(MODES, value="Auto (router)", label="Arm", info="Auto routes via the trained street/photo router")
            top_k_slider = gr.Slider(
                minimum=1, maximum=FUSE_K, value=TOP_K_DEFAULT, step=1,
                label="Top-K candidates",
                info="Number of nearest candidates to show",
            )
            run_btn = gr.Button("📍 Locate", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_text = gr.Markdown(label="Prediction Results")
            map_output = gr.HTML(label="Map", visible=True)

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            "### Pipeline\n"
            "1. **Encode**: the image runs through three frozen towers — SigLIP2 SO400M-256, GOPT ViT-g/16 "
            "(both open_clip `webli`) and DINOv3 ViT-L/16 — each projected by a trained MLP head to a unit-norm "
            "512-d vector.\n"
            "2. **Route**: a small router network reads the DINOv3-projected query (standardized by the checkpoint's "
            "μ/σ); logit > 0 selects the general-photo arm, otherwise the street-view arm. You can override it with "
            "the Arm selector.\n"
            "3. **Retrieve (street)**: every enabled tower scores cosine similarity against its own 4.89M-row "
            "OSV-5M gallery in GPU chunks; per-tower z-scored similarities are summed over the union of the top-200 "
            "lists and the best 64 candidates kept.\n"
            "4. **Retrieve (photo)**: the query uses whichever tower's projection space the unified 14.5M-row photo "
            "gallery was built in (identified by a startup self-check) and scans all four shards the same way.\n"
            "5. **Predict**: cluster-consensus fallback — a global mean over candidates spanning continents is "
            "meaningless, so a cluster is seeded at the top-1 candidate and the similarity-weighted spherical mean "
            "(inverse-sim⁴) is taken over that cluster only (250 km radius, relaxed to 500/1000 km when fewer than "
            "3 candidates fall inside). The raw top-1 candidate is also shown.\n\n"
            "### Notes\n"
            "- Galleries stay on disk as memory maps and are streamed through the GPU in 1M-row chunks (~1 GB VRAM per chunk), so VRAM "
            "use stays flat no matter how large the index grows.\n"
            "- The full chipoint-2 rerankers (`streetview_k512_joint.pt`, `photoarm_k768_thr.pt`, 23-feature "
            "SR/ThrNet heads) are wired but currently **off**: the exact training-time feature formulas are not "
            "published, so this demo serves the honest fused-retrieval fallback instead of guessing.\n"
            "- Best results with outdoor photos, streets, landscapes, and recognizable landmarks.\n\n"
            "### Model\n"
            "- **Model**: [chiikabu-labs/chipoint-2](https://huggingface.co/chiikabu-labs/chipoint-2)\n"
            "- **License**: MIT\n"
            "- **Towers**: [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) · "
            "[timm/ViT-gopt-16-SigLIP2-256](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-256) · "
            "[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (via ungated ModelScope mirror)\n"
        )

    gr.Examples(
        examples=[
            ["examples/city_skyline_toronto.jpg", 10, "Auto (router)"],
            ["examples/lighthouse_coast.jpg", 10, "Auto (router)"],
            ["examples/tropical_beach_sunset.jpg", 10, "Auto (router)"],
            ["examples/waterfall_iceland.jpg", 10, "Auto (router)"],
            ["examples/cherry_blossom_pond.jpg", 10, "Auto (router)"],
            ["examples/desert_dunes.jpg", 10, "Auto (router)"],
        ],
        inputs=[input_image, top_k_slider, mode_radio],
        outputs=[result_text, map_output],
        fn=geolocate,
        cache_examples=False,
    )

    run_btn.click(
        fn=geolocate,
        inputs=[input_image, top_k_slider, mode_radio],
        outputs=[result_text, map_output],
        api_name="geolocate",
    )

demo.launch(mcp_server=True, theme=gr.themes.Citrus())

