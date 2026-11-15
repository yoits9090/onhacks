#!/usr/bin/env python3
"""Street-arm plumbing smoke test — NOT an accuracy evaluation.

Loads the selected encoder towers + their projection heads, encodes one
image, retrieves against deterministic SYNTHETIC 512-d galleries (default)
or the real downloaded street index (--real-gallery), then runs
StreetArm.predict (baseline.fuse_pools + baseline.consensus).

Synthetic mode exercises shapes, finiteness, fusion and consensus plumbing
only — predicted coordinates carry NO semantic meaning. Real-gallery mode
uses the actual downloaded index but still makes no accuracy claim.
See EVALUATION.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GALLERY_DIM = 512
SYNTH_N = 4096
SYNTH_SEED = 20251115


def _versions():
    import torch
    out = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'numpy': np.__version__,
    }
    for mod in ('transformers', 'torchvision', 'safetensors',
                'huggingface_hub', 'open_clip', 'PIL'):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, '__version__', '?')
        except Exception:
            out[mod] = 'not installed'
    return out


def _synthetic_image(seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, 'RGB')


def _write_synthetic_galleries(root, tags, device):
    """Deterministic synthetic (N,512) fp16 galleries + shared (N,2) coords.

    Every tower gets different random rows but the SAME coords — mirroring
    the real asset where all towers share one row-aligned coord file.
    """
    from chipoint.gallery import Gallery
    base = root / 'artifacts' / 'smoke_synthetic'
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTH_SEED)
    coords = np.stack([
        rng.uniform(-60.0, 70.0, SYNTH_N),    # lat
        rng.uniform(-170.0, 170.0, SYNTH_N),  # lon
    ], axis=1)
    coords_path = base / 'coords.npy'
    np.save(coords_path, coords.astype(np.float64))
    galleries = {}
    for tag in tags:
        emb = rng.normal(size=(SYNTH_N, GALLERY_DIM)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        emb_path = base / f'gallery_{tag}.npy'
        np.save(emb_path, emb.astype(np.float16))
        galleries[tag] = Gallery(emb_path, coords_path, device=device,
                                 chunk=1024, batch=4)
    return galleries, 'SYNTHETIC'


def _real_galleries(root, tags, device):
    from chipoint.gallery import Gallery
    coords_path = root / 'street_gallery' / 'gallery_all_C_so400m256.npy'
    galleries = {}
    for tag in tags:
        emb_path = root / 'street_gallery' / f'proj_head_gallery_512_{tag}.npy'
        for p in (emb_path, coords_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f'{p} missing — fetch with: '
                    'python scripts/fetch_assets.py --galleries '
                    f'--towers {" ".join(tags)}')
        galleries[tag] = Gallery(emb_path, coords_path, device=device)
    return galleries, 'REAL-DOWNLOADED-INDEX'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--towers', nargs='+',
                    choices=['so400m256', 'gopt', 'dinov3'],
                    default=['dinov3'])
    ap.add_argument('--image', type=Path, default=None,
                    help='query image path (default: deterministic synthetic)')
    ap.add_argument('--real-gallery', action='store_true',
                    help='use the downloaded street index instead of synthetic')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--k', type=int, default=200)
    args = ap.parse_args(argv)
    root = args.root.resolve()

    print('== versions ==')
    for k, v in _versions().items():
        print(f'  {k}: {v}')
    print(f'  device: {args.device} | towers: {args.towers} | k: {args.k}')

    from chipoint.load import load_proj_head
    from chipoint.pipeline import StreetArm
    from chipoint.towers import Tower

    towers = {}
    for tag in args.towers:
        tower = Tower(tag, device=args.device, root=root)
        head, head_dim = load_proj_head(root / 'weights' / f'proj_head_{tag}.pt')
        if head_dim != tower.dim:
            raise RuntimeError(
                f'{tag}: proj head input dim {head_dim} != tower dim {tower.dim}')
        towers[tag] = (tower, head)
        print(f'tower {tag}: dim={tower.dim} head={head_dim}->512 ok')

    if args.real_gallery:
        galleries, gallery_kind = _real_galleries(root, args.towers, args.device)
    else:
        galleries, gallery_kind = _write_synthetic_galleries(
            root, args.towers, args.device)
    for tag, g in galleries.items():
        print(f'gallery {tag}: N={g.n} dim={g.dim} dtype=float16 '
              f'chunk={g.chunk} batch={g.batch} kind={gallery_kind}')

    arm = StreetArm(towers, galleries, k=args.k)

    if args.image is not None:
        from PIL import Image
        image = Image.open(args.image).convert('RGB')
        print(f'image: {args.image}')
    else:
        image = _synthetic_image()
        print('image: deterministic synthetic 256x256 RGB (no semantic content)')

    pools = arm.retrieve([image], k=args.k)
    batch = next(iter(pools.values()))[0].shape[0]
    assert batch == 1
    for tag, (sims, ids) in pools.items():
        assert sims.shape == (1, args.k) and ids.shape == (1, args.k)
        assert sims.dtype == torch.float32, sims.dtype
        assert ids.dtype == torch.int64, ids.dtype
        assert torch.isfinite(sims).all(), f'{tag}: non-finite sims'
        assert int(ids.min()) >= 0 and int(ids.max()) < galleries[tag].n
        # sorted descending check
        assert bool((sims[0, 1:] <= sims[0, :-1] + 1e-6).all()), f'{tag}: unsorted'
        c = galleries[tag].coords[int(ids[0, 0])]
        print(f'  {tag}: top1 sim={float(sims[0,0]):.4f} '
              f'gallery_row={int(ids[0,0])} coords=({c[0]:.3f},{c[1]:.3f})')

    # unit-norm sanity on the raw tower embedding path
    e = towers[args.towers[0]][0].encode([image])
    dev = (e.norm(dim=-1) - 1.0).abs().max().item()
    print(f'  tower-embed unit-norm max |dev|: {dev:.3e}')
    assert dev < 1e-3, 'tower embedding not unit-norm'

    pred = arm.predict([image])
    lat, lon = float(pred[0, 0]), float(pred[0, 1])
    kind = ('SYNTHETIC — no semantic meaning' if gallery_kind == 'SYNTHETIC'
            else 'REAL-DOWNLOADED-INDEX — plumbing check only, not an eval')
    print(f'PREDICTION ({kind}): lat={lat:.4f} lon={lon:.4f}')
    print('smoke test PASSED')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - smoke should report, not traceback-dump
        print(f'smoke test FAILED: {e!r}', file=sys.stderr)
        raise
