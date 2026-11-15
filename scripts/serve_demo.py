#!/usr/bin/env python3
"""Local demo server for the Chipoint street-view baseline.

Recorded mode (default): serves artifacts/experiment-v1/{report.json,
prediction_config.json} and demo_examples/examples.json + images exactly as
exported — no re-encoding, no metric derivation, no fabricated content.
Missing files produce an explicit waiting state.

Live mode (--live): lazy-loads the frozen three-tower StreetArm on first
POST /api/predict. Requires CUDA and the downloaded assets; CPU cannot
pretend to be live — failures are explicit HTTP errors, never a silent
fallback to recorded data.

Binds 127.0.0.1 by default. No uploads are persisted; no external calls.
"""
import argparse
import io
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # chipoint imports resolve from the project root

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

RUN_DIR = 'artifacts/experiment-v1'
EXAMPLES_DIR = 'demo_examples'  # under RUN_DIR, written by run_experiment.py export
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise HTTPException(500, f'corrupt JSON artifact {path}: {e}')


def _examples(root):
    """Pass stored example rows through unchanged; only add image_url."""
    rows = _read_json(root / RUN_DIR / EXAMPLES_DIR / 'examples.json')
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise HTTPException(500, 'demo_examples/examples.json is not a list')
    out = []
    for row in rows:
        if not isinstance(row, dict):
            raise HTTPException(500, 'demo_examples/examples.json has non-object rows')
        image = row.get('image')
        safe = isinstance(image, str) and image and '/' not in image \
            and '\\' not in image and image not in ('.', '..')
        out.append({**row, 'image_url': f'/examples/{image}' if safe else None})
    return out


def _baseline_top_n(root):
    cfg = _read_json(root / 'experiment_config.json') or {}
    return int(cfg.get('baseline_top_n', 10))


def create_app(root, live=False):
    root = Path(root).resolve()
    demo_dir = Path(__file__).resolve().parents[1] / 'demo'
    examples_dir = root / RUN_DIR / EXAMPLES_DIR
    app = FastAPI(title='Chipoint street-view demo', docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.arm_lock = threading.Lock()

    @app.get('/api/status')
    def status():
        return {
            'mode': 'live' if live else 'recorded',
            'live_requested': live,
            'cuda_available': _cuda_available(),
            'report': _read_json(root / RUN_DIR / 'report.json'),
            'prediction_config': _read_json(root / RUN_DIR / 'prediction_config.json'),
            'examples': _examples(root),
            'run_dir': RUN_DIR,
        }

    @app.get('/api/examples')
    def examples():
        return {'examples': _examples(root)}

    @app.post('/api/predict')
    def predict(file: UploadFile = File(...)):
        if not live:
            raise HTTPException(
                400, 'live inference is not enabled on this server '
                     '(restart with --live on a CUDA host)')
        image = _decode_upload(file)
        arm, arm_error, top_n, variant, calibrated = _predict_config_and_arm(root)
        if arm is None:
            raise HTTPException(503, arm_error or 'live arm unavailable')
        if not app.state.arm_lock.acquire(blocking=False):
            raise HTTPException(409, 'inference already in progress; retry shortly')
        try:
            coords = arm.predict([image], top_n=top_n)
        finally:
            app.state.arm_lock.release()
        lat, lon = float(coords[0, 0]), float(coords[0, 1])
        return {'lat': lat, 'lon': lon, 'top_n': top_n,
                'variant': variant, 'calibrated': calibrated}

    @app.get('/examples/{filename}')
    def example_file(filename: str):
        """Single exported image only — restricted to the demo_examples dir,
        never the project root. Missing dir/file is a clean 404."""
        if (not filename or filename in ('.', '..') or '/' in filename
                or '\\' in filename):
            raise HTTPException(404, 'not found')
        path = examples_dir / filename
        if not path.is_file():
            raise HTTPException(404, 'not found')
        return FileResponse(path)

    # Static mount last: demo/ serves only the single-page UI.
    app.mount('/', StaticFiles(directory=demo_dir, html=True), name='demo')
    return app


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _decode_upload(file):
    declared = file.headers.get('content-length')
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'upload exceeds {MAX_UPLOAD_BYTES} bytes')
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'upload exceeds {MAX_UPLOAD_BYTES} bytes')
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # extra net; explicit check below is authoritative
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or '').upper()
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(
                400, f'unsupported image format {fmt!r}; use JPEG, PNG, or WebP')
        width, height = probe.size
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                413, f'image is {width}x{height} ({width * height} px); '
                     f'limit is {MAX_IMAGE_PIXELS} pixels')
        probe.verify()
        probe = Image.open(io.BytesIO(data))
        return probe.convert('RGB')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f'not a decodable JPEG/PNG/WebP image: {e}')


def _predict_config_and_arm(root):
    """Return (arm_or_None, arm_error, top_n, variant, calibrated)."""
    pred_cfg = _read_json(root / RUN_DIR / 'prediction_config.json')
    base_n = _baseline_top_n(root)
    if isinstance(pred_cfg, dict) and isinstance(pred_cfg.get('top_n'), int):
        top_n = pred_cfg['top_n']
        variant = 'candidate' if top_n != base_n else 'baseline'
        calibrated = True
    else:
        top_n, variant, calibrated = base_n, 'baseline', False
    arm, error = _get_arm(root)
    return arm, error, top_n, variant, calibrated


_ARM = {'arm': None, 'error': None, 'lock': threading.Lock()}


def _get_arm(root):
    """Process-wide lazy live arm (created once, on first live request)."""
    if _ARM['arm'] is not None:
        return _ARM['arm'], None
    if _ARM['error'] is not None:
        return None, _ARM['error']
    with _ARM['lock']:
        if _ARM['arm'] is not None:
            return _ARM['arm'], None
        if _ARM['error'] is not None:
            return None, _ARM['error']
        try:
            _ARM['arm'] = _build_arm(root)
        except Exception as e:
            _ARM['error'] = f'live arm init failed: {e}'
            return None, _ARM['error']
    return _ARM['arm'], None


def _build_arm(root):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('live mode requires CUDA; this host has none '
                           '(CPU cannot serve live inference)')
    from chipoint.load import load_proj_head
    from chipoint.pipeline import StreetArm
    from chipoint.gallery import Gallery
    from chipoint.towers import Tower
    cfg = _read_json(root / 'experiment_config.json') or {}
    towers_cfg = cfg.get('towers') or ['so400m256', 'gopt', 'dinov3']
    towers, galleries = {}, {}
    coords = root / 'street_gallery/gallery_all_C_so400m256.npy'
    for tag in towers_cfg:
        tower = Tower(tag, device='cuda', root=root)
        head, dim = load_proj_head(root / 'weights' / f'proj_head_{tag}.pt')
        if dim != tower.dim:
            raise RuntimeError(f'projection head dim {dim} != tower dim {tower.dim} ({tag})')
        towers[tag] = (tower, head)
        galleries[tag] = Gallery(
            root / 'street_gallery' / f'proj_head_gallery_512_{tag}.npy',
            coords, device='cuda')
    return StreetArm(towers, galleries,
                     k=int(cfg.get('retrieval_k', 200)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT,
                    help='project root (default: repo containing this script)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--live', action='store_true',
                    help='enable POST /api/predict (requires CUDA + assets)')
    args = ap.parse_args(argv)
    app = create_app(args.root, live=args.live)
    import uvicorn
    mode = 'live' if args.live else 'recorded'
    print(f'serving {mode} demo at http://{args.host}:{args.port} '
          f'(root={Path(args.root).resolve()})', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    sys.exit(main())
