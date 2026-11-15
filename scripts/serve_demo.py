#!/usr/bin/env python3
"""Local demo server for the Chipoint street-view baseline.

Recorded mode (default): serves report.json, prediction_config.json and
demo_examples/examples.json + images exactly as exported — no re-encoding,
no metric derivation, no fabricated content. Missing files produce an explicit
waiting state. The CLI prefers artifacts/experiment-v2 with valid report/config
objects and an examples list, then results/attempt-2 with valid report/config
objects (examples optional), then artifacts/experiment-v1. An explicit
--run-dir always wins; create_app's default remains experiment-v1.

Live mode (--live): lazy-loads the frozen three-tower StreetArm on first
POST /api/predict. Requires CUDA and the downloaded assets; CPU cannot
pretend to be live — failures are explicit HTTP errors, never a silent
fallback to recorded data.

Remote live mode (--inference-url http://127.0.0.1:8002): forwards validated
uploads to a trusted demo server, for example through a private SSH tunnel.
Mutually exclusive with --live. Readiness is checked on every status request;
upstream failures never fall back to recordings. No browser-supplied URLs,
redirect following, or environment proxy configuration are used.

Binds 127.0.0.1 by default. Uploads are decoded in memory only and never
persisted; remote mode forwards the original bytes only to that configured
server. Live initialization may contact the upstream model hosts to
resolve or fetch assets. The page's map tiles and fonts are third-party
browser requests; uploaded image bytes are not sent to those services.
"""
import argparse
import io
import json
import math
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # chipoint imports resolve from the project root

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

RUN_DIR = 'artifacts/experiment-v1'
DEMO_DIR = ROOT / 'demo'
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


def _examples(run_path):
    """Pass stored example rows through unchanged; only add image_url."""
    rows = _read_json(Path(run_path) / EXAMPLES_DIR / 'examples.json')
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


def _default_run_dir(root):
    """Choose complete v2 artifacts, then the attempt-2 report, then legacy v1."""
    required = {'report.json': dict, 'prediction_config.json': dict}
    for directory, files in (
            ('artifacts/experiment-v2',
             {**required, f'{EXAMPLES_DIR}/examples.json': list}),
            ('results/attempt-2', required)):
        try:
            artifacts = {name: json.loads((Path(root) / directory / name).read_text())
                         for name in files}
        except (OSError, ValueError):
            continue
        if all(isinstance(artifacts[name], kind) for name, kind in files.items()) \
                and all(isinstance(row, dict)
                        for row in artifacts.get(f'{EXAMPLES_DIR}/examples.json', [])):
            return directory
    return RUN_DIR


def _validate_inference_url(value):
    """A server-configured HTTP origin only; never accept arbitrary endpoints."""
    message = 'inference URL must be an HTTP(S) base URL without credentials, path, query, or fragment'
    try:
        if not isinstance(value, str) or any(
                c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value) \
                or any(c in value for c in ('\\', '?', '#')):
            raise ValueError(message)
        parts = urlsplit(value)
        if parts.scheme not in ('http', 'https') or not parts.hostname \
                or parts.username is not None or parts.password is not None \
                or parts.path not in ('', '/') or '%' in parts.hostname \
                or (parts.port is not None and not 1 <= parts.port <= 65535):
            raise ValueError(message)
        # httpx also validates the authority before any request is possible.
        httpx.URL(value)
    except (ValueError, httpx.InvalidURL):
        raise ValueError(message) from None
    return value.rstrip('/')


async def _upstream_request(base_url, method, *, data=None):
    """Fixed API routes only. Clients close promptly; image bytes stay in memory."""
    predicting = method == 'POST'
    timeout = httpx.Timeout(connect=3.0, read=180.0 if predicting else 3.0,
                            write=10.0, pool=3.0)
    try:
        async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, trust_env=False,
                limits=httpx.Limits(max_connections=2,
                                   max_keepalive_connections=0)) as client:
            return await client.request(
                method, base_url + ('/api/predict' if predicting else '/api/status'),
                content=data,
                headers={'Content-Type': 'application/octet-stream'} if predicting else {})
    except httpx.TimeoutException:
        raise HTTPException(
            504, 'The inference server timed out. Please try again.') from None
    except httpx.RequestError:
        raise HTTPException(
            503, 'The inference server is unreachable. Check the connection.') from None


async def _upstream_readiness(base_url):
    """Return only readiness, never upstream artifacts, paths or exception text."""
    try:
        response = await _upstream_request(base_url, 'GET')
    except HTTPException as exc:
        return False, False, exc.detail
    if response.status_code != 200:
        return False, False, 'The inference server is unavailable. Try again shortly.'
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return False, False, 'The inference server returned an invalid status.'
    cuda = body.get('cuda_available') is True
    ready = body.get('mode') == 'live' and (
        body.get('can_predict') is True if 'can_predict' in body else cuda)
    return cuda, ready, None if ready else 'The inference server is not ready for predictions.'


async def _upstream_predict(base_url, data):
    response = await _upstream_request(base_url, 'POST', data=data)
    try:
        body = response.json()
    except ValueError:
        body = None
    # Structured errors, including busy responses, retain their status and body.
    # Never forward redirects, HTML error pages, cookies or upstream headers.
    if not isinstance(body, dict) or not (
            response.status_code == 200 or 400 <= response.status_code < 600):
        raise HTTPException(502, 'The inference server returned an invalid response.')
    return JSONResponse(body, status_code=response.status_code)


def create_app(root, live=False, run_dir=RUN_DIR, inference_url=None):
    if live and inference_url is not None:
        raise ValueError('live and inference_url are mutually exclusive')
    if inference_url is not None:
        inference_url = _validate_inference_url(inference_url)
    live_requested = live or inference_url is not None
    root = Path(root).resolve()
    run_path = Path(run_dir)
    if not run_path.is_absolute():
        run_path = root / run_path
    run_path = run_path.resolve()
    demo_dir = DEMO_DIR
    examples_dir = run_path / EXAMPLES_DIR
    app = FastAPI(title='Chipoint street-view demo', docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.state.arm_lock = threading.Lock()

    @app.get('/api/status')
    async def status():
        if inference_url is not None:
            cuda, can_predict, reason = await _upstream_readiness(inference_url)
        else:
            cuda = await run_in_threadpool(_cuda_available)
            can_predict = live and cuda and _ARM['error'] is None
            if not live:
                reason = 'Recorded demo only. Connect a live inference server to upload photos.'
            elif not cuda:
                reason = 'Live predictions require a CUDA GPU or a connected inference server.'
            elif _ARM['error'] is not None:
                reason = 'The live model could not initialize. Check model assets and restart.'
            else:
                reason = None
        return {
            'mode': 'live' if live_requested else 'recorded',
            'live_requested': live_requested,
            'cuda_available': cuda,
            'can_predict': can_predict,
            'prediction_unavailable_reason': reason,
            'report': _read_json(run_path / 'report.json'),
            'prediction_config': _read_json(run_path / 'prediction_config.json'),
            'examples': _examples(run_path),
            'run_dir': str(run_path),
        }

    @app.get('/api/examples')
    def examples():
        return {'examples': _examples(run_path)}

    @app.post('/api/predict')
    async def predict(request: Request):
        # Raw image body only — never request.form()/UploadFile, so Starlette's
        # multipart parser (which spools >1 MiB to disk) is never invoked and
        # uploads stay purely in memory.
        if not live_requested:
            raise HTTPException(
                400, 'live inference is not enabled on this server '
                     '(restart with --live on a CUDA host)')
        ctype = (request.headers.get('content-type') or '') \
            .split(';', 1)[0].strip().lower()
        if ctype.startswith('multipart/'):
            raise HTTPException(
                400, 'raw image bytes required — POST the image body directly '
                     '(Content-Type image/jpeg, image/png, image/webp, or '
                     'application/octet-stream), not multipart/form-data')
        declared = request.headers.get('content-length')
        if declared and declared.isdigit() \
                and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f'upload exceeds {MAX_UPLOAD_BYTES} bytes')
        chunks, total = [], 0
        async for chunk in request.stream():
            if total + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f'upload exceeds {MAX_UPLOAD_BYTES} bytes')
            chunks.append(chunk)
            total += len(chunk)
        data = b''.join(chunks)
        image = _decode_image(data)
        # One GPU job at a time: the lock is taken BEFORE any model work
        # (lazy arm init included) and held across inference, which runs in
        # the threadpool so the event loop stays responsive for a fast 409.
        if not app.state.arm_lock.acquire(blocking=False):
            image.close()
            raise HTTPException(409, 'inference already in progress; retry shortly')
        try:
            if inference_url is not None:
                return await _upstream_predict(inference_url, data)
            return await run_in_threadpool(
                _live_predict, root, run_path, image)
        finally:
            app.state.arm_lock.release()
            image.close()

    @app.get('/examples/{filename}')
    def example_file(filename: str):
        """Single exported image only — restricted to the demo_examples dir,
        never the project root. Missing dir/file is a clean 404."""
        if (not filename or filename in ('.', '..') or '/' in filename
                or '\\' in filename):
            raise HTTPException(404, 'not found')
        path = examples_dir / filename
        resolved = path.resolve()
        if resolved.parent != examples_dir.resolve() or not resolved.is_file():
            raise HTTPException(404, 'not found')
        return FileResponse(resolved)

    # Use StaticFiles even for the entry pages to retain its symlink/traversal
    # protection. The old lab and all assets remain available, never project files.
    demo_files = StaticFiles(directory=demo_dir, html=True)

    @app.get('/')
    async def home(request: Request):
        page = 'index.html'
        for candidate in ('app/index.html', 'exposure.html'):
            if (demo_dir / candidate).is_file():
                page = candidate
                break
        return await demo_files.get_response(page, request.scope)

    @app.get('/lab')
    @app.get('/lab/')
    async def lab(request: Request):
        return await demo_files.get_response('index.html', request.scope)

    # Static mount last: demo/ only, including /exposure.* and vendor assets.
    app.mount('/', demo_files, name='demo')
    return app


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _decode_image(data):
    """Decode raw image bytes fully in memory — the byte cap was already
    enforced while streaming, so nothing here touches disk."""
    try:
        from PIL import Image, ImageOps
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
        return ImageOps.exif_transpose(probe).convert('RGB')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f'not a decodable JPEG/PNG/WebP image: {e}')


def _live_predict(root, run_path, image):
    """Synchronous live inference — config resolution, lazy arm init, and
    StreetArm.predict. Runs in the threadpool, never on the event loop."""
    arm, arm_error, top_n, variant, weights, calibrated = \
        _predict_config_and_arm(root, run_path)
    if arm is None:
        raise HTTPException(503, arm_error or 'live arm unavailable')
    coords = arm.predict([image], top_n=top_n, tower_weights=weights)
    lat, lon = float(coords[0, 0]), float(coords[0, 1])
    return {'lat': lat, 'lon': lon, 'top_n': top_n,
            'variant': variant, 'calibrated': calibrated,
            'tower_weights': weights}


def _validate_tower_weights(root, tw):
    """Check declared weights against the configured tower set.

    Returns (weights_dict, None) or (None, error). Keys must be exactly the
    towers from experiment_config.json; every value a positive finite number.
    """
    cfg = _read_json(root / 'experiment_config.json') or {}
    towers = cfg.get('towers') or ['so400m256', 'gopt', 'dinov3']
    expected = set(towers)
    if not isinstance(tw, dict) or set(tw) != expected:
        return None, ('tower_weights must map exactly the configured towers '
                      f'{sorted(expected)}; got {tw!r}')
    weights = {}
    for tag in expected:
        w = tw[tag]
        if isinstance(w, bool) or not isinstance(w, (int, float)) \
                or not math.isfinite(w) or w <= 0:
            return None, (f'tower_weights[{tag!r}]={w!r} is not a positive '
                          'finite number')
        weights[tag] = float(w)
    return weights, None


def _predict_config_and_arm(root, run_path):
    """Return (arm_or_None, arm_error, top_n, variant, weights, calibrated).

    weights is None for a legacy prediction_config without tower_weights
    (StreetArm then takes its unchanged equal-weight path); an invalid
    provided mapping fails before any arm is built.
    """
    pred_cfg = _read_json(Path(run_path) / 'prediction_config.json')
    base_n = _baseline_top_n(root)
    top_n, calibrated = base_n, False
    weights, weight_error, config_error = None, None, None
    if isinstance(pred_cfg, dict):
        if 'top_n' in pred_cfg:
            tn = pred_cfg['top_n']
            if isinstance(tn, bool) or not isinstance(tn, int) \
                    or not 1 <= tn <= 64:
                config_error = (
                    f'prediction_config.top_n={tn!r} is invalid; expected a '
                    'non-bool integer in [1, 64]')
            else:
                top_n, calibrated = tn, True
        if config_error is None and pred_cfg.get('tower_weights') is not None:
            weights, weight_error = _validate_tower_weights(
                root, pred_cfg['tower_weights'])
    nondefault = weights is not None and any(w != 1 for w in weights.values())
    variant = 'candidate' if (top_n != base_n or nondefault) else 'baseline'
    if config_error is not None:
        return None, config_error, top_n, variant, None, calibrated
    if weight_error is not None:
        return None, weight_error, top_n, variant, None, calibrated
    arm, error = _get_arm(root)
    return arm, error, top_n, variant, weights, calibrated


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
    transport = ap.add_mutually_exclusive_group()
    transport.add_argument('--live', action='store_true',
                           help='enable POST /api/predict (requires CUDA + assets)')
    transport.add_argument('--inference-url', type=_validate_inference_url,
                           help='trusted HTTP(S) demo-server base URL, e.g. an SSH tunnel')
    ap.add_argument('--run-dir',
                    help='run directory with report/prediction_config/'
                         'demo_examples, relative to --root unless absolute '
                         '(default: complete experiment-v2, then attempt-2, then experiment-v1)')
    args = ap.parse_args(argv)
    run_dir = args.run_dir if args.run_dir is not None else _default_run_dir(args.root)
    app = create_app(args.root, live=args.live, run_dir=run_dir,
                     inference_url=args.inference_url)
    import uvicorn
    mode = 'live' if args.live or args.inference_url else 'recorded'
    print(f'serving {mode} demo at http://{args.host}:{args.port} '
          f'(root={Path(args.root).resolve()}, run_dir={run_dir})',
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    sys.exit(main())
