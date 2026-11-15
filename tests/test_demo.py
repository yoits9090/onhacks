"""Narrow demo-server checks: immutable recordings, UI routing, validated uploads
and trusted upstream transport. No models/GPU or network requests."""
import asyncio
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'serve_demo', ROOT / 'scripts' / 'serve_demo.py')
serve_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_demo)

from fastapi.testclient import TestClient  # noqa: E402


def fixture_root(tmp):
    root = Path(tmp)
    (root / 'experiment_config.json').write_text(json.dumps({
        'towers': ['so400m256', 'gopt', 'dinov3'],
        'baseline_top_n': 10, 'retrieval_k': 200}))
    run = root / 'artifacts' / 'experiment-v1'
    run.mkdir(parents=True)
    (run / 'report.json').write_text(json.dumps({
        'scope': 'fixture scope', 'evaluation_queries': 4,
        'evaluation_sequences': 4, 'selected_top_n': 5,
        'mean_geoscore_delta': 12.5,
        'paired_sequence_bootstrap_95pct': [1.0, 24.0],
        'accept_candidate': True, 'decision': 'use selected candidate',
        'baseline': {'geoscore': 40.0, 'mean_km': 900.0},
        'selected': {'geoscore': 52.5, 'mean_km': 700.0}}))
    (run / 'prediction_config.json').write_text(json.dumps(
        {'top_n': 5, 'towers': ['dinov3'], 'kind': 'fixture'}))
    exdir = run / 'demo_examples'  # export stage writes under the run dir
    exdir.mkdir()
    (exdir / 'example-0.jpg').write_bytes(b'\xff\xd8\xfffixture-jpeg-bytes')
    (exdir / 'examples.json').write_text(json.dumps([{
        'id': 'img-1', 'sequence': 'seq-1', 'image': 'example-0.jpg',
        'true_lat': 48.1, 'true_lon': 17.9,
        'baseline_lat': 48.0, 'baseline_lon': 17.8, 'baseline_km': 900.0,
        'selected_lat': 48.05, 'selected_lon': 17.85, 'selected_km': 700.0,
        'prediction_lat': 48.05, 'prediction_lon': 17.85,
        'deployed': 'candidate'}]))
    return root


def mock_upstream(handler):
    """Real async httpx client, but every upstream call stays in MockTransport."""
    client_class = httpx.AsyncClient
    return mock.patch.object(
        serve_demo.httpx, 'AsyncClient',
        side_effect=lambda **kwargs: client_class(
            transport=httpx.MockTransport(handler), **kwargs))


class DemoTests(unittest.TestCase):
    def test_empty_root_waiting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=False))
            r = client.get('/api/status')
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body['mode'], 'recorded')
            self.assertFalse(body['live_requested'])
            self.assertFalse(body['can_predict'])
            self.assertIn('Recorded', body['prediction_unavailable_reason'])
            self.assertIsNone(body['report'])
            self.assertIsNone(body['prediction_config'])
            self.assertEqual(body['examples'], [])

    def test_frozen_fixtures_returned_unchanged_and_static_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            client = TestClient(serve_demo.create_app(root, live=False))
            body = client.get('/api/status').json()
            self.assertEqual(body['mode'], 'recorded')
            self.assertEqual(body['report']['decision'], 'use selected candidate')
            self.assertEqual(body['report']['baseline']['geoscore'], 40.0)
            self.assertEqual(body['prediction_config']['top_n'], 5)
            ex = body['examples'][0]
            self.assertEqual(ex['image_url'], '/examples/example-0.jpg')
            self.assertEqual(ex['true_lat'], 48.1)
            self.assertEqual(ex['deployed'], 'candidate')
            r = client.get('/examples/example-0.jpg')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.content, b'\xff\xd8\xfffixture-jpeg-bytes')
            r2 = client.get('/api/examples')
            self.assertEqual(r2.json()['examples'][0]['id'], 'img-1')

    def test_recorded_mode_rejects_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=False))
            r = client.post('/api/predict', content=b'not-an-image',
                            headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r.status_code, 400)
            self.assertIn('live', r.json()['detail'])

    def test_home_and_lab_routing_with_and_without_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            demo = root / 'demo'
            demo.mkdir()
            lab = b'<html>original lab</html>'
            (demo / 'index.html').write_bytes(lab)
            with mock.patch.object(serve_demo, 'DEMO_DIR', demo):
                client = TestClient(serve_demo.create_app(root))
            self.assertEqual(client.get('/').content, lab)
            exposure = b'<html>new exposure app</html>'
            (demo / 'exposure.html').write_bytes(exposure)
            (demo / 'exposure.css').write_text('body { color: blue; }')
            (demo / 'vendor').mkdir()
            (demo / 'vendor' / 'map.js').write_text('/* vendor */')
            for url in ('/', '/exposure.html'):
                self.assertEqual(client.get(url).content, exposure)
            for url in ('/lab', '/lab/', '/index.html'):
                self.assertEqual(client.get(url).content, lab)
            for url in ('/exposure.css', '/vendor/map.js'):
                self.assertEqual(client.get(url).status_code, 200)
            for url in ('/experiment_config.json', '/scripts/serve_demo.py',
                        '/artifacts/experiment-v1/report.json'):
                self.assertEqual(client.get(url).status_code, 404)
            # Entry pages retain StaticFiles' symlink-escape protection.
            (demo / 'exposure.html').unlink()
            (demo / 'exposure.html').symlink_to(root / 'experiment_config.json')
            self.assertEqual(client.get('/').status_code, 404)
            self.assertEqual((demo / 'index.html').read_bytes(), lab)

    def test_repo_lab_serves_existing_index_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp))
            expected = (ROOT / 'demo' / 'index.html').read_bytes()
            self.assertEqual(client.get('/lab').content, expected)
            self.assertEqual(client.get('/lab/').content, expected)

    def test_cli_default_run_selection_and_explicit_override(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            self.assertEqual(serve_demo._default_run_dir(root), serve_demo.RUN_DIR)
            attempt = root / 'results' / 'attempt-2'
            attempt.mkdir(parents=True)
            for name in ('report.json', 'prediction_config.json'):
                (attempt / name).write_text('{}')
            self.assertEqual(serve_demo._default_run_dir(root), 'results/attempt-2')
            v2 = root / 'artifacts' / 'experiment-v2'
            (v2 / 'demo_examples').mkdir(parents=True)
            for name in ('report.json', 'prediction_config.json'):
                (v2 / name).write_text('{}')
            examples = v2 / 'demo_examples' / 'examples.json'
            for invalid in (None, '{', '{}', '[1]'):
                if invalid is not None:
                    examples.write_text(invalid)
                self.assertEqual(serve_demo._default_run_dir(root), 'results/attempt-2')
            examples.write_text('[]')
            self.assertEqual(serve_demo._default_run_dir(root), 'artifacts/experiment-v2')
            with mock.patch.object(serve_demo, 'create_app') as create, \
                    mock.patch('uvicorn.run') as run, redirect_stdout(io.StringIO()):
                serve_demo.main(['--root', tmp])
                self.assertEqual(create.call_args.kwargs['run_dir'], 'artifacts/experiment-v2')
                serve_demo.main(['--root', tmp, '--run-dir', 'chosen-missing-run',
                                 '--inference-url', 'http://127.0.0.1:8002/'])
                self.assertEqual(create.call_args.kwargs['run_dir'], 'chosen-missing-run')
                self.assertEqual(create.call_args.kwargs['inference_url'], 'http://127.0.0.1:8002')
                self.assertEqual(run.call_args.kwargs['host'], '127.0.0.1')
            (v2 / 'report.json').write_text('broken')
            self.assertEqual(serve_demo._default_run_dir(root), 'results/attempt-2')
            (attempt / 'prediction_config.json').unlink()
            self.assertEqual(serve_demo._default_run_dir(root), serve_demo.RUN_DIR)

    def test_exif_orientation_is_applied_before_rgb_conversion(self):
        from PIL import Image
        image = Image.new('RGBA', (2, 3))
        image.putdata([(255, 0, 0, 255), (0, 255, 0, 255),
                       (0, 0, 255, 255), (255, 255, 0, 255),
                       (255, 0, 255, 255), (0, 255, 255, 255)])
        exif = Image.Exif()
        exif[274] = 6  # Display rotated 90 degrees clockwise.
        for fmt in ('PNG', 'JPEG'):
            with self.subTest(format=fmt):
                buf = io.BytesIO()
                source = image if fmt == 'PNG' else image.convert('RGB')
                source.save(buf, fmt, exif=exif)
                decoded = serve_demo._decode_image(buf.getvalue())
                self.assertEqual(decoded.size, (3, 2))
                self.assertEqual(decoded.mode, 'RGB')
                self.assertIsNone(decoded.getexif().get(274))
                if fmt == 'PNG':
                    self.assertEqual(decoded.getpixel((0, 0)), (255, 0, 255))
                    self.assertEqual(decoded.getpixel((2, 0)), (255, 0, 0))
                decoded.close()

    def test_failed_live_initialization_is_not_advertised_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(serve_demo, '_cuda_available', return_value=True), \
                mock.patch.dict(serve_demo._ARM, {'arm': None, 'error': None}), \
                mock.patch.object(serve_demo, '_build_arm',
                                  side_effect=RuntimeError('/private/asset missing')) as build:
            client = TestClient(serve_demo.create_app(fixture_root(tmp), live=True))
            status = client.get('/api/status').json()
            self.assertTrue(status['can_predict'])  # CUDA-capable, lazy init not tried yet.
            self.assertIsNone(status['prediction_unavailable_reason'])
            build.assert_not_called()
            result = client.post('/api/predict', content=self._live_jpeg())
            self.assertEqual(result.status_code, 503)
            status = client.get('/api/status').json()
            self.assertTrue(status['cuda_available'])
            self.assertEqual(status['mode'], 'live')
            self.assertFalse(status['can_predict'])
            self.assertIn('initialize', status['prediction_unavailable_reason'])
            self.assertNotIn('/private/', status['prediction_unavailable_reason'])

    def test_live_rejects_unsupported_format_and_oversize_pixels(self):
        import io
        import torch
        if torch.cuda.is_available():
            self.skipTest('CUDA present; live arm path not exercised here')
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=True))
            # GIF is decodable but not an allowed upload format.
            buf = io.BytesIO()
            Image.new('RGB', (8, 8)).save(buf, 'GIF')
            r = client.post('/api/predict', content=buf.getvalue(),
                            headers={'Content-Type': 'image/gif'})
            self.assertEqual(r.status_code, 400)
            self.assertIn('format', r.json()['detail'])
            # multipart/form-data is refused before any parsing.
            r = client.post('/api/predict', content=b'--x\r\n\r\n',
                            headers={'Content-Type':
                                     'multipart/form-data; boundary=x'})
            self.assertEqual(r.status_code, 400)
            self.assertIn('raw image bytes', r.json()['detail'])
            # Pixel cap enforced before decode: patch the module constant to a
            # tiny value so an 8x8 image exceeds it (no large allocation).
            jpeg = io.BytesIO()
            Image.new('RGB', (8, 8)).save(jpeg, 'JPEG')
            payload = jpeg.getvalue()
            old = serve_demo.MAX_IMAGE_PIXELS
            serve_demo.MAX_IMAGE_PIXELS = 40  # 8x8=64 > 40 -> 413
            try:
                r = client.post('/api/predict', content=payload,
                                headers={'Content-Type': 'image/jpeg'})
            finally:
                serve_demo.MAX_IMAGE_PIXELS = old
            self.assertEqual(r.status_code, 413)
            self.assertIn('pixels', r.json()['detail'])

    def test_custom_run_dir_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            run2 = root / 'artifacts' / 'experiment-v2'
            (run2 / 'demo_examples').mkdir(parents=True)
            (run2 / 'report.json').write_text(json.dumps(
                {'decision': 'v2 marker', 'scope': 'v2 fixture'}))
            (run2 / 'prediction_config.json').write_text(json.dumps(
                {'top_n': 20, 'kind': 'v2-fixture'}))
            (run2 / 'demo_examples' / 'examples.json').write_text('[]')
            client = TestClient(serve_demo.create_app(
                root, live=False, run_dir='artifacts/experiment-v2'))
            body = client.get('/api/status').json()
            self.assertEqual(body['report']['decision'], 'v2 marker')
            self.assertEqual(body['prediction_config']['top_n'], 20)
            self.assertTrue(body['run_dir'].endswith('experiment-v2'))
            self.assertEqual(body['examples'], [])
            # Default run_dir still serves the v1 fixture unchanged.
            client1 = TestClient(serve_demo.create_app(root, live=False))
            self.assertEqual(client1.get('/api/status').json()['report']
                             ['decision'], 'use selected candidate')

    def _live_jpeg(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (8, 8)).save(buf, 'JPEG')
        return buf.getvalue()

    def test_live_forwards_top_n_and_tower_weights(self):
        import numpy as np
        from unittest import mock
        weights = {'so400m256': 1.0, 'gopt': 0.5, 'dinov3': 2.0}
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            run = root / 'artifacts' / 'experiment-v1'
            (run / 'prediction_config.json').write_text(json.dumps(
                {'top_n': 10, 'tower_weights': weights, 'kind': 'fusion'}))

            class FakeArm:
                def __init__(self):
                    self.calls = []

                def predict(self, images, *, top_n=10, tower_weights=None):
                    self.calls.append({'top_n': top_n,
                                       'tower_weights': tower_weights,
                                       'n_images': len(images)})
                    return np.array([[48.0, 17.0]])

            fake = FakeArm()
            client = TestClient(serve_demo.create_app(root, live=True))
            with mock.patch.object(serve_demo, '_get_arm',
                                   return_value=(fake, None)):
                r = client.post('/api/predict', content=self._live_jpeg(),
                                headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r.status_code, 200)
            j = r.json()
            # top_n equals baseline 10, but non-unit weights still deploy
            # the candidate variant.
            self.assertEqual(j['variant'], 'candidate')
            self.assertEqual(j['top_n'], 10)
            self.assertEqual(j['tower_weights'], weights)
            self.assertTrue(j['calibrated'])
            self.assertEqual(fake.calls[0]['top_n'], 10)
            self.assertEqual(fake.calls[0]['tower_weights'], weights)
            self.assertEqual(fake.calls[0]['n_images'], 1)

    def test_invalid_tower_weights_fail_clear(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            run = root / 'artifacts' / 'experiment-v1'
            cfg_path = run / 'prediction_config.json'
            client = TestClient(serve_demo.create_app(root, live=True))
            arm = mock.Mock()
            arm.predict.return_value = [[48.0, 17.0]]
            for bad in ({'so400m256': 1.0},                      # missing keys
                        {'so400m256': 1.0, 'gopt': -1.0,
                         'dinov3': 2.0},                         # non-positive
                        {'so400m256': 1.0, 'gopt': 0.5,
                         'dinov3': 'x'}):                        # non-numeric
                cfg_path.write_text(json.dumps(
                    {'top_n': 10, 'tower_weights': bad}))
                with mock.patch.object(serve_demo, '_get_arm',
                                       return_value=(arm, None)) as m:
                    r = client.post('/api/predict', content=self._live_jpeg(),
                                    headers={'Content-Type': 'image/jpeg'})
                self.assertEqual(r.status_code, 503)
                self.assertIn('tower_weights', r.json()['detail'])
                arm.predict.assert_not_called()

    def test_live_mode_without_cuda_fails_explicitly(self):
        # --live on a CUDA-less host must error, never pretend recorded.
        import torch
        if torch.cuda.is_available():
            self.skipTest('CUDA present; CPU-fallback path not exercised')
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=True))
            status = client.get('/api/status').json()
            self.assertEqual(status['mode'], 'live')
            self.assertTrue(status['live_requested'])
            self.assertFalse(status['cuda_available'])
            self.assertFalse(status['can_predict'])
            self.assertIn('CUDA', status['prediction_unavailable_reason'])
            from PIL import Image
            buf = __import__('io').BytesIO()
            Image.new('RGB', (8, 8), (120, 30, 200)).save(buf, 'JPEG')
            r = client.post('/api/predict', content=buf.getvalue(),
                            headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r.status_code, 503)
            self.assertIn('CUDA', r.json()['detail'])

    def test_large_raw_upload_never_touches_multipart_spool(self):
        # A >1 MiB valid image must succeed without Starlette's multipart
        # parser ever running — it would spool the body to disk.
        import io
        import numpy as np
        import starlette.formparsers as sfp
        from PIL import Image
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            client = TestClient(serve_demo.create_app(root, live=True))
            rng = np.random.default_rng(0)
            noise = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(noise, 'RGB').save(buf, 'PNG')
            payload = buf.getvalue()
            self.assertGreater(len(payload), 1024 * 1024)

            class FakeArm:
                def predict(self, images, *, top_n=10, tower_weights=None):
                    return np.array([[48.0, 17.0]])

            def _boom(*a, **k):
                raise AssertionError('multipart SpooledTemporaryFile used')

            with mock.patch.object(serve_demo, '_get_arm',
                                   return_value=(FakeArm(), None)), \
                    mock.patch.object(sfp, 'SpooledTemporaryFile', _boom):
                r = client.post('/api/predict', content=payload,
                                headers={'Content-Type': 'image/png'})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['lat'], 48.0)

    def test_oversize_raw_body_rejected_before_decode(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=True))
            big = b'\xff\xd8\xff' + b'\x00' * serve_demo.MAX_UPLOAD_BYTES
            with mock.patch.object(serve_demo, '_decode_image') as dec:
                r = client.post('/api/predict', content=big,
                                headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r.status_code, 413)
            dec.assert_not_called()
            # Recorded mode rejects before the body is even consumed.
            client_rec = TestClient(serve_demo.create_app(tmp, live=False))
            r2 = client_rec.post('/api/predict', content=big,
                                 headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r2.status_code, 400)

    def test_concurrent_live_requests_second_gets_409(self):
        # The arm lock must be held across lazy init + inference in the
        # threadpool, so a second request gets an immediate 409 while the
        # first is still blocked inside arm.predict — the event loop stays
        # free. Events, not sleeps, drive the interleaving.
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from unittest import mock
        entered, release = threading.Event(), threading.Event()

        class FakeArm:
            def predict(self, images, *, top_n=10, tower_weights=None):
                entered.set()
                release.wait(timeout=5)
                return np.array([[48.0, 17.0]])

        with tempfile.TemporaryDirectory() as tmp:
            app = serve_demo.create_app(fixture_root(tmp), live=True)
            jpeg = self._live_jpeg()

            with TestClient(app) as client, \
                    mock.patch.object(serve_demo, '_get_arm',
                                      return_value=(FakeArm(), None)):
                def do_post():
                    return client.post('/api/predict', content=jpeg,
                                       headers={'Content-Type': 'image/jpeg'})
                with ThreadPoolExecutor(max_workers=2) as ex:
                    first = ex.submit(do_post)
                    self.assertTrue(entered.wait(timeout=5),
                                    'first request never reached arm.predict')
                    second = ex.submit(do_post)
                    r2 = second.result(timeout=5)
                    self.assertEqual(r2.status_code, 409)
                    release.set()
                    r1 = first.result(timeout=10)
                    self.assertEqual(r1.status_code, 200)
                    self.assertEqual(r1.json()['lat'], 48.0)

    def test_example_route_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            exdir = root / 'artifacts' / 'experiment-v1' / 'demo_examples'
            outside = Path(tmp) / 'outside-secret.jpg'
            outside.write_bytes(b'not-for-serving')
            (exdir / 'linked.jpg').symlink_to(outside)
            client = TestClient(serve_demo.create_app(root, live=False))
            self.assertEqual(client.get('/examples/linked.jpg')
                             .status_code, 404)
            self.assertEqual(client.get('/examples/example-0.jpg')
                             .status_code, 200)

    def test_invalid_prediction_config_top_n_fails_before_arm(self):
        from unittest import mock
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            cfg_path = (root / 'artifacts' / 'experiment-v1'
                        / 'prediction_config.json')
            client = TestClient(serve_demo.create_app(root, live=True))
            arm = mock.Mock()
            arm.predict.return_value = np.array([[48.0, 17.0]])
            for bad in (0, -1, 65, True, '5', None):
                cfg_path.write_text(json.dumps({'top_n': bad}))
                with mock.patch.object(serve_demo, '_get_arm',
                                       return_value=(arm, None)) as m:
                    r = client.post('/api/predict',
                                    content=self._live_jpeg(),
                                    headers={'Content-Type': 'image/jpeg'})
                self.assertEqual(r.status_code, 503, f'top_n={bad!r}')
                self.assertIn('top_n', r.json()['detail'])
                arm.predict.assert_not_called()
                m.assert_not_called()  # invalid config never builds the arm
            # A missing top_n key still takes the legacy baseline fallback.
            cfg_path.write_text(json.dumps({}))
            with mock.patch.object(serve_demo, '_get_arm',
                                   return_value=(arm, None)):
                r = client.post('/api/predict', content=self._live_jpeg(),
                                headers={'Content-Type': 'image/jpeg'})
            self.assertEqual(r.status_code, 200)
            j = r.json()
            self.assertEqual(j['top_n'], 10)
            self.assertFalse(j['calibrated'])
            self.assertEqual(j['variant'], 'baseline')


class RemoteDemoTests(unittest.TestCase):
    upstream = 'http://127.0.0.1:8002'
    prediction = {'lat': 48.0, 'lon': 17.0, 'top_n': 5, 'variant': 'candidate',
                  'calibrated': True, 'tower_weights': None}

    def image_bytes(self):
        from PIL import Image
        buf = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (8, 4)).save(buf, 'JPEG', exif=exif)
        return buf.getvalue()

    def test_invalid_urls_and_mutually_exclusive_modes_fail(self):
        from contextlib import redirect_stderr
        invalid = ('', '127.0.0.1:8002', 'file:///tmp/server', 'ftp://host',
                   'http://', 'http:///status', 'http://user:secret@host',
                   'http://host/api/predict', 'http://host/?token=secret',
                   'http://host/#secret', 'http://host?', 'http://host#',
                   'http://host:bad', 'http://host:0', 'http://host:70000',
                   'http://host\\path', 'http://ho st', 'http://host\n')
        with tempfile.TemporaryDirectory() as tmp:
            for value in invalid:
                with self.subTest(url=value), self.assertRaises(ValueError):
                    serve_demo.create_app(tmp, inference_url=value)
            with self.assertRaises(ValueError):
                serve_demo.create_app(tmp, live=True, inference_url=self.upstream)
        for value in ('http://localhost:8002/', 'https://gpu.example',
                      'http://[::1]:8002'):
            self.assertEqual(serve_demo._validate_inference_url(value), value.rstrip('/'))
        with redirect_stderr(io.StringIO()):
            for flags in (['--live', '--inference-url', self.upstream],
                          ['--inference-url', 'ftp://host']):
                with self.assertRaises(SystemExit) as caught:
                    serve_demo.main(flags)
                self.assertEqual(caught.exception.code, 2)

    def test_proxy_readiness_is_fresh_and_keeps_local_artifacts(self):
        cases = [
            ({'mode': 'live', 'cuda_available': True}, True),
            ({'mode': 'live', 'cuda_available': True, 'can_predict': False,
              'prediction_unavailable_reason': '/private/upstream/model'}, False),
            ({'mode': 'recorded', 'cuda_available': True, 'can_predict': True}, False),
            ({'mode': 'live', 'cuda_available': False, 'can_predict': True}, True),
            ({'mode': 'live', 'cuda_available': False}, False),
            ({'mode': 'live', 'cuda_available': True, 'can_predict': 'yes'}, False),
            (httpx.ConnectError(self.upstream + '/secret'), False),
            (httpx.ReadTimeout(self.upstream + '/secret'), False),
            (httpx.Response(503, json={'detail': 'busy'}), False),
            (httpx.Response(200, content=b'invalid'), False),
            (httpx.Response(200, json=[]), False),
            ({'mode': 'live', 'cuda_available': True, 'can_predict': True}, True),
        ]
        remaining = iter(cases)

        def handle(request):
            self.assertEqual(str(request.url), self.upstream + '/api/status')
            self.assertEqual(request.method, 'GET')
            body, _ = next(remaining)
            if isinstance(body, Exception):
                raise body
            if isinstance(body, httpx.Response):
                return body
            return httpx.Response(200, json={**body, 'run_dir': '/private/upstream',
                                            'report': {'wrong': True}})

        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            expected = json.loads((root / serve_demo.RUN_DIR / 'report.json').read_text())
            client = TestClient(serve_demo.create_app(root, inference_url=self.upstream))
            with mock_upstream(handle) as factory, \
                    mock.patch.object(serve_demo, '_cuda_available') as local_cuda:
                for upstream, ready in cases:
                    body = client.get('/api/status').json()
                    self.assertEqual(body['mode'], 'live')
                    self.assertTrue(body['live_requested'])
                    self.assertIs(body['can_predict'], ready)
                    self.assertIs(body['cuda_available'],
                                  isinstance(upstream, dict)
                                  and upstream.get('cuda_available') is True)
                    self.assertEqual(body['report'], expected)
                    self.assertEqual(body['prediction_config']['top_n'], 5)
                    self.assertEqual(body['examples'][0]['id'], 'img-1')
                    self.assertEqual(body['run_dir'], str((root / serve_demo.RUN_DIR).resolve()))
                    self.assertNotIn('/private/upstream', json.dumps(body))
                    self.assertNotIn(self.upstream, json.dumps(body))
                    if ready:
                        self.assertIsNone(body['prediction_unavailable_reason'])
                    else:
                        self.assertTrue(body['prediction_unavailable_reason'])
                    if isinstance(upstream, httpx.ConnectError):
                        self.assertIn('unreachable', body['prediction_unavailable_reason'])
                    if isinstance(upstream, httpx.ReadTimeout):
                        self.assertIn('timed out', body['prediction_unavailable_reason'])
                local_cuda.assert_not_called()
                self.assertEqual(factory.call_count, len(cases))
                options = factory.call_args.kwargs
                self.assertEqual(options['timeout'].read, 3.0)
                self.assertFalse(options['trust_env'])
                self.assertFalse(options['follow_redirects'])

    def test_proxy_forwards_validated_original_bytes_without_disk_or_local_model(self):
        import starlette.formparsers as sfp
        payload = self.image_bytes() + b'\x00' * (1024 * 1024)

        def handle(request):
            self.assertEqual(request.method, 'POST')
            self.assertEqual(str(request.url), self.upstream + '/api/predict')
            self.assertEqual(request.content, payload)  # Includes original EXIF bytes.
            self.assertEqual(request.headers['content-type'], 'application/octet-stream')
            self.assertNotIn('authorization', request.headers)
            self.assertNotIn('cookie', request.headers)
            return httpx.Response(200, json=self.prediction)

        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_root(tmp)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            client = TestClient(serve_demo.create_app(root, inference_url=self.upstream))
            with mock_upstream(handle) as factory, \
                    mock.patch.object(serve_demo, '_live_predict') as local, \
                    mock.patch.object(serve_demo, '_decode_image',
                                      wraps=serve_demo._decode_image) as decode, \
                    mock.patch.object(sfp, 'SpooledTemporaryFile',
                                      side_effect=AssertionError('multipart spooling')):
                result = client.post('/api/predict?inference_url=http://untrusted.invalid',
                                     content=payload, headers={'Content-Type': 'image/jpeg',
                                                               'Authorization': 'secret',
                                                               'Cookie': 'secret=token'})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json(), self.prediction)
                decode.assert_called_once_with(payload)
                local.assert_not_called()
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(factory.call_args.kwargs['timeout'].read, 180.0)
                self.assertEqual(factory.call_args.kwargs['timeout'].connect, 3.0)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_proxy_propagates_structured_errors_and_sanitizes_transport_errors(self):
        errors = [httpx.Response(code, json={'detail': {'message': 'busy or unavailable',
                                                      'retry': True}})
                  for code in (400, 409, 422, 500, 503)]
        failures = [
            (httpx.ConnectError(self.upstream + '/private/secret'), 503),
            (httpx.ReadTimeout(self.upstream + '/private/secret'), 504),
            (httpx.Response(502, text='secret HTML traceback'), 502),
            (httpx.Response(200, json=[]), 502),
            (httpx.Response(302, headers={'Location': 'http://untrusted.invalid'}), 502),
        ]
        pending = iter(errors + [item for item, _ in failures]
                       + [httpx.Response(200, json=self.prediction)])

        def handle(request):
            response = next(pending)
            if isinstance(response, Exception):
                raise response
            return response

        with tempfile.TemporaryDirectory() as tmp, mock_upstream(handle) as factory, \
                mock.patch.object(serve_demo, '_live_predict') as local:
            client = TestClient(serve_demo.create_app(fixture_root(tmp), inference_url=self.upstream))
            payload = self.image_bytes()
            for upstream in errors:
                result = client.post('/api/predict', content=payload)
                self.assertEqual(result.status_code, upstream.status_code)
                self.assertEqual(result.json(), upstream.json())
            for _, expected_status in failures:
                result = client.post('/api/predict', content=payload)
                self.assertEqual(result.status_code, expected_status)
                self.assertNotIn('secret', result.text)
                self.assertNotIn(self.upstream, result.text)
                self.assertNotIn('Location', result.headers)
                self.assertIn('inference server', result.json()['detail'])
            # All error paths release the lock; there is never a recorded fallback.
            result = client.post('/api/predict', content=payload)
            self.assertEqual(result.json(), self.prediction)
            self.assertEqual(factory.call_count, len(errors) + len(failures) + 1)
            local.assert_not_called()

    def test_proxy_rejects_invalid_and_oversize_uploads_before_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock_upstream(lambda request: self.fail('invalid image reached upstream')) as factory:
            client = TestClient(serve_demo.create_app(tmp, inference_url=self.upstream))
            for data, ctype in ((b'--x\r\n', 'multipart/form-data; boundary=x'),
                                (b'not an image', 'image/jpeg')):
                result = client.post('/api/predict', content=data, headers={'Content-Type': ctype})
                self.assertEqual(result.status_code, 400)
            with mock.patch.object(serve_demo, 'MAX_UPLOAD_BYTES', 8), \
                    mock.patch.object(serve_demo, '_decode_image') as decode:
                for content in (b'0123456789', iter([b'01234', b'56789'])):
                    result = client.post('/api/predict', content=content)
                    self.assertEqual(result.status_code, 413)
                decode.assert_not_called()
            factory.assert_not_called()

    def test_proxy_lock_rejects_concurrent_request_and_releases_afterwards(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        entered, release = threading.Event(), threading.Event()

        async def handle(request):
            entered.set()
            await asyncio.to_thread(release.wait, 5)
            return httpx.Response(200, json=self.prediction)

        with tempfile.TemporaryDirectory() as tmp, mock_upstream(handle) as factory:
            app = serve_demo.create_app(tmp, inference_url=self.upstream)
            with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as workers:
                def post():
                    return client.post('/api/predict', content=self.image_bytes())
                first = workers.submit(post)
                try:
                    self.assertTrue(entered.wait(5), 'first request never reached upstream')
                    second = workers.submit(post).result(timeout=5)
                    self.assertEqual(second.status_code, 409)
                    self.assertIn('in progress', second.json()['detail'])
                    self.assertEqual(factory.call_count, 1)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5).json(), self.prediction)
                self.assertEqual(post().status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
