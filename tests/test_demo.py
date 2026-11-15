"""Narrow demo-server checks: recorded waiting state, frozen fixtures,
static example route, upload rejection without --live. No models/GPU."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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


class DemoTests(unittest.TestCase):
    def test_empty_root_waiting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(serve_demo.create_app(tmp, live=False))
            r = client.get('/api/status')
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body['mode'], 'recorded')
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
            self.assertFalse(status['cuda_available'])
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
