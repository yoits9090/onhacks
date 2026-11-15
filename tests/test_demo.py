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
            r = client.post('/api/predict',
                            files={'file': ('x.jpg', b'not-an-image', 'image/jpeg')})
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
            r = client.post('/api/predict',
                            files={'file': ('x.gif', buf.getvalue(), 'image/gif')})
            self.assertEqual(r.status_code, 400)
            self.assertIn('format', r.json()['detail'])
            # Pixel cap enforced before decode: patch the module constant to a
            # tiny value so an 8x8 image exceeds it (no large allocation).
            jpeg = io.BytesIO()
            Image.new('RGB', (8, 8)).save(jpeg, 'JPEG')
            payload = jpeg.getvalue()
            old = serve_demo.MAX_IMAGE_PIXELS
            serve_demo.MAX_IMAGE_PIXELS = 40  # 8x8=64 > 40 -> 413
            try:
                r = client.post('/api/predict',
                                files={'file': ('x.jpg', payload, 'image/jpeg')})
            finally:
                serve_demo.MAX_IMAGE_PIXELS = old
            self.assertEqual(r.status_code, 413)
            self.assertIn('pixels', r.json()['detail'])

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
            r = client.post('/api/predict',
                            files={'file': ('x.jpg', buf.getvalue(), 'image/jpeg')})
            self.assertEqual(r.status_code, 503)
            self.assertIn('CUDA', r.json()['detail'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
