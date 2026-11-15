"""Narrow tests for chunked Gallery.topk and StreetArm validation.

CPU-only, tiny deterministic arrays, no downloads. Run with pytest or
directly:  python tests/test_gallery.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chipoint.gallery import Gallery  # noqa: E402
from chipoint.pipeline import StreetArm  # noqa: E402
from chipoint.nn import ProjHead  # noqa: E402


def _make_gallery(tmp, n=67, dim=512, chunk=8, batch=3, seed=0, coords=None,
                  name='g'):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb_path = Path(tmp) / f'{name}_emb.npy'
    co_path = Path(tmp) / f'{name}_coords.npy'
    np.save(emb_path, emb.astype(np.float16))
    if coords is None:
        coords = np.stack([rng.uniform(-60, 70, n),
                           rng.uniform(-170, 170, n)], axis=1)
    np.save(co_path, np.asarray(coords, dtype=np.float64))
    return Gallery(emb_path, co_path, device='cpu', chunk=chunk, batch=batch)


def _queries(b=5, dim=512, seed=1):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(rng.normal(size=(b, dim)).astype(np.float32))
    return torch.nn.functional.normalize(q, dim=-1)


def test_chunked_topk_matches_dense():
    """Chunked merge must equal a single dense fp32 topk on the same rows."""
    with tempfile.TemporaryDirectory() as td:
        g = _make_gallery(td, n=67, chunk=8, batch=3)
        q = _queries(5)
        sims, idx = g.topk(q, k=10)
        emb = torch.from_numpy(np.load(Path(td) / 'g_emb.npy')).float()
        ref_sims, ref_idx = (q @ emb.T).topk(10, dim=1)
        assert sims.shape == (5, 10) and idx.shape == (5, 10)
        assert sims.dtype == torch.float32 and idx.dtype == torch.int64
        assert torch.allclose(sims, ref_sims, atol=1e-5), \
            (sims - ref_sims).abs().max()
        assert torch.equal(idx, ref_idx)
        # sorted descending per row
        assert bool((sims[:, 1:] <= sims[:, :-1] + 1e-6).all())


def test_topk_validates_k_and_query():
    with tempfile.TemporaryDirectory() as td:
        g = _make_gallery(td, n=20, chunk=5)
        q = _queries(2)
        for bad_k in (0, -1, 21):
            try:
                g.topk(q, k=bad_k)
            except ValueError:
                pass
            else:
                raise AssertionError(f'k={bad_k} should raise')
        try:
            g.topk(torch.randn(2, 128), k=5)  # wrong dim
        except ValueError:
            pass
        else:
            raise AssertionError('wrong query dim should raise')
        try:
            g.topk(q[:1], k=20)  # k == n is allowed
        except ValueError:
            raise AssertionError('k == N should be allowed')


def test_gallery_rejects_mismatched_or_bad_files():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # coords rowcount mismatch
        emb = np.zeros((10, 512), np.float16)
        np.save(td / 'e.npy', emb)
        np.save(td / 'c.npy', np.zeros((11, 2)))
        try:
            Gallery(td / 'e.npy', td / 'c.npy', device='cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('rowcount mismatch should raise')
        # non-fp16 / wrong dim
        np.save(td / 'e32.npy', np.zeros((10, 512), np.float32))
        np.save(td / 'c10.npy', np.zeros((10, 2)))
        try:
            Gallery(td / 'e32.npy', td / 'c10.npy', device='cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('non-fp16 gallery should raise')


class _FakeTower:
    device = torch.device('cpu')
    dim = 8

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def encode(self, images):
        return torch.from_numpy(
            self.rng.normal(size=(len(images), self.dim)).astype(np.float32))


def _arm_with_galleries(galleries, tags=('a', 'b'), n_cells=8):
    towers = {t: (_FakeTower(seed=i), ProjHead(8, n_cells))
              for i, t in enumerate(tags)}
    return StreetArm(towers, galleries, k=15)


def test_streetarm_rejects_key_and_rowcount_mismatch():
    with tempfile.TemporaryDirectory() as td:
        g10 = _make_gallery(td, n=30, name='a')
        g11 = _make_gallery(td, n=31, name='b')
        # tower/gallery key mismatch
        try:
            StreetArm({'a': (_FakeTower(), ProjHead(8, 4))},
                      {'a': g10, 'b': g11}, k=5)
        except ValueError:
            pass
        else:
            raise AssertionError('key mismatch should raise')
        # rowcount mismatch across galleries
        towers = {'a': (_FakeTower(), ProjHead(8, 4)),
                  'b': (_FakeTower(1), ProjHead(8, 4))}
        try:
            StreetArm(towers, {'a': g10, 'b': g11}, k=5)
        except ValueError:
            pass
        else:
            raise AssertionError('rowcount mismatch should raise')


def test_streetarm_rejects_coord_mismatch():
    with tempfile.TemporaryDirectory() as td:
        coords_a = np.stack([np.full(30, 10.0), np.full(30, 20.0)], 1)
        coords_b = np.stack([np.full(30, 11.0), np.full(30, 20.0)], 1)
        g_a = _make_gallery(td, n=30, name='a', coords=coords_a)
        g_b = _make_gallery(td, n=30, name='b', coords=coords_b)
        towers = {'a': (_FakeTower(), ProjHead(8, 4)),
                  'b': (_FakeTower(1), ProjHead(8, 4))}
        try:
            StreetArm(towers, {'a': g_a, 'b': g_b}, k=5)
        except ValueError:
            pass
        else:
            raise AssertionError('coord mismatch should raise')


def test_streetarm_predict_plumbing():
    """End-to-end retrieve+predict on synthetic galleries (no models)."""
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(7)
        coords = np.stack([rng.uniform(-60, 70, 30),
                           rng.uniform(-170, 170, 30)], 1)
        g_a = _make_gallery(td, n=30, name='a', coords=coords, chunk=7)
        g_b = _make_gallery(td, n=30, name='b', coords=coords, chunk=7)
        arm = _arm_with_galleries({'a': g_a, 'b': g_b})
        pools = arm.retrieve([None, None, None], k=15)
        assert set(pools) == {'a', 'b'}
        for tag, (sims, ids) in pools.items():
            assert sims.shape == (3, 15) and ids.shape == (3, 15)
            assert sims.dtype == torch.float32 and ids.dtype == torch.int64
            assert torch.isfinite(sims).all()
        pred = arm.predict([None, None])
        assert pred.shape == (2, 2)
        assert np.isfinite(pred).all()
        assert np.all(np.abs(pred[:, 0]) <= 90) and np.all(np.abs(pred[:, 1]) <= 180)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'PASS {t.__name__}')
    print(f'{len(tests)} tests passed')


if __name__ == '__main__':
    main()
