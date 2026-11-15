"""Narrow tests for fetch_assets verification logic (mocked hub, no network).

Run with pytest or directly:  python tests/test_fetch_assets.py
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fetch_assets as fa  # noqa: E402


def _meta_entry(payload: bytes):
    return {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def _sibling(rfilename, payload=None, sha256=None, size=None):
    lfs = None
    if sha256 is not None or size is not None:
        lfs = SimpleNamespace(sha256=sha256, size=size, oid=sha256)
    return SimpleNamespace(rfilename=rfilename, size=size, lfs=lfs)


class TestHfMetadata(unittest.TestCase):
    def test_metadata_exception_is_hard_error(self):
        with mock.patch('huggingface_hub.HfApi') as api:
            api.return_value.model_info.side_effect = RuntimeError('boom')
            with self.assertRaises(RuntimeError):
                fa._hf_metadata()

    def test_metadata_collects_lfs_hashes(self):
        payload = b'x' * 16
        sib = _sibling('weights/proj_head_dinov3.pt',
                       sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
        info = SimpleNamespace(siblings=[sib])
        with mock.patch('huggingface_hub.HfApi') as api:
            api.return_value.model_info.return_value = info
            meta = fa._hf_metadata()
        self.assertEqual(meta['weights/proj_head_dinov3.pt']['sha256'],
                         hashlib.sha256(payload).hexdigest())
        self.assertEqual(meta['weights/proj_head_dinov3.pt']['size'], len(payload))


class TestFetchHfVerification(unittest.TestCase):
    REL = 'weights/proj_head_dinov3.pt'

    def test_missing_hash_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta = {self.REL: {'size': 10}}  # no sha256
            with mock.patch('huggingface_hub.hf_hub_download') as dl:
                with self.assertRaises(RuntimeError):
                    fa._fetch_hf(self.REL, root, meta, dry_run=False)
            dl.assert_not_called()
            self.assertFalse((root / self.REL).exists())

    def test_preexisting_wrong_bytes_rejected_without_hub_call(self):
        payload = b'good-bytes'
        wrong = b'bad-bytes!'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dest = root / self.REL
            dest.parent.mkdir(parents=True)
            dest.write_bytes(wrong)
            meta = {self.REL: _meta_entry(payload)}
            with mock.patch('huggingface_hub.hf_hub_download') as dl:
                with self.assertRaises(RuntimeError):
                    fa._fetch_hf(self.REL, root, meta, dry_run=False)
            dl.assert_not_called()
            self.assertEqual(dest.read_bytes(), wrong)  # untouched

    def test_valid_existing_bytes_skips_hub(self):
        payload = b'good-bytes'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dest = root / self.REL
            dest.parent.mkdir(parents=True)
            dest.write_bytes(payload)
            meta = {self.REL: _meta_entry(payload)}
            with mock.patch('huggingface_hub.hf_hub_download') as dl:
                fa._fetch_hf(self.REL, root, meta, dry_run=False)
            dl.assert_not_called()
            self.assertEqual(dest.read_bytes(), payload)

    def test_absent_file_downloads_then_verifies(self):
        payload = b'downloaded-content'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = Path(td) / 'cache_file'
            src.write_bytes(payload)
            meta = {self.REL: _meta_entry(payload)}
            with mock.patch('huggingface_hub.hf_hub_download',
                            return_value=str(src)) as dl:
                fa._fetch_hf(self.REL, root, meta, dry_run=False)
            dl.assert_called_once()

    def test_absent_file_bad_download_raises(self):
        good = b'expected'
        bad = b'corrupt!!'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = Path(td) / 'cache_file'
            src.write_bytes(bad)
            meta = {self.REL: _meta_entry(good)}
            with mock.patch('huggingface_hub.hf_hub_download',
                            return_value=str(src)):
                with self.assertRaises(RuntimeError):
                    fa._fetch_hf(self.REL, root, meta, dry_run=False)


class TestVerifyStrict(unittest.TestCase):
    def test_verify_requires_size_and_sha(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'f.bin'
            p.write_bytes(b'abc')
            for size, sha in [(None, 'x'), (3, None), (0, 'x'), (-1, 'x'), (3, '')]:
                with self.assertRaises(RuntimeError, msg=f'{size}/{sha}'):
                    fa._verify(p, size, sha, 'lbl')


def main():
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
