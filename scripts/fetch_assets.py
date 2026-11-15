#!/usr/bin/env python3
"""Fetch the pinned chipoint-2 street-arm assets.

Authoritative manifest — do not edit casually:

  HF model   : chiikabu-labs/chipoint-2
               @ a150120cd968d48310d092caf26854f109c90b51
  DINO mirror: ModelScope facebook/dinov3-vitl16-pretrain-lvd1689m
               @ bab39f7f4023c45534a5326e9378edc5b2016e48
               (ungated mirror of gated facebook/dinov3-vitl16-pretrain-lvd1689m)

Fetches ONLY:
  - weights/proj_head_{tower}.pt                      (per --towers)
  - street_gallery/gallery_all_C_so400m256.npy        (shared coords, --galleries)
  - street_gallery/proj_head_gallery_512_{tower}.npy  (--galleries)
  - encoders/dinov3-vitl16-pretrain-lvd1689m/{config.json, preprocessor_config.json,
    LICENSE.md, README.md}                            (always; tiny, sha256-verified)
  - encoders/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
    (only with --dinov3-weights; ~1.2 GB, size+sha256 verified via .part+rename)

NOT fetched: ghead_*, router_*, streetview_k512_joint.pt, photo_gallery/*,
features/*, benchmark images/datasets. OpenCLIP tower weights are NOT fetched
here — Tower loads them lazily via open_clip pretrained='webli' at runtime;
their content is not hash-pinned by this script (the library resolves the
checkpoint itself).

HF files go through huggingface_hub at the pinned revision. Verification is
mandatory, not best-effort: Hub metadata must be fetched successfully and
every requested file must carry BOTH an integer size>0 and an LFS sha256 in
that metadata, or the run stops. An existing destination is verified BEFORE
huggingface_hub is invoked — exact match skips, anything else raises without
overwriting. No tokens are embedded; HF_TOKEN in the environment is used
only if the public repo ever requires it.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HF_REPO = 'chiikabu-labs/chipoint-2'
HF_REVISION = 'a150120cd968d48310d092caf26854f109c90b51'

MODELSCOPE_REPO = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
MODELSCOPE_REV = 'bab39f7f4023c45534a5326e9378edc5b2016e48'
MODELSCOPE_BASE = (f'https://modelscope.cn/models/{MODELSCOPE_REPO}'
                   f'/resolve/{MODELSCOPE_REV}')

DINO_DIR = 'encoders/dinov3-vitl16-pretrain-lvd1689m'
DINO_WEIGHTS_NAME = 'model.safetensors'
# Lead-verified authoritative values (matches HF metadata).
DINO_WEIGHTS_SIZE = 1212559808
DINO_WEIGHTS_SHA256 = 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
# sha256 of the small files as served at MODELSCOPE_REV (recorded in
# encoders/.../provenance.json).
DINO_SMALL_SHA256 = {
    'config.json': '135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5',
    'preprocessor_config.json': '960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7',
    'LICENSE.md': '1e22708c509c2253ebf44f54a080e8c60708f59037288635cd5d5220fe87c418',
    'README.md': 'f9ea6e22356d0040d53c7ead60c7c48719f8c094a1a2fede9a6e6c79aee5331f',
}
DINO_SMALL_FILES = list(DINO_SMALL_SHA256)

STREET_COORDS = 'street_gallery/gallery_all_C_so400m256.npy'
TOWERS = ('so400m256', 'gopt', 'dinov3')


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _hf_metadata():
    """{repo_path: {'sha256','size'}} from Hub metadata at the pinned rev.

    Hard error on failure — verification must never silently degrade.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(HF_REPO, revision=HF_REVISION,
                                  files_metadata=True)
    except Exception as e:
        raise RuntimeError(
            f'Hub metadata fetch failed for {HF_REPO}@{HF_REVISION}; '
            'cannot verify downloads without it') from e
    meta = {}
    for s in getattr(info, 'siblings', None) or []:
        entry = {}
        lfs = getattr(s, 'lfs', None)
        if lfs is not None:
            # BlobLfsInfo exposes sha256 (newer hub) / oid (older) for the
            # content hash; size is the real byte count.
            entry['sha256'] = (getattr(lfs, 'sha256', None)
                               or getattr(lfs, 'oid', None))
            entry['size'] = getattr(lfs, 'size', None)
        if entry.get('size') is None:
            entry['size'] = getattr(s, 'size', None)
        meta[s.rfilename] = entry
    return meta


def _verify(path, expected_size, expected_sha256, label):
    """Strict verify: both authoritative values are required, file must
    match exactly. Raises on any missing expectation or mismatch."""
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) \
            or expected_size <= 0 or not expected_sha256:
        raise RuntimeError(
            f'{label}: authoritative size>0/sha256 missing from Hub metadata; '
            'refusing unverifiable fetch')
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'{label}: expected file missing at {path}')
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(
            f'{label}: size {size} != expected {expected_size} ({path})')
    got = _sha256(path)
    if got != expected_sha256:
        raise RuntimeError(
            f'{label}: sha256 {got} != expected {expected_sha256} ({path})')
    print(f'[verified] {label}: {size} bytes sha256={got}', flush=True)


def _part_path(dest):
    return Path(dest).with_name(Path(dest).name + '.part')


def _curl_download(url, dest):
    """curl -fL with retries and resume into dest.part; returns the .part path."""
    dest = Path(dest)
    tmp = _part_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['curl', '-fL', '--retry', '5', '--retry-all-errors',
           '--retry-delay', '3', '--connect-timeout', '30',
           '-C', '-', '-o', str(tmp), url]
    print(f'[fetch] {url}\n       -> {tmp}', flush=True)
    subprocess.run(cmd, check=True)
    return tmp


def _completed_part(tmp, expected_sha256, expected_size=None):
    """True if a leftover .part is already complete — rename it instead of
    re-downloading (curl -C - cannot resume a finished file)."""
    tmp = Path(tmp)
    if not tmp.is_file():
        return False
    if expected_size is not None and tmp.stat().st_size != expected_size:
        return False
    return _sha256(tmp) == expected_sha256


def _fetch_hf(rel, root, meta, dry_run):
    dest = root / rel
    if dry_run:
        state = 'present' if dest.exists() else 'missing'
        print(f'[dry-run] HF {HF_REPO}@{HF_REVISION[:12]} {rel} -> {dest} [{state}]')
        return
    entry = meta.get(rel) or {}
    exp_size = entry.get('size')
    exp_sha = entry.get('sha256')
    # Authoritative expectations must exist BEFORE any hub interaction —
    # missing size>0/sha256 is a hard stop, never a download-then-fail.
    if not isinstance(exp_size, int) or isinstance(exp_size, bool) \
            or exp_size <= 0 or not exp_sha:
        raise RuntimeError(
            f'{HF_REPO}:{rel}: authoritative size>0/sha256 missing from '
            'Hub metadata; refusing unverifiable fetch')
    # Existing destination is verified BEFORE any hub call: exact match
    # skips the download; anything else raises without overwriting.
    if dest.exists():
        _verify(dest, exp_size, exp_sha, f'{HF_REPO}:{rel}')
        print(f'[skip] {rel} present and verified', flush=True)
        return
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(HF_REPO, rel, revision=HF_REVISION,
                                local_dir=str(root)))
    if not path.is_file():  # defensive: local_dir mode should return root/rel
        path = dest
    _verify(path, exp_size, exp_sha, f'{HF_REPO}:{rel}')


def _fetch_ms_small(name, root, dry_run):
    dest = root / DINO_DIR / name
    expected = DINO_SMALL_SHA256[name]
    if dry_run:
        state = 'present' if dest.exists() else 'missing'
        print(f'[dry-run] ModelScope {MODELSCOPE_REPO}@{MODELSCOPE_REV[:12]} '
              f'{name} -> {dest} [{state}]')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        got = _sha256(dest)
        if got == expected:
            print(f'[skip] {name} present and sha256-verified', flush=True)
            return
        raise RuntimeError(
            f'{dest} exists but sha256 {got} != expected {expected}; '
            'refusing to overwrite — resolve manually')
    tmp = _part_path(dest)
    if _completed_part(tmp, expected):
        tmp.replace(dest)
        print(f'[verified] {name}: sha256 {expected} (completed .part)', flush=True)
        return
    tmp = _curl_download(f'{MODELSCOPE_BASE}/{name}', dest)
    got = _sha256(tmp)
    if got != expected:
        raise RuntimeError(
            f'{name}: sha256 {got} != expected {expected}; kept {tmp} for inspection')
    tmp.replace(dest)
    print(f'[verified] {name}: sha256 {got}', flush=True)


def _fetch_dino_weights(root, dry_run):
    dest = root / DINO_DIR / DINO_WEIGHTS_NAME
    if dry_run:
        state = 'present' if dest.exists() else 'missing'
        print(f'[dry-run] ModelScope {MODELSCOPE_REPO}@{MODELSCOPE_REV[:12]} '
              f'{DINO_WEIGHTS_NAME} -> {dest} [{state}] '
              f'(~1.2GB; size+sha256 verified before rename)')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if (dest.stat().st_size == DINO_WEIGHTS_SIZE
                and _sha256(dest) == DINO_WEIGHTS_SHA256):
            print('[skip] model.safetensors present and verified', flush=True)
            return
        raise RuntimeError(
            f'{dest} exists but fails size/sha256 verification; '
            'refusing to overwrite — resolve manually')
    tmp = _part_path(dest)
    if _completed_part(tmp, DINO_WEIGHTS_SHA256, DINO_WEIGHTS_SIZE):
        tmp.replace(dest)
        print(f'[verified] {DINO_WEIGHTS_NAME}: {DINO_WEIGHTS_SIZE} bytes '
              'sha256 (completed .part)', flush=True)
        return
    tmp = _curl_download(f'{MODELSCOPE_BASE}/{DINO_WEIGHTS_NAME}', dest)
    size = tmp.stat().st_size
    if size != DINO_WEIGHTS_SIZE:
        raise RuntimeError(
            f'{DINO_WEIGHTS_NAME}: size {size} != expected {DINO_WEIGHTS_SIZE}; '
            f'kept {tmp} for inspection/resume')
    got = _sha256(tmp)
    if got != DINO_WEIGHTS_SHA256:
        raise RuntimeError(
            f'{DINO_WEIGHTS_NAME}: sha256 {got} != expected {DINO_WEIGHTS_SHA256}; '
            f'kept {tmp} for inspection')
    tmp.replace(dest)
    print(f'[verified] {DINO_WEIGHTS_NAME}: {size} bytes sha256={got}', flush=True)


def _plan_hf_files(towers, galleries):
    files = []
    for t in towers:
        files.append(f'weights/proj_head_{t}.pt')
    if galleries:
        files.append(STREET_COORDS)
        for t in towers:
            files.append(f'street_gallery/proj_head_gallery_512_{t}.npy')
    return files


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Fetch pinned chipoint-2 street-arm assets (see docstring).')
    ap.add_argument('--root', type=Path, default=None,
                    help='project root (default: parent of scripts/)')
    ap.add_argument('--galleries', action='store_true',
                    help='also fetch the shared street coords and per-tower '
                         '512-d street galleries (~5 GB each)')
    ap.add_argument('--dinov3-weights', action='store_true',
                    help='also fetch the 1.2 GB DINOv3 model.safetensors '
                         '(size+sha256 verified)')
    ap.add_argument('--towers', nargs='+', choices=TOWERS, default=list(TOWERS),
                    help='towers to fetch assets for (default: all three)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the fetch plan without downloading anything')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.towers = list(dict.fromkeys(args.towers))  # dedupe, keep order
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    print(f'[fetch] root={root}')
    print(f'[fetch] towers={args.towers} galleries={args.galleries} '
          f'dinov3_weights={args.dinov3_weights} dry_run={args.dry_run}')

    hf_files = _plan_hf_files(args.towers, args.galleries)
    meta = {} if args.dry_run else _hf_metadata()
    for rel in hf_files:
        _fetch_hf(rel, root, meta, args.dry_run)

    for name in DINO_SMALL_FILES:
        _fetch_ms_small(name, root, args.dry_run)

    if args.dinov3_weights:
        _fetch_dino_weights(root, args.dry_run)

    print('[fetch] done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
