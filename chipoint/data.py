"""OSV-5M zip extraction + query sampling."""
import csv
import io
import os
import zipfile

import numpy as np


def load_metadata(csv_path):
    """Returns {id_str: (lat, lon)}."""
    meta = {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            meta[r['id']] = (float(r['latitude']), float(r['longitude']))
    return meta


def iter_zip_images(zip_path, meta=None, limit=None):
    """Yields (image_id, bytes, lat, lon). Skips rows missing metadata."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            iid = os.path.splitext(os.path.basename(name))[0]
            ll = meta.get(iid) if meta else None
            if meta is not None and ll is None:
                continue
            yield iid, zf.read(name), (ll[0] if ll else None), (ll[1] if ll else None)
            n += 1
            if limit and n >= limit:
                return


def sample_queries(zip_paths, meta, n, seed=0, limit_per_zip=None):
    """Collect up to n (id, png_bytes, lat, lon) sampled across zips."""
    rng = np.random.default_rng(seed)
    items = []
    for zp in zip_paths:
        items.extend(iter_zip_images(zp, meta, limit_per_zip))
    idx = rng.permutation(len(items))[:n]
    return [items[i] for i in idx]
