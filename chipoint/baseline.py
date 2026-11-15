"""Street-view demo baseline, without the unpublished 23-feature reranker.

Reference: chiikabu-labs/chipoint-v2-geolocation, revision
65dd8df82f36608ad819f5ecd4371da16f73704e, app.py.
"""
import numpy as np


def fuse_pools(pools, keep=64):
    """Fuse {tower: (row_indices, cosine_scores)} for a single query.

    Each tower must supply the same retrieval depth, in descending score order.
    The baseline uses depth 200, equal tower weights, and keeps 64 candidates.
    """
    if not pools or keep < 1:
        raise ValueError("Provide at least one tower and a positive keep value.")
    totals = {}
    depth = None
    for indices, similarities in pools.values():
        indices = np.asarray(indices)
        similarities = np.asarray(similarities, dtype=np.float64)
        if indices.ndim != 1 or similarities.shape != indices.shape or not len(indices):
            raise ValueError("Each pool needs matching, nonempty 1-D indices and scores.")
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
            raise ValueError("Gallery row indices must be nonnegative integers.")
        if len(np.unique(indices)) != len(indices) or not np.isfinite(similarities).all():
            raise ValueError("Duplicate indices or nonfinite retrieval scores.")
        if np.any(np.diff(similarities) > 0):
            raise ValueError("Retrieval scores must be in descending order.")
        if depth is not None and len(indices) != depth:
            raise ValueError("All towers must use the same retrieval depth.")
        depth = len(indices)
        z_scores = (similarities - similarities.mean()) / (similarities.std() + 1e-9)
        for index, score in zip(indices, z_scores):
            key = int(index)
            totals[key] = totals.get(key, 0.0) + float(score)
    # Stable insertion-order ties match the reference Python dictionary/sort.
    indices = np.array(sorted(totals, key=lambda key: -totals[key])[:keep], dtype=np.int64)
    return indices, np.array([totals[int(index)] for index in indices], dtype=np.float32)


def consensus(coords, scores, top_n=10):
    """Match the demo's default top-10, 250/500/1000 km cluster consensus.

    Input candidates are already sorted by fused score. Returns (lat, lon).
    This is an inference baseline, not the published full-reranker pipeline.
    """
    coords = np.asarray(coords, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (coords.ndim != 2 or coords.shape[1] != 2 or
            scores.shape != (len(coords),) or not len(coords) or top_n < 1):
        raise ValueError("Expected nonempty (K,2) coordinates and (K,) scores.")
    if not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError("Coordinates and scores must be finite.")
    if np.any(np.abs(coords[:, 0]) > 90) or np.any(np.abs(coords[:, 1]) > 180):
        raise ValueError("Coordinates must be latitude/longitude in degrees.")
    coords, scores = coords[:top_n], scores[:top_n]
    latitude, longitude = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    a = (np.sin((latitude - latitude[0]) / 2) ** 2 +
         np.cos(latitude[0]) * np.cos(latitude) *
         np.sin((longitude - longitude[0]) / 2) ** 2)
    distances = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    for radius in (250.0, 500.0, 1000.0):
        members = np.flatnonzero(distances <= radius)
        if len(members) >= 3:
            break
    weights = np.clip(scores[members], 0, None) ** 4
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    la, lo = latitude[members], longitude[members]
    x = np.sum(weights * np.cos(la) * np.cos(lo))
    y = np.sum(weights * np.cos(la) * np.sin(lo))
    z = np.sum(weights * np.sin(la))
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))
