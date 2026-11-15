"""Explicit tower-weight calibration, not the unpublished Chipoint reranker."""
import numpy as np

from .baseline import fuse_pools


def weighted_fuse_pools(pools, tower_weights, keep=64):
    """Weight each tower's population-z-scored top-200 contributions.

    All-one weights take the exact reference path, preserving insertion-order
    tie behavior and fp32 fused scores. Tower spaces remain separate.
    """
    if set(pools) != set(tower_weights):
        raise ValueError("Every selected tower needs exactly one weight.")
    if any(not np.isfinite(w) or w <= 0 for w in tower_weights.values()):
        raise ValueError("Tower weights must be finite and positive.")
    if all(w == 1 for w in tower_weights.values()):
        return fuse_pools(pools, keep=keep)
    # The same validation as the baseline applies before weighted aggregation.
    fuse_pools(pools, keep=keep)
    totals = {}
    for tag, (indices, similarities) in pools.items():
        scores = np.asarray(similarities, dtype=np.float64)
        z = (scores - scores.mean()) / (scores.std() + 1e-9)
        for index, score in zip(indices, z * tower_weights[tag]):
            key = int(index)
            totals[key] = totals.get(key, 0.0) + float(score)
    ids = np.array(sorted(totals, key=lambda key: -totals[key])[:keep], dtype=np.int64)
    return ids, np.array([totals[int(index)] for index in ids], dtype=np.float32)
