"""Candidate-feature stack for the unpublished 23-feature reranker.

DISABLED: the exact training-time feature contract was never published
(see upstream/demo_app.py ENABLE_RERANKER note and EVALUATION.md), so the
public entry point raises NotImplementedError. The verified path is
StreetArm.retrieve/predict -> baseline.fuse_pools/consensus. The legacy
implementation is kept below as _build_features_legacy for reference only.
"""
import numpy as np
import torch

from .geo import haversine_km, ll_to_xyz, quadtree_key, xyz_to_ll

VOTE_BWS = (100.0, 500.0, 1492.0, 3000.0)
DENS_RS = (25.0, 200.0, 750.0)


def _rankmap(idx_row):
    return {int(i): r for r, i in enumerate(idx_row)}


def build_features(sims, cand_idx, cand_ll, other_pools=None,
                   ghead_logps=None, q_reg_ll=None, extra=False):
    """Disabled — raises NotImplementedError (see module docstring)."""
    raise NotImplementedError(
        'build_features is disabled: the 23-feature reranker contract was '
        'never published and this guessed stack is not safe to score. Use '
        'StreetArm.predict (baseline.fuse_pools + baseline.consensus).')


def _build_features_legacy(sims, cand_idx, cand_ll, other_pools=None,
                           ghead_logps=None, q_reg_ll=None, extra=False):
    """One query's candidate pool -> (K, nf) float32 array.

    sims: (K,) float, sorted desc. cand_idx: (K,) gallery rows.
    cand_ll: (K,2) lat/lon.
    other_pools: list of (idx_row (Kt,),) from additional towers, for agree_*.
    ghead_logps: dict {zoom: (n_classes,) log-probs} for THIS query.
    q_reg_ll: (2,) regression-head prediction for THIS query.
    """
    sims = np.asarray(sims, dtype=np.float64)
    K = len(sims)
    lat, lon = cand_ll[:, 0], cand_ll[:, 1]

    rank_norm = np.arange(K) / max(K - 1, 1)
    margin = np.empty(K)
    margin[:-1] = sims[:-1] - sims[1:]
    margin[-1] = 0.0

    # cross-encoder agreement: per-tower normalized rank (absent -> 0)
    pres = [1.0 - rank_norm]
    n_enc = np.ones(K)
    if other_pools:
        n_enc = np.ones(K)
        for idx_row in other_pools:
            rm = _rankmap(idx_row)
            kt = max(len(idx_row) - 1, 1)
            pos = np.array([1.0 - rm[int(i)] / kt if int(i) in rm else 0.0
                            for i in cand_idx])
            pres.append(pos)
            n_enc += (pos > 0)
    P = np.stack(pres, 1)
    agree_min, agree_max, agree_mean = P.min(1), P.max(1), P.mean(1)

    # pairwise candidate distances for votes/density
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    votes = [((D <= b) * sims[None, :]).sum(1) for b in VOTE_BWS]
    dens = [(D <= r).mean(1) for r in DENS_RS]

    w = np.exp(sims - sims.max()); w /= w.sum()
    cent = w @ ll_to_xyz(lat, lon)
    clat, clon = xyz_to_ll(cent)
    dist_cent = haversine_km(lat, lon, clat, clon)

    heads = []
    for z in (4, 6, 8, 10):
        if ghead_logps and z in ghead_logps:
            lp = ghead_logps[z]
            cells = quadtree_key(lat, lon, z)
            heads.append(lp[np.clip(cells, 0, len(lp) - 1)])
        else:
            heads.append(np.zeros(K))
    reg_dist = (haversine_km(q_reg_ll[0], q_reg_ll[1], lat, lon)
                if q_reg_ll is not None else np.zeros(K))

    feats = np.stack([
        sims, margin, rank_norm, agree_min, agree_mean, agree_max,
        agree_max - agree_min, n_enc, *votes, *dens,
        dist_cent, np.abs(lat), np.cos(np.radians(lat)),
        *heads, reg_dist,
    ], axis=1).astype(np.float32)

    if extra:
        top1_lat, top1_lon = lat[0], lon[0]
        extra_cols = np.stack([
            sims - sims[0],
            np.log(np.maximum(sims - sims.min(), 1e-9)),
            haversine_km(lat, lon, top1_lat, top1_lon),
            np.sqrt(np.maximum(votes[1], 0)),
            np.array([(D[i] <= 25.0).sum() * sims[i] for i in range(K)]),
        ], axis=1)
        feats = np.concatenate([feats, extra_cols.astype(np.float32)], axis=1)
    return feats
