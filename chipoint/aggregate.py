"""Collapse a reranked/scored candidate pool into a single lat/lon."""
import numpy as np

from .geo import ll_to_xyz, xyz_to_ll


def predict_point(cand_ll, scores, mode='argmax', tau=0.02, top_m=8,
                  radius_km=200.0):
    """cand_ll (K,2), scores (K,) -> (lat, lon).

    modes:
      argmax   — top candidate's coords
      wcent    — softmax(scores/tau) weighted sphere-centroid of top_m
      density  — weighted centroid restricted to top candidate's radius_km hood
    """
    cand_ll = np.asarray(cand_ll, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if mode == 'argmax':
        return tuple(cand_ll[int(np.argmax(scores))])
    m = min(top_m, len(scores))
    order = np.argsort(-scores)[:m]
    s = scores[order]
    w = np.exp((s - s.max()) / max(tau, 1e-9)); w /= w.sum()
    if mode == 'wcent':
        lat, lon = xyz_to_ll(w @ ll_to_xyz(cand_ll[order, 0], cand_ll[order, 1]))
        return float(lat), float(lon)
    if mode == 'density':
        from .geo import haversine_km
        tlat, tlon = cand_ll[order[0]]
        near = haversine_km(cand_ll[order, 0], cand_ll[order, 1], tlat, tlon) <= radius_km
        w = w * near
        if w.sum() < 1e-9:
            return float(tlat), float(tlon)
        w /= w.sum()
        lat, lon = xyz_to_ll(w @ ll_to_xyz(cand_ll[order, 0], cand_ll[order, 1]))
        return float(lat), float(lon)
    raise ValueError(mode)
