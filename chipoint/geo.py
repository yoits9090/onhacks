"""Geodesy + benchmark metrics for chipoint pipelines."""
import numpy as np
import torch

EARTH_R_KM = 6371.0088
GEOSCALE = 1492.7  # OSV-5M geoscore scale


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or broadcastable arrays."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_km_torch(lat1, lon1, lat2, lon2):
    lat1 = torch.as_tensor(lat1, dtype=torch.float64)
    lon1 = torch.as_tensor(lon1, dtype=torch.float64)
    lat2 = torch.as_tensor(lat2, dtype=torch.float64)
    lon2 = torch.as_tensor(lon2, dtype=torch.float64)
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dp = torch.deg2rad(lat2 - lat1)
    dl = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dp / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * torch.arcsin(torch.sqrt(a.clamp(0, 1)))


def ll_to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def xyz_to_ll(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def quadtree_key(lat, lon, zoom):
    """Quadtree cell id at a given zoom ('z/x/y' web-mercator tiling packed as int)."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05112878, 85.05112878)
    lon = np.asarray(lon, dtype=np.float64)
    n = 1 << zoom
    x = np.floor((lon + 180.0) / 360.0 * n).astype(np.int64)
    lat_r = np.radians(lat)
    y = np.floor((1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / np.pi) / 2.0 * n).astype(np.int64)
    x = np.clip(x, 0, n - 1); y = np.clip(y, 0, n - 1)
    return y * n + x


def metrics(pred_lat, pred_lon, true_lat, true_lon, thr_km=(1, 25, 200, 750, 2500)):
    d = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
    acc = {f"acc@{int(t)}km": float((d <= t).mean() * 100) for t in thr_km}
    return {
        **acc,
        "geoscore": float(np.mean(5000.0 * np.exp(-d / GEOSCALE))),
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
    }
