"""Street-view arm: per-tower retrieval fused by the demo baseline.

Reference: upstream/demo_app.py street arm @
65dd8df82f36608ad819f5ecd4371da16f73704e, scored by chipoint.baseline
(fuse_pools + consensus). No router, photo arm, ghead, or reranker paths —
the 23-feature contract is unpublished and intentionally unused.
"""
import numpy as np
import torch

from . import baseline
from .calibration import weighted_fuse_pools


class StreetArm:
    """towers: {tag: (Tower, ProjHead)}; galleries: {tag: Gallery}.

    Both dicts must have identical keys — every selected tower retrieves
    against its own gallery and a missing/extra entry is an error, never a
    silent drop. All galleries must share the same row count and identical
    coordinates (the demo indexes every tower's gallery with one shared
    street coord file); mismatches raise at construction.
    """

    def __init__(self, towers, galleries, k=200):
        if not towers:
            raise ValueError('StreetArm requires at least one tower')
        if set(towers) != set(galleries):
            raise ValueError(
                f'tower/gallery key mismatch: towers={sorted(towers)} '
                f'galleries={sorted(galleries)}')
        self.towers = dict(towers)
        self.galleries = dict(galleries)
        self.k = int(k)
        row_counts = {g.n for g in self.galleries.values()}
        if len(row_counts) != 1:
            raise ValueError(f'gallery row-count mismatch across towers: {sorted(row_counts)}')
        self.n = row_counts.pop()
        tags = list(self.galleries)
        ref = np.asarray(self.galleries[tags[0]].coords)
        for tag in tags[1:]:
            other = np.asarray(self.galleries[tag].coords)
            if other.shape != ref.shape or not np.array_equal(other, ref):
                raise ValueError(
                    f'gallery {tag!r} coordinates differ from {tags[0]!r}; '
                    'fused row indices would not be row-aligned')
        self.coords = ref  # shared (N,2) lat/lon for fused indices
        for tower, head in self.towers.values():
            # Demo runs proj heads in fp16 on CUDA, fp32 on CPU.
            dtype = torch.float16 if tower.device.type == 'cuda' else torch.float32
            head.to(device=tower.device, dtype=dtype).eval()

    @torch.inference_mode()
    def retrieve(self, images, k=None):
        """images: list[PIL.Image] -> {tag: (sims, ids)}.

        sims: (B,k) float32 CPU, ids: (B,k) int64 CPU, sorted descending —
        each tower searched only against its own gallery.
        """
        k = self.k if k is None else int(k)
        if not 1 <= k <= self.n:
            raise ValueError(f'k must be in [1, {self.n}], got {k}')
        pools = {}
        for tag, (tower, head) in self.towers.items():
            e = tower.encode(images)          # (B, tower_dim) fp32 unit-norm
            p0 = next(head.parameters())
            # Demo casts the raw embedding to the head's device/dtype
            # (fp16 on CUDA) before the MLP.
            e = e.to(device=p0.device, dtype=p0.dtype)
            p = head(e)                       # (B,512) fp32 unit-norm
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f'tower {tag!r}: projection head produced non-finite output')
            pools[tag] = self.galleries[tag].topk(p, k=k)
        return pools

    def predict(self, images, *, top_n=10, tower_weights=None):
        """images: list[PIL.Image] -> (B,2) float64 array of (lat, lon).

        Per image: fuse each tower's top-k ids/scores — baseline.fuse_pools
        (equal weights) unless tower_weights={tag: float} selects
        calibration.weighted_fuse_pools — then baseline.consensus over the
        shared coords with the given top_n (10 = published demo baseline).
        Output is exactly as meaningful as the gallery.
        """
        pools = self.retrieve(images, k=self.k)
        batch = next(iter(pools.values()))[0].shape[0]
        out = np.empty((batch, 2), dtype=np.float64)
        for b in range(batch):
            fused_in = {
                tag: (ids[b].numpy(), sims[b].numpy())
                for tag, (sims, ids) in pools.items()
            }
            if tower_weights is None:
                fused_ids, fused_scores = baseline.fuse_pools(fused_in)
            else:
                fused_ids, fused_scores = weighted_fuse_pools(
                    fused_in, tower_weights)
            out[b] = baseline.consensus(self.coords[fused_ids], fused_scores,
                                        top_n=top_n)
        return out
