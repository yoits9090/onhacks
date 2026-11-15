"""Chunked top-K retrieval against fp16 .npy gallery memmaps.

A Gallery pairs an (N,512) float16 projected-embedding memmap with an
(N,2) lat/lon coords memmap, row-aligned. Gallery rows are already
unit-norm projection outputs and are NOT renormalized here — this matches
the reference demo, which scores raw fp16 rows against unit-norm queries.

Scoring streams the memmap in row chunks so peak device memory stays flat
regardless of N: per chunk a (b, chunk_rows) similarity block is computed
and locally top-k'd, then chunk candidates are merged into the global
top-k. CPU uses float32 matmul; CUDA uses float16 (demo precision).
"""
import warnings

import numpy as np
import torch

GALLERY_DIM = 512  # projected retrieval-space width (nn.ProjHead output)


class Gallery:
    def __init__(self, emb_path, coords_path, device='cuda', chunk=65536, batch=8):
        """emb_path: (N,512) fp16 npy memmap. coords_path: (N,2) lat/lon npy
        with the same row count. chunk: gallery rows per device matmul.
        batch: query rows processed per pass."""
        emb = np.load(emb_path, mmap_mode='r')
        coords = np.load(coords_path, mmap_mode='r')
        if emb.ndim != 2 or emb.shape[1] != GALLERY_DIM or emb.dtype != np.float16:
            raise ValueError(
                f'{emb_path}: expected (N,{GALLERY_DIM}) float16 gallery, '
                f'got shape={emb.shape} dtype={emb.dtype}')
        if (coords.ndim != 2 or coords.shape[1] != 2
                or coords.shape[0] != emb.shape[0]):
            raise ValueError(
                f'{coords_path}: expected ({emb.shape[0]},2) coords row-aligned '
                f'with the gallery, got shape={coords.shape}')
        if int(chunk) < 1 or int(batch) < 1:
            raise ValueError(f'chunk and batch must be >= 1, got {chunk}/{batch}')
        self.emb_path = str(emb_path)
        self.coords_path = str(coords_path)
        self._emb = emb
        self.coords = coords
        self.device = torch.device(device)
        self.n = int(emb.shape[0])
        self.dim = int(emb.shape[1])
        self.chunk = int(chunk)
        self.batch = int(batch)

    @torch.inference_mode()
    def topk(self, q_emb, k=200):
        """q_emb: (B,512) unit-norm torch/array -> (sims, ids) on CPU.

        sims: (B,k) float32, ids: (B,k) int64 — both sorted by descending
        similarity. k must satisfy 1 <= k <= N.
        """
        q = torch.as_tensor(q_emb)
        if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != self.dim:
            raise ValueError(
                f'q_emb must be a nonempty (B,{self.dim}) array, got {tuple(q.shape)}')
        if not torch.isfinite(q).all():
            raise RuntimeError('q_emb contains non-finite values')
        k = int(k)
        if not 1 <= k <= self.n:
            raise ValueError(f'k must be in [1, {self.n}], got {k}')
        # CUDA: fp16 matmul like the demo; CPU: float32.
        compute = torch.float16 if self.device.type == 'cuda' else torch.float32
        out_sims, out_ids = [], []
        for i in range(0, q.shape[0], self.batch):
            qb = q[i:i + self.batch].to(self.device, compute)
            cand_s, cand_i = [], []
            for start in range(0, self.n, self.chunk):
                block = self._emb[start:start + self.chunk]
                if not block.flags.c_contiguous:
                    block = np.ascontiguousarray(block)
                with warnings.catch_warnings():
                    # memmaps are read-only; we only read, never write
                    warnings.simplefilter('ignore', UserWarning)
                    gb = torch.from_numpy(block)
                gb = gb.to(self.device)
                if compute == torch.float32:
                    gb = gb.float()
                sims = (qb @ gb.T).float()  # (b, rows) fp32 for stable merge
                kk = min(k, sims.shape[1])
                v, ix = sims.topk(kk, dim=1)
                cand_s.append(v)
                cand_i.append(ix + start)  # global gallery row
            cs = torch.cat(cand_s, dim=1)
            ci = torch.cat(cand_i, dim=1)
            v, ix = cs.topk(k, dim=1)
            out_sims.append(v.cpu())
            out_ids.append(ci.gather(1, ix).cpu())
        return torch.cat(out_sims, dim=0), torch.cat(out_ids, dim=0)


def fused_topk(tower_pools, k=512):
    """Reciprocal-rank fusion of multiple towers' top-K pools.

    tower_pools: list of (sims (B,kt), idx (B,kt)). Returns merged
    (idx (B,k), fused_score (B,k)) deduplicated per query.

    NOTE: legacy helper kept for compatibility; the verified baseline uses
    chipoint.baseline.fuse_pools instead (per EVALUATION.md).
    """
    B = tower_pools[0][0].shape[0]
    out_idx, out_score = [], []
    for b in range(B):
        score = {}
        for sims, idx in tower_pools:
            for r, (s, i) in enumerate(zip(sims[b].tolist(), idx[b].tolist())):
                score[i] = score.get(i, 0.0) + 1.0 / (60.0 + r) + 0.001 * s
        items = sorted(score.items(), key=lambda kv: -kv[1])[:k]
        out_idx.append([i for i, _ in items])
        out_score.append([s for _, s in items])
    return (torch.tensor(out_idx, dtype=torch.long),
            torch.tensor(out_score, dtype=torch.float32))
