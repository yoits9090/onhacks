"""Module definitions matching the chipoint-2 checkpoint schemas.

SR/ThrNet/Blk are transcribed from the repo's run_chipoint.py.
ProjHead/GHead/Router were recovered from state-dict shapes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Blk(nn.Module):
    def __init__(s, d, h, p=0.1):
        super().__init__(); s.h = h; s.dk = d // h
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d); s.dr = nn.Dropout(p)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(s, x):
        B, K, D = x.shape
        q, k, v = s.qkv(s.n1(x)).chunk(3, -1)
        q, k, v = (t.view(B, K, s.h, s.dk).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + s.dr(s.proj(a.transpose(1, 2).reshape(B, K, D)))
        return x + s.ff(s.n2(x))


class SR(nn.Module):
    """Set-ranker: scores each of K candidates from its nf-feature row."""
    def __init__(s, nf, d=256, h=4, l=3, p=0.1):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.LayerNorm(d))
        s.b = nn.ModuleList([Blk(d, h, p) for _ in range(l)])
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, 1)

    def forward(s, x):
        h = s.inp(x)
        for b in s.b:
            h = b(h)
        return s.o(s.n(h)).squeeze(-1)


class ThrNet(nn.Module):
    """Per-threshold confidence head over the same K x nf stack."""
    def __init__(s, nf, d=256, h=4, l=3, p=0.1, n_thr=5, per_bin=True):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.LayerNorm(d))
        s.b = nn.ModuleList([Blk(d, h, p) for _ in range(l)])
        if per_bin:
            s.ns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_thr)])
            s.os = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_thr)])
        else:
            s.n = nn.LayerNorm(d); s.o = nn.Linear(d, n_thr)

    def forward(s, x):
        h = s.inp(x)
        for b in s.b:
            h = b(h)
        if hasattr(s, 'ns'):
            return torch.cat([o(n(h)) for n, o in zip(s.ns, s.os)], -1)
        return s.o(s.n(h))


class ProjHead(nn.Module):
    """tower_dim -> 1024 -> 512 GELU MLP with normalized output, plus the W
    geocell-codebook matrix carried by the checkpoints.

    State dict: net.0 Linear(tower_dim,1024), net.3 Linear(1024,512), W (C,512).
    Tower dims seen: so400m256=1152, dinov3=1024, gopt=1536.
    The demo filters 'W' out at load time and never uses it; geocell_logits
    is kept only for completeness and is UNVERIFIED — it is not part of the
    scored baseline (see EVALUATION.md).
    """
    def __init__(s, tower_dim, n_cells, d=1024, out=512):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(tower_dim, d), nn.GELU(), nn.Dropout(0.0),
                              nn.Linear(d, out))
        s.W = nn.Parameter(torch.empty(n_cells, out))

    def forward(s, x):
        # Demo path: head may run in fp16 on CUDA; normalize the fp32 cast
        # so the 512-d output is always fp32 unit-norm.
        return F.normalize(s.net(x).float(), dim=-1)

    def geocell_logits(s, e):
        # e (B,512) @ W.T (512,C) -> (B,C); previously e @ s.W was a
        # shape error. Unused by the verified pipeline.
        return e @ s.W.T


class GHead(nn.Module):
    """Merged-512 -> 1024 -> (LayerNorm) -> n_classes logits, or 3-dim xyz regression.

    State dict: net.0 Linear(512,1024), net.3 LayerNorm(1024), net.4 Linear(1024,C).
    """
    def __init__(s, n_classes, d=1024, reg=False):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(512, d), nn.GELU(), nn.Dropout(0.0),
                              nn.LayerNorm(d), nn.Linear(d, 3 if reg else n_classes))

    def forward(s, x):
        return s.net(x)


class Router(nn.Module):
    """512 -> 128 -> 1 logit (>0 means photo arm)."""
    def __init__(s, d=512, h=128):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.0), nn.Linear(h, 1))

    def forward(s, x):
        return s.net(x).squeeze(-1)
