"""Load chipoint-2 reranker arms. Run: python run_chipoint.py <ckpt.pt>"""
import sys
import torch
import torch.nn as nn


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
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + s.dr(s.proj(a.transpose(1, 2).reshape(B, K, D)))
        return x + s.ff(s.n2(x))


class SR(nn.Module):
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
    def __init__(s, nf, d=256, h=4, l=3, p=0.1, n_thr=5, per_bin=False):
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


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    nf = ck['nf']
    net = SR(nf)
    net.load_state_dict(ck['state'])
    net.eval()
    print('nf', nf, '| names', len(ck['names']), '| mu/sd', tuple(ck['mu'].shape))
    if 'thr_state' in ck:
        per_bin = any(k.startswith('ns.') for k in ck['thr_state'])
        thr = ThrNet(nf, per_bin=per_bin)
        thr.load_state_dict(ck['thr_state'])
        thr.eval()
        print('thr_km', ck['thr_km'], '| per_bin', per_bin)
    print('OK — feed (Nq,K,nf) features standardized by ck[mu]/ck[sd]')


if __name__ == '__main__':
    main()
