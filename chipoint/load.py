"""Checkpoint loaders for chipoint-2 artifacts."""
import numpy as np
import torch

from .nn import GHead, ProjHead, Router, SR, ThrNet

FEATURE_NAMES_23 = [
    'sim_e0', 'margin_e0', 'rank_e0', 'agree_min', 'agree_mean', 'agree_max',
    'agree_spread', 'n_enc', 'vote_bw100', 'vote_bw500', 'vote_bw1492',
    'vote_bw3000', 'dens_25', 'dens_200', 'dens_750', 'dist_centroid',
    'abs_lat', 'cos_lat', 'head_z4', 'head_z6', 'head_z8', 'head_z10',
    'reg_dist',
]


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_reranker(path):
    """Returns (sr, thr, names, mu, sd, thr_km)."""
    ck = _load(path)
    nf = ck['nf']
    sr = SR(nf); sr.load_state_dict(ck['state']); sr.eval()
    thr = None
    if 'thr_state' in ck:
        per_bin = any(k.startswith('ns.') for k in ck['thr_state'])
        thr = ThrNet(nf, n_thr=len(ck['thr_km']), per_bin=per_bin)
        thr.load_state_dict(ck['thr_state']); thr.eval()
    return sr, thr, ck['names'], ck['mu'], ck['sd'], list(ck['thr_km'])


def load_proj_head(path):
    """Returns (head, tower_dim)."""
    ck = _load(path)
    tower_dim = ck['net.0.weight'].shape[1]
    head = ProjHead(tower_dim, ck['W'].shape[0])
    head.load_state_dict(ck, strict=True)
    head.eval()
    return head, tower_dim


def load_ghead(path):
    """Returns (head, mu, sd, n_classes, reg)."""
    ck = _load(path)
    head = GHead(ck['n_classes'], reg=ck['reg'])
    head.load_state_dict(ck['state'])
    head.eval()
    return head, ck['mu'], ck['sd'], ck['n_classes'], ck['reg']


def load_router(path):
    ck = _load(path)
    r = Router()
    r.load_state_dict(ck['state'])
    r.eval()
    return r, ck['mu'], ck['sd']


def load_gallery(emb_path, ll_path=None, device='cuda', dtype=torch.float16):
    """LEGACY — superseded by chipoint.gallery.Gallery.

    Note: this helper L2-renormalizes gallery rows, which the verified
    baseline deliberately does not do (the reference scores raw fp16 rows).
    Do not use it for the scored path; kept for compatibility only.
    Loads fp16 gallery embeddings to GPU. Returns (emb[N,512] on device, ll[N,2] np or None)."""
    emb = np.load(emb_path, mmap_mode='r')
    g = torch.from_numpy(np.ascontiguousarray(emb)).to(device=device, dtype=dtype)
    g = torch.nn.functional.normalize(g.float(), dim=-1).to(dtype)
    ll = np.load(ll_path) if ll_path else None
    return g, ll
