"""Vision tower wrappers matching the chipoint-v2 demo street arm.

Each tower produces an L2-normalized fp32 embedding of its native width
(so400m256=1152, gopt=1536, dinov3=1024). Each tower's ProjHead maps its
embedding into that tower's own 512-d projection space — the spaces are
NOT shared across towers (each tower retrieves against its own gallery).

Reference: upstream/demo_app.py, Space chiikabu-labs/chipoint-v2-geolocation
@ 65dd8df82f36608ad819f5ecd4371da16f73704e (TOWER_REGISTRY, _encode_tower,
_make_dinov3_preprocess). The demo's photo arm, router, ghead and reranker
paths are intentionally not reproduced here (see EVALUATION.md).
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Demo TOWER_REGISTRY, street arm only. open_clip weights are fetched lazily
# by open_clip itself (pretrained='webli'); their content is not hash-pinned
# by this repo — the library resolves the checkpoint at load time.
TOWER_SPECS = {
    'so400m256': {'kind': 'open_clip', 'open_clip_name': 'ViT-SO400M-16-SigLIP2-256',
                  'pretrained': 'webli', 'dim': 1152},
    'gopt': {'kind': 'open_clip', 'open_clip_name': 'ViT-gopt-16-SigLIP2-256',
             'pretrained': 'webli', 'dim': 1536},
    'dinov3': {'kind': 'dinov3', 'dim': 1024},
}

# Local ungated mirror of facebook/dinov3-vitl16-pretrain-lvd1689m
# (ModelScope rev bab39f7f4023c45534a5326e9378edc5b2016e48; see
# encoders/dinov3-vitl16-pretrain-lvd1689m/provenance.json).
DINOV3_DIR = 'encoders/dinov3-vitl16-pretrain-lvd1689m'


class Tower:
    """Frozen image encoder.

    encode(list[PIL.Image]) -> (B, dim) fp32 L2-normalized embedding.
    Non-finite outputs raise instead of silently propagating.
    """

    def __init__(self, name, device='cuda', root=None):
        if name not in TOWER_SPECS:
            raise ValueError(f'unknown tower {name!r}; expected one of {sorted(TOWER_SPECS)}')
        self.name = name
        self.spec = TOWER_SPECS[name]
        self.device = torch.device(device)
        self.dim = self.spec['dim']
        if self.spec['kind'] == 'open_clip':
            self._init_open_clip()
        elif self.spec['kind'] == 'dinov3':
            self._init_dinov3(root)
        else:
            raise ValueError(f"unhandled tower kind {self.spec['kind']!r}")

    # -- open_clip towers (so400m256, gopt) ---------------------------------

    def _init_open_clip(self):
        try:
            import open_clip  # lazy: not needed for the dinov3-only CPU smoke
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'open_clip_torch is required for open_clip towers '
                '(so400m256/gopt); install it or select --towers dinov3') from e
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.spec['open_clip_name'], pretrained=self.spec['pretrained'])
        # Demo runs these encoders in fp16 on CUDA.
        self.compute_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        if self.device.type == 'cuda':
            model = model.half()
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess  # open_clip's own eval transforms

    # -- DINOv3 (local checkpoint dir) --------------------------------------

    def _init_dinov3(self, root):
        from transformers import AutoImageProcessor, AutoModel
        src = Path(root) / DINOV3_DIR if root else PROJECT_ROOT / DINOV3_DIR
        if not (src / 'config.json').is_file() or not (src / 'model.safetensors').is_file():
            raise FileNotFoundError(
                f'DINOv3 local checkpoint incomplete at {src}; expected config.json '
                f'and model.safetensors (fetch via scripts/fetch_assets.py)')
        # bf16 on CUDA (fp16 NaNs on A10G per the demo), float32 on CPU.
        self.compute_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.model = AutoModel.from_pretrained(src, dtype=self.compute_dtype).to(self.device).eval()
        proc = AutoImageProcessor.from_pretrained(src)
        # Demo preprocessing: Resize(256) -> CenterCrop(224) -> ToTensor ->
        # normalize with the processor's mean/std (NOT the processor pipeline).
        import torchvision.transforms as T
        self._tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        self._mean = torch.tensor(proc.image_mean).view(3, 1, 1)
        self._std = torch.tensor(proc.image_std).view(3, 1, 1)

    def _preprocess_dinov3(self, image):
        return (self._tf(image.convert('RGB')) - self._mean) / self._std

    # -- shared --------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, images):
        """images: list[PIL.Image] (a single PIL.Image is also accepted).

        Returns (B, dim) fp32 L2-normalized raw tower embedding — apply the
        tower's ProjHead separately to reach the 512-d retrieval space.
        """
        if isinstance(images, Image.Image):
            images = [images]
        if not images:
            raise ValueError('encode() requires at least one image')
        if self.spec['kind'] == 'open_clip':
            pixels = torch.stack([self.preprocess(im.convert('RGB')) for im in images])
            pixels = pixels.to(self.device, self.compute_dtype)
            e = self.model.encode_image(pixels)
        else:
            xs = torch.stack([self._preprocess_dinov3(im) for im in images])
            xs = xs.to(self.device, self.compute_dtype)
            out = self.model(pixel_values=xs)
            # Pooled output when the checkpoint provides it, else CLS token.
            e = (out.pooler_output if getattr(out, 'pooler_output', None) is not None
                 else out.last_hidden_state[:, 0])
        e = F.normalize(e.float(), dim=-1)
        if not torch.isfinite(e).all():
            raise RuntimeError(f'tower {self.name!r} produced non-finite embeddings')
        return e
