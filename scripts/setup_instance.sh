#!/usr/bin/env bash
# GPU instance setup: run ONCE on the Linux CUDA box (not on macOS).
#
# Creates or reuses .venv, installs CUDA 12.8 PyTorch + torchvision from the
# explicit PyTorch index (NO CPU fallback), installs the remaining
# requirements, and fails if CUDA is not visible. Exits nonzero on
# non-Linux hosts before touching pip.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Linux" ]; then
  echo "error: setup_instance.sh targets the Linux GPU instance only" >&2
  exit 1
fi

echo "== python check (need >=3.10; 3.11 recommended) =="
python3 - <<'PY'
import sys
v = sys.version_info
if v < (3, 10):
    sys.exit(f'error: python3 is {v.major}.{v.minor}; need >=3.10 (3.11 recommended)')
print(f'python3 {v.major}.{v.minor}.{v.micro} ok')
PY

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
VENV=.venv/bin/python
"$VENV" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV" -m pip install -q --upgrade pip

echo "== torch/torchvision from CUDA 12.8 index (no CPU fallback) =="
"$VENV" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "== CUDA check (fails on a GPU-less host) =="
"$VENV" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('FATAL: CUDA torch is installed but no GPU is visible; '
             'this script is for the GPU instance only.')
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '|', torch.cuda.get_device_name(0))
PY

echo "== remaining requirements =="
"$VENV" -m pip install -q -r requirements-gpu.txt

echo "== versions + final CUDA check (post-resolution) =="
"$VENV" - <<'PY'
import importlib
import sys
import torch
for mod in ['torch', 'torchvision', 'transformers', 'open_clip',
            'huggingface_hub', 'safetensors', 'numpy', 'PIL']:
    m = importlib.import_module(mod)
    print(f'{mod}: {getattr(m, "__version__", "?")}')
if not torch.cuda.is_available():
    sys.exit('FATAL: CUDA lost after requirements install; '
             'check for a CPU torch being pulled in')
print('final CUDA check:', torch.version.cuda, '|',
      torch.cuda.get_device_name(0))
PY
echo "setup complete"
