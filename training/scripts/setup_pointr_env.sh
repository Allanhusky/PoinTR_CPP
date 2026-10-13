#!/usr/bin/env bash
# Run after: conda activate pointr
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "${CONDA_DEFAULT_ENV:-}" != "pointr" ]]; then
  echo '请先执行 conda activate pointr' >&2
  exit 1
fi
python - <<'PY'
import torch
assert torch.__version__ == '2.1.2+cu118', 'This setup is validated with torch 2.1.2+cu118'
assert torch.version.cuda == '11.8'
PY
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Set TORCH_CUDA_ARCH_LIST explicitly when building without GPU access'
major, minor = torch.cuda.get_device_capability(0)
print(f'{major}.{minor}')
PY
)"
  export TORCH_CUDA_ARCH_LIST
fi
python -m pip install -r requirements-training.txt
# The cu118 PyTorch wheel ships a hashed NVRTC library name; cuDNN probes libnvrtc.so.
python - <<'PY'
from pathlib import Path
import torch
root = Path(torch.__file__).resolve().parent / 'lib'
alias = root / 'libnvrtc.so'
libraries = list(root.glob('libnvrtc-*.so.11.2'))
if not alias.exists() and len(libraries) == 1:
    alias.symlink_to(libraries[0].name)
PY
POINTNET_SOURCE="$PROJECT_DIR/third_party/Pointnet2_PyTorch"
if [[ ! -d "$POINTNET_SOURCE" ]]; then
  git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$POINTNET_SOURCE"
  git -C "$POINTNET_SOURCE" checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
fi
if git -C "$POINTNET_SOURCE" apply --check "$PROJECT_DIR/scripts/pointnet2_compat.patch"; then
  git -C "$POINTNET_SOURCE" apply "$PROJECT_DIR/scripts/pointnet2_compat.patch"
else
  git -C "$POINTNET_SOURCE" apply --reverse --check "$PROJECT_DIR/scripts/pointnet2_compat.patch"
fi
for extension in extensions/chamfer_dist "$POINTNET_SOURCE/pointnet2_ops_lib" \
                 extensions/emd extensions/cubic_feature_sampling extensions/gridding extensions/gridding_loss; do
  python -m pip install --no-build-isolation --no-deps "$extension"
done
python -m pip check
python tools/check_training_env.py --report environment/training_check.json
