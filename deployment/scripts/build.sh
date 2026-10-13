#!/usr/bin/env bash
# 使用目标设备的 CUDA/TensorRT 开发环境；不复制主机二进制。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  if [[ -f /usr/include/NvInfer.h || -f /usr/include/aarch64-linux-gnu/NvInfer.h || -f /usr/include/x86_64-linux-gnu/NvInfer.h ]]; then
    TENSORRT_ROOT=/usr
  elif [[ -d "$HOME/opt/tensorrt/TensorRT-10.14.1.48" ]]; then
    TENSORRT_ROOT="$HOME/opt/tensorrt/TensorRT-10.14.1.48"
  else
    echo '请 export TENSORRT_ROOT=/你的/TensorRT-SDK目录' >&2; exit 1
  fi
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then CUDA_HOME=/usr/local/cuda
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then CUDA_HOME="$CONDA_PREFIX"
  else echo '请 export CUDA_HOME=/包含bin/nvcc的CUDA目录' >&2; exit 1
  fi
fi
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-87;89}"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
 -DTENSORRT_ROOT="$TENSORRT_ROOT" -DCUDAToolkit_ROOT="$CUDA_HOME" \
 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" "$@"
cmake --build "$ROOT/build" -j "${BUILD_JOBS:-2}"
ctest --test-dir "$ROOT/build" --output-on-failure
