#!/usr/bin/env bash
# 原生 C++ 推理，不调用 Python。tar SDK 通过 TENSORRT_ROOT 指定；系统 SDK 用系统库路径。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TENSORRT_ROOT="${TENSORRT_ROOT:-/usr}"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
exec "$ROOT/build/bin/pointr_batch" "$@"
