#!/usr/bin/env bash
set -euo pipefail
SIM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$SIM_DIR/.venv"
"$SIM_DIR/.venv/bin/python" -m pip install -r "$SIM_DIR/requirements.txt"
"$SIM_DIR/.venv/bin/python" "$SIM_DIR/sim.py" check
