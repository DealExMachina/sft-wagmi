#!/usr/bin/env bash
# Run auth-profile red-team evals on a Hugging Face Space (or any host) with an L40 / CUDA.
# Prerequisites: repo root, Python env with torch + unsloth + eval deps (see requirements.txt),
# HF_TOKEN if loading adapters from the Hub by repo id.
#
# Usage (inside Space / SSH):
#   chmod +x scripts/hf/run_redteam_auth_l40.sh
#   ./scripts/hf/run_redteam_auth_l40.sh
#
# Override adapter paths with AUTH_OUTPUT_DIR / AUTH_HUB_MODEL_ID (see config.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== Red team auth (Qwen 14B + LFM2) — cwd=$ROOT =="

python3 <<'PY'
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch is not installed in this environment.")
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. Run this script on the HF GPU Space (L40).")
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -f VERSION ]]; then
  echo "VERSION: $(tr -d '\r\n' <VERSION)"
else
  echo "WARN: VERSION file missing"
fi

echo ""
echo "--- LLM_FAMILY=qwen MODEL_PROFILE=auth ---"
export LLM_FAMILY=qwen
export MODEL_PROFILE=auth
python3 eval_redteam.py

echo ""
echo "--- LLM_FAMILY=lfm2 MODEL_PROFILE=auth ---"
export LLM_FAMILY=lfm2
export MODEL_PROFILE=auth
python3 eval_redteam.py

echo ""
VER="unknown"
if [[ -f VERSION ]]; then VER="$(tr -d '\r\n' <VERSION)"; fi
echo "Done. Reports under reports/redteam/v${VER}/"
