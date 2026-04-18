#!/usr/bin/env bash
# Run inside the HF Space container (SSH or one-off) to confirm /app matches a fresh image build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "== sft-wagmi HF / Docker self-check (cwd=$ROOT) =="
if [[ -f VERSION ]]; then
  echo "VERSION file: $(tr -d '\r\n' <VERSION)"
else
  echo "VERSION file: MISSING"
  exit 1
fi
if grep -q "_parse_cli" retrain_step.py && grep -q -- '--family' retrain_step.py; then
  echo "retrain_step.py: OK (argparse --family / deferred Unsloth import)"
else
  echo "retrain_step.py: STALE — this tree is not 0.3.1+ ; wait for Space rebuild after git push"
  exit 1
fi
if grep -q "read_package_version" autotune.py; then
  echo "autotune.py: OK (prints package version at startup)"
else
  echo "autotune.py: STALE"
  exit 1
fi
echo "All checks passed."
