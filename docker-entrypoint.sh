#!/usr/bin/env bash
set -euo pipefail
# HF Spaces / some runtimes set HOME=/ so Triton (pulled in via torchao → unsloth import chain)
# tries to create /.triton and hits PermissionError.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
if [[ -z "${HOME:-}" || "${HOME}" == "/" ]]; then
  export HOME=/tmp
fi
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" 2>/dev/null || true
exec "$@"
