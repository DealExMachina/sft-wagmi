#!/usr/bin/env bash
set -euo pipefail
# HF Spaces / some runtimes set HOME=/ so Triton (pulled in via torchao -> unsloth import chain)
# tries to create /.triton and hits PermissionError.
if [[ -n "${CACHE_BASE_DIR:-}" ]]; then
  _cache_base="${CACHE_BASE_DIR}"
elif [[ -d /data ]]; then
  _cache_base="/data"
else
  _cache_base="/tmp"
fi

if [[ -z "${HOME:-}" || "${HOME}" == "/" ]]; then
  export HOME=/tmp
fi

# Ensure cache base is writable, else fallback to /tmp.
mkdir -p "${_cache_base}" 2>/dev/null || true
if [[ ! -w "${_cache_base}" ]]; then
  _cache_base="/tmp"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_cache_base}/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_cache_base}/.cache}"
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" 2>/dev/null || true
chmod 777 "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" 2>/dev/null || true
if [[ -f /app/VERSION ]]; then
  echo "sft-wagmi: container /app/VERSION=$(tr -d '\r\n' </app/VERSION) (rebuild this Space after git push to refresh code in SSH)"
fi
exec "$@"
