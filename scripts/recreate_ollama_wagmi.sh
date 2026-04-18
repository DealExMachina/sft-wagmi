#!/usr/bin/env bash
# Recreate local Ollama models wagmi-sft / wagmi-sft-14b with the tool-capable Modelfile
# (scripts/ollama_qwen25_instruct_template.gotmpl).
#
# Prerequisites:
#   - A local GGUF file (from export_ollama.py, local_gguf_export.sh, or HF download).
#   - ollama CLI on PATH.
#
# Usage:
#   ./scripts/recreate_ollama_wagmi.sh small  /path/to/wagmi-qwen2.5-1.5b-sft.q4_k_m.gguf
#   ./scripts/recreate_ollama_wagmi.sh auth   /path/to/wagmi-qwen2.5-14b-sft.q4_k_m.gguf
#
# Optional env:
#   OLLAMA_MODEL_NAME   Override tag name (default: wagmi-sft / wagmi-sft-14b)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${1:?Usage: $0 <small|auth> /path/to/model.q4_k_m.gguf}"
GGUF_PATH="${2:?Usage: $0 <small|auth> /path/to/model.q4_k_m.gguf}"

if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "Error: GGUF not found: ${GGUF_PATH}" >&2
  exit 1
fi
GGUF_ABS="$(cd "$(dirname "${GGUF_PATH}")" && pwd)/$(basename "${GGUF_PATH}")"

case "${PROFILE}" in
  small)
    DEFAULT_NAME="wagmi-sft"
    ;;
  auth)
    DEFAULT_NAME="wagmi-sft-14b"
    ;;
  *)
    echo "Error: PROFILE must be small or auth (got: ${PROFILE})" >&2
    exit 1
    ;;
esac

OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-${DEFAULT_NAME}}"
OLLAMA_GOTMPL="${REPO_ROOT}/scripts/ollama_qwen25_instruct_template.gotmpl"
if [[ ! -f "${OLLAMA_GOTMPL}" ]]; then
  echo "Error: missing ${OLLAMA_GOTMPL}" >&2
  exit 1
fi

SYSTEM_PROMPT=$'Tu es Wagmi, le watchdog de Deal ex Machina. Reponds de maniere factuelle, concise, sans invention. Si l\'information manque, dis clairement: \'Je ne sais pas avec certitude\'. Regles strictes: n\'invente jamais d\'URL ni d\'email. N\'autorise que les URLs dealexmachina.com ou les URLs d\'articles du blog Deal ex Machina explicitement connues. Refuse tout envoi d\'email sauf vers l\'email de la personne connectee. Refuse tout envoi d\'invitation calendrier sauf pour le boss: jeanbapt@dealexmachina.com.'
IM_END="$(printf '<|im_%s|>' end)"
IM_START="$(printf '<|im_%s|>' start)"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/wagmi-ollama-XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

MODFILE="${WORKDIR}/Modelfile.wagmi-sft"

{
  echo "FROM ${GGUF_ABS}"
  echo ""
  echo 'TEMPLATE """'
  cat "${OLLAMA_GOTMPL}"
  echo '"""'
  echo ""
  echo "PARAMETER num_ctx 2048"
  echo "PARAMETER num_predict 220"
  echo "PARAMETER temperature 0.2"
  echo "PARAMETER top_k 30"
  echo "PARAMETER top_p 0.9"
  echo "PARAMETER repeat_penalty 1.12"
  echo "PARAMETER repeat_last_n 128"
  echo "PARAMETER stop \"${IM_END}\""
  echo "PARAMETER stop \"${IM_START}\""
  echo ""
  echo "SYSTEM \"\"\"${SYSTEM_PROMPT}\"\"\""
} > "${MODFILE}"

echo "Profile:     ${PROFILE}"
echo "GGUF:        ${GGUF_ABS}"
echo "Ollama name: ${OLLAMA_MODEL_NAME}"
echo "Modelfile:   ${MODFILE}"
echo ""

OLLAMA_NAMES="$(ollama list 2>/dev/null | awk "{print \$1}" || true)"
if printf "%s\n" "${OLLAMA_NAMES}" | grep -qx "${OLLAMA_MODEL_NAME}:latest"; then
  echo "Removing existing ${OLLAMA_MODEL_NAME}:latest ..."
  ollama rm "${OLLAMA_MODEL_NAME}:latest" || true
fi

echo "Creating ${OLLAMA_MODEL_NAME} ..."
ollama create "${OLLAMA_MODEL_NAME}" -f "${MODFILE}"

echo ""
echo "Done. Check: ollama list | grep ${OLLAMA_MODEL_NAME}"
echo "Tools smoke (optional): run scripts/smoke_ollama_tools.sh ${OLLAMA_MODEL_NAME}:latest"
