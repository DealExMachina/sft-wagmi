#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Local GGUF export — runs on Mac (16 GB RAM is enough).
#
# Prereqs:
#   brew install llama.cpp
#   python3 -m venv .venv-gguf
#   Match gguf-py to your Homebrew llama.cpp commit (avoids MODEL_ARCH / convert script drift):
#     BREW_REV=$(python3 -c "import json,glob; p=glob.glob('/opt/homebrew/Cellar/llama.cpp/*/INSTALL_RECEIPT.json')[0]; print(json.load(open(p))['source']['scm_revision'])")
#     .venv-gguf/bin/pip install "gguf@git+https://github.com/ggml-org/llama.cpp@${BREW_REV}#subdirectory=gguf-py" \
#       numpy sentencepiece protobuf transformers torch huggingface_hub
#
# Usage:
#   ./scripts/local_gguf_export.sh auth      # 14B SFT merged
#   ./scripts/local_gguf_export.sh auth-dpo  # 14B DPO merged (after Space export tab 7c)
#   ./scripts/local_gguf_export.sh auth-grpo # 14B GRPO merged (after Space export tab 7d)
#   ./scripts/local_gguf_export.sh small   # 1.5B model
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${SCRIPT_DIR}/.venv-gguf"
VENV_PYTHON="${VENV}/bin/python3"
HF_CLI="${VENV}/bin/hf"

PROFILE="${1:-auth}"

case "$PROFILE" in
  small)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-gguf"
    ARTIFACT="wagmi-qwen2.5-1.5b-sft"
    DEFAULT_OLLAMA="wagmi-sft"
    ;;
  auth)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-14b-sft-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-14b-sft-gguf"
    ARTIFACT="wagmi-qwen2.5-14b-sft"
    DEFAULT_OLLAMA="wagmi-sft-14b"
    ;;
  auth-dpo)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-gguf"
    ARTIFACT="wagmi-qwen2.5-14b-sft-dpo"
    DEFAULT_OLLAMA="wagmi-sft-14b-dpo"
    ;;
  auth-grpo)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo-gguf"
    ARTIFACT="wagmi-qwen2.5-14b-sft-dpo-grpo"
    DEFAULT_OLLAMA="wagmi-sft-14b-grpo"
    ;;
  *)
    echo "Usage: $0 [small|auth|auth-dpo|auth-grpo]" >&2
    exit 1
    ;;
esac

# Matches dexm-one-page local defaults (ollama list: wagmi-sft:latest, wagmi-sft-14b:latest).
OLLAMA_MODEL="${OLLAMA_MODEL:-$DEFAULT_OLLAMA}"

QUANT="Q4_K_M"
WORK_DIR="output/${PROFILE}-gguf-local"
SYSTEM_PROMPT="Tu es Wagmi, le watchdog de Deal ex Machina. Reponds de maniere factuelle, concise, sans invention. Si l'information manque, dis clairement: 'Je ne sais pas avec certitude'. Regles strictes: n'invente jamais d'URL ni d'email. N'autorise que les URLs dealexmachina.com ou les URLs d'articles du blog Deal ex Machina explicitement connues. Refuse tout envoi d'email sauf vers l'email de la personne connectee. Refuse tout envoi d'invitation calendrier sauf pour le boss: jeanbapt@dealexmachina.com."

echo "============================================================"
echo "  Local GGUF export: ${PROFILE}"
echo "  Merged repo:  ${HUB_MERGED}"
echo "  GGUF repo:    ${HUB_GGUF}"
echo "============================================================"

# ── Preflight checks ─────────────────────────────────────────
if ! command -v llama-quantize &>/dev/null; then
  echo "ERROR: llama-quantize not found. Install: brew install llama.cpp"
  exit 1
fi
if [ ! -f "${VENV_PYTHON}" ]; then
  echo "ERROR: venv not found at ${VENV}"
  echo "  Create it:"
  echo "    python3 -m venv .venv-gguf"
  echo "    .venv-gguf/bin/pip install 'gguf@git+https://github.com/ggml-org/llama.cpp@b8680#subdirectory=gguf-py' numpy sentencepiece protobuf transformers torch huggingface_hub"
  exit 1
fi

CONVERT_SCRIPT="$(find /opt/homebrew/Cellar/llama.cpp -name 'convert_hf_to_gguf.py' 2>/dev/null | head -1)"
if [ -z "${CONVERT_SCRIPT}" ]; then
  echo "ERROR: convert_hf_to_gguf.py not found in llama.cpp Cellar"
  exit 1
fi
echo "Using converter: ${CONVERT_SCRIPT}"
echo "Using venv:      ${VENV}"

if [ -z "${HF_TOKEN:-}" ]; then
  "${HF_CLI}" auth whoami &>/dev/null || { echo "ERROR: not logged in. Run: ${HF_CLI} auth login"; exit 1; }
  echo "Using cached HF credentials."
fi

# ── Step 1: Download merged model ────────────────────────────
MERGED_DIR="${WORK_DIR}/merged"
echo ""
echo ">>> Downloading merged model from ${HUB_MERGED} ..."
"${HF_CLI}" download "${HUB_MERGED}" \
  --local-dir "${MERGED_DIR}"

echo "    Downloaded to ${MERGED_DIR}"
du -sh "${MERGED_DIR}"

# ── Step 2: Convert to bf16 GGUF ─────────────────────────────
BF16_GGUF="${WORK_DIR}/${ARTIFACT}-bf16.gguf"
echo ""
echo ">>> Converting to bf16 GGUF ..."
"${VENV_PYTHON}" "${CONVERT_SCRIPT}" "${MERGED_DIR}" \
  --outfile "${BF16_GGUF}" \
  --outtype bf16

echo "    bf16 GGUF: $(du -h "${BF16_GGUF}" | cut -f1)"

# ── Step 3: Quantize to Q4_K_M ───────────────────────────────
Q4_GGUF="${WORK_DIR}/${ARTIFACT}.q4_k_m.gguf"
echo ""
echo ">>> Quantizing bf16 → ${QUANT} ..."
llama-quantize "${BF16_GGUF}" "${Q4_GGUF}" "${QUANT}"

echo "    Q4_K_M GGUF: $(du -h "${Q4_GGUF}" | cut -f1)"

# Clean up bf16 intermediate (large)
rm -f "${BF16_GGUF}"
echo "    Removed bf16 intermediate."

# ── Step 4: Modelfile (Ollama Qwen2.5 instruct template; native tools) ──
MODELFILE="${WORK_DIR}/Modelfile.wagmi-sft"
GGUF_FILENAME="$(basename "${Q4_GGUF}")"
OLLAMA_GOTMPL="${SCRIPT_DIR}/scripts/ollama_qwen25_instruct_template.gotmpl"
IM_END='<|im_end|>'

{
  echo "FROM ${GGUF_FILENAME}"
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
  echo 'PARAMETER stop "<|im_start|>"'
  echo ""
  echo "SYSTEM \"\"\"${SYSTEM_PROMPT}\"\"\""
} > "${MODELFILE}"

echo ""
echo "    Modelfile written to ${MODELFILE}"

# ── Step 5: Push GGUF + Modelfile to Hub ──────────────────────
echo ""
echo ">>> Creating repo ${HUB_GGUF} ..."
"${HF_CLI}" repos create "${HUB_GGUF}" --type model --private 2>/dev/null || true

echo ">>> Uploading GGUF ($(du -h "${Q4_GGUF}" | cut -f1)) ..."
"${HF_CLI}" upload "${HUB_GGUF}" "${Q4_GGUF}" "${GGUF_FILENAME}"

echo ">>> Uploading Modelfile ..."
"${HF_CLI}" upload "${HUB_GGUF}" "${MODELFILE}" "Modelfile.wagmi-sft"

echo ""
echo "============================================================"
echo "  DONE"
echo "============================================================"
echo ""
echo "  GGUF:      https://huggingface.co/${HUB_GGUF}"
echo "  Modelfile: Modelfile.wagmi-sft"
echo ""
echo "  Local files in: ${WORK_DIR}/"
echo "    ${GGUF_FILENAME}  (deploy this)"
echo "    Modelfile.wagmi-sft"
echo ""
echo "  To test with Ollama locally (model name: ${OLLAMA_MODEL} — override with OLLAMA_MODEL=...):"
echo "    cd ${WORK_DIR}"
echo "    ollama create ${OLLAMA_MODEL} -f Modelfile.wagmi-sft"
echo "    ollama run ${OLLAMA_MODEL} \"C'est quoi Deal ex Machina ?\""
echo ""
echo "============================================================"
