#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Local GGUF export — runs on Mac (16 GB RAM is enough).
#
# Prereqs:
#   brew install llama.cpp          # provides llama-quantize + convert_hf_to_gguf.py
#   pip install huggingface_hub gguf numpy sentencepiece protobuf transformers torch
#
# Usage:
#   ./scripts/local_gguf_export.sh auth      # 14B model
#   ./scripts/local_gguf_export.sh small     # 1.5B model
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PROFILE="${1:-auth}"

case "$PROFILE" in
  small)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-gguf"
    ARTIFACT="wagmi-qwen2.5-1.5b-sft"
    ;;
  auth)
    HUB_MERGED="jeanbaptdzd/wagmi-qwen2.5-14b-sft-merged"
    HUB_GGUF="jeanbaptdzd/wagmi-qwen2.5-14b-sft-gguf"
    ARTIFACT="wagmi-qwen2.5-14b-sft"
    ;;
  *)
    echo "Usage: $0 [small|auth]" >&2
    exit 1
    ;;
esac

QUANT="Q4_K_M"
WORK_DIR="output/${PROFILE}-gguf-local"
SYSTEM_PROMPT="Tu es Wagmi, le watchdog de Deal ex Machina. Reponds de maniere factuelle, concise, sans invention. Si l'information manque, dis clairement: 'Je ne sais pas avec certitude'."

echo "============================================================"
echo "  Local GGUF export: ${PROFILE}"
echo "  Merged repo:  ${HUB_MERGED}"
echo "  GGUF repo:    ${HUB_GGUF}"
echo "============================================================"

# ── Preflight checks ─────────────────────────────────────────
for cmd in llama-quantize convert_hf_to_gguf huggingface-cli python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' not found."
    case "$cmd" in
      llama-quantize|convert_hf_to_gguf)
        echo "  Install: brew install llama.cpp" ;;
      huggingface-cli)
        echo "  Install: pip install huggingface_hub" ;;
    esac
    exit 1
  fi
done

if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN="$(huggingface-cli whoami 2>/dev/null | head -1 || true)"
  if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set and not logged in. Run: huggingface-cli login"
    exit 1
  fi
  echo "Using cached HF credentials."
fi

# ── Step 1: Download merged model ────────────────────────────
MERGED_DIR="${WORK_DIR}/merged"
echo ""
echo ">>> Downloading merged model from ${HUB_MERGED} ..."
huggingface-cli download "${HUB_MERGED}" \
  --local-dir "${MERGED_DIR}" \
  --local-dir-use-symlinks False

echo "    Downloaded to ${MERGED_DIR}"
du -sh "${MERGED_DIR}"

# ── Step 2: Convert to bf16 GGUF ─────────────────────────────
BF16_GGUF="${WORK_DIR}/${ARTIFACT}-bf16.gguf"
echo ""
echo ">>> Converting to bf16 GGUF ..."
convert_hf_to_gguf "${MERGED_DIR}" \
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

# ── Step 4: Generate Modelfile ────────────────────────────────
MODELFILE="${WORK_DIR}/Modelfile.wagmi-sft"
GGUF_FILENAME="$(basename "${Q4_GGUF}")"

cat > "${MODELFILE}" <<MEOF
FROM ${GGUF_FILENAME}

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER num_ctx 2048
PARAMETER num_predict 220
PARAMETER temperature 0.2
PARAMETER top_k 30
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.12
PARAMETER repeat_last_n 128
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM """${SYSTEM_PROMPT}"""
MEOF

echo ""
echo "    Modelfile written to ${MODELFILE}"

# ── Step 5: Push GGUF + Modelfile to Hub ──────────────────────
echo ""
echo ">>> Creating repo ${HUB_GGUF} ..."
huggingface-cli repo create "${HUB_GGUF}" --type model --private 2>/dev/null || true

echo ">>> Uploading GGUF ($(du -h "${Q4_GGUF}" | cut -f1)) ..."
huggingface-cli upload "${HUB_GGUF}" "${Q4_GGUF}" "${GGUF_FILENAME}"

echo ">>> Uploading Modelfile ..."
huggingface-cli upload "${HUB_GGUF}" "${MODELFILE}" "Modelfile.wagmi-sft"

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
echo "  To test with Ollama locally:"
echo "    cd ${WORK_DIR}"
echo "    ollama create wagmi-sft -f Modelfile.wagmi-sft"
echo "    ollama run wagmi-sft \"C'est quoi Deal ex Machina ?\""
echo ""
echo "============================================================"
