#!/usr/bin/env bash
# POST /api/chat with a dummy tool. Exit 0 if Ollama accepts tools for this model.
set -euo pipefail
MODEL="${1:?Usage: $0 <model:tag e.g. wagmi-sft:latest>}"
BODY=$(python3 -c "import json; print(json.dumps({'model':'${MODEL}','messages':[{'role':'user','content':'hi'}],'tools':[{'type':'function','function':{'name':'ping','description':'noop','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}}],'stream':False}))")
RESP=$(curl -sS --max-time 120 http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" -d "${BODY}" || true)
if echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(1 if d.get('error') else 0)" 2>/dev/null; then
  echo "OK: ${MODEL} accepts tools in /api/chat"
else
  echo "FAIL: ${RESP}" >&2
  exit 1
fi
