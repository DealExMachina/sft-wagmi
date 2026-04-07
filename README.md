---
title: sft-wagmi
sdk: docker
app_port: 7860
---

<p align="center">
  <strong>sft-wagmi</strong><br>
  Supervised fine-tuning pipeline for <a href="https://dealexmachina.com">Deal ex Machina</a>'s AI watchdog
</p>

<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"><img src="https://img.shields.io/badge/Base_Model-Qwen_2.5_1.5B-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="Qwen 2.5"></a>
  <a href="https://github.com/unslothai/unsloth"><img src="https://img.shields.io/badge/Training-Unsloth-orange?logo=github" alt="Unsloth"></a>
  <a href="https://huggingface.co/jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"><img src="https://img.shields.io/badge/HuggingFace-Adapter-yellow?logo=huggingface" alt="HuggingFace"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/Inference-Ollama-black?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IndoaXRlIi8+PC9zdmc+" alt="Ollama"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/Judge-GPT--4o-412991?logo=openai&logoColor=white" alt="GPT-4o"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</p>

---

**Current version: `0.1.0`** (see [CHANGELOG.md](CHANGELOG.md))

Fine-tune [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) into **Wagmi**: a small quantized model that runs on CPU and answers questions about [Deal ex Machina](https://dealexmachina.com), its services, blog, and founder, in French and English, with guardrails against hallucination.

The stack has three layers:

1. **RAG** — BM25-style local retrieval injects verified facts into the system prompt at inference time in the site ([`local-rag.ts`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/local-rag.ts)).
2. **SFT** — JSONL chat examples generated from repo content (see below); identity, tone, uncertainty, and refusal reflexes.
3. **Autotune** — iterative judge–correct–retrain loop with GPT-4o scoring on six criteria until convergence (small + auth profiles).

**Dataset snapshot** (from `data/metadata.json` after sync; regenerate to refresh):

| | Train | Eval | Total |
| --- | --- | --- | --- |
| Examples | 663 | 117 | **780** |
| EN | 347 | | |
| FR | 433 | | |

Source mix (same generation run, `bySourceType`):

| Type | Rows | Role |
| --- | ---: | --- |
| `content` | 440 | Chunked blog + site markdown |
| `obsidian` | 140 | Optional vault notes (`wagmi_sft` / `sft` frontmatter, `OBSIDIAN_VAULT_PATH` in dexm) |
| `synthetic:guardrail` | 112 | Identity, refusal, uncertainty, auth nudges |
| `synthetic:grounded-qa` | 34 | Grounded Q&A |
| `synthetic:wagmi-qa` | 28 | Direct Wagmi / company Q&A |
| `synthetic:hard-negative` | 18 | Corrections of known bad patterns |
| `synthetic:multi-turn` | 8 | Multi-turn coherence |

Underlying content includes [`wagmi-skills.md`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/wagmi-skills.md), [`ai.txt`](https://github.com/DealExMachina/dexm-one-page/blob/dev/public/ai.txt), bilingual blog posts, and optional Obsidian. **Authoritative counts** after each generation live in `dexm-one-page/datasets/wagmi-sft/metadata.json`.

## Regenerating and syncing the dataset

From **dexm-one-page** (sibling repo):

```bash
cd ../dexm-one-page
npm run dataset:wagmi:refresh   # generate JSONL + copy to ../sft-wagmi/data
```

Or only copy already-generated files:

```bash
npm run dataset:wagmi:sync
```

Or from **this repo**:

```bash
python3 scripts/pipeline.py --sync-dataset
```

That runs `npm run dataset:wagmi:refresh` in `../dexm-one-page` when that path exists.

## Training

| Parameter | Value |
| --- | --- |
| Base model (small profile) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Method | LoRA (rank 32, alpha 64) |
| Target modules | q/k/v/o + gate/up/down proj |
| Max seq length | 2048 tokens |
| Learning rate | 2e-4 (cosine decay) |
| Epochs | 2 |
| Effective batch | 16 (4 per device, grad accum 4) |
| Precision | bf16 |
| Framework | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |
| Typical GPU | Hugging Face L40-class (48 GB VRAM) |

### Run training

```bash
pip install -r requirements.txt
```

Set `HF_TOKEN`, then either:

- Run `train.ipynb` top-to-bottom on a CUDA host, or
- Run `python3 train.py` (same profile env as the pipeline; see below).

The adapter is pushed to [`jeanbaptdzd/wagmi-qwen2.5-1.5b-sft`](https://huggingface.co/jeanbaptdzd/wagmi-qwen2.5-1.5b-sft) (small profile defaults).

## Autotune loop

`autotune.ipynb` or `autotune.py` implements the self-improvement loop:

```
[SFT model] --> inference on eval prompts (with RAG context where configured)
     |
     v
[GPT-4o judge] --> scores on 6 criteria (0-3 each)
     |              factual_accuracy, language_match, tone_persona,
     |              guardrail_compliance, conciseness, hallucination_free
     v
[GPT-4o corrector] --> ideal responses for failures (total < 14/18)
     |
     v
[Merge corrections] --> retrain --> push to Hub
     |
     v
[Repeat] --> until mean score > 2.5/3.0 or 3 iterations
```

Requires `OPENAI_API_KEY` and `HF_TOKEN`. Supports both `--profile small` and `--profile auth`.

## Export to Ollama / GGUF

**Primary (pipeline):** `export_gguf.py` merges the LoRA adapter, builds GGUF, and can push to Hub; invoked via:

```bash
python3 scripts/pipeline.py --export
# or: python3 export_gguf.py
```

Profiles: `MODEL_PROFILE=small` (default) or `auth` (Qwen 2.5 14B path; separate Hub repos — see env defaults in `export_gguf.py`).

For auth profile tool-calling specialization (email + calendar), `train.py` auto-injects
`data/tooling_email_calendar.jsonl` with oversampling:

- `AUTH_TOOLING_FILE` (default: `data/tooling_email_calendar.jsonl`)
- `AUTH_TOOLING_MULTIPLIER` (default: `3`)

**Standalone Mac/Linux helper:** `scripts/export_ollama.py` — download adapter, merge on CPU (~6 GB RAM for 1.5B), convert with [llama.cpp](https://github.com/ggerganov/llama.cpp), quantize Q4_K_M, register with Ollama:

```bash
pip install torch transformers peft huggingface_hub
HF_TOKEN=hf_xxx python3.11 scripts/export_ollama.py
```

Needs [Ollama](https://ollama.com) >= 0.5 and `llama.cpp` on PATH (e.g. `brew install llama.cpp`).

```bash
ollama run wagmi-sft
```

Quantized small build is on the order of **~1 GB** (Q4_K_M), CPU-friendly.

## One-command launcher

Full pipeline (on L40 HF Space):

```bash
python3 scripts/pipeline.py --all --profile auth
```

This runs: preflight -> merge `data/next/` -> train -> eval -> eval-rag -> export merged model to Hub.

Useful variants:

```bash
python3 scripts/pipeline.py --preflight --dry-run
python3 scripts/pipeline.py --merge-next                    # merge data/next/ + bump version
python3 scripts/pipeline.py --merge-next --bump patch       # patch bump instead of minor
python3 scripts/pipeline.py --profile auth --train
python3 scripts/pipeline.py --train --export-merged         # train + push merged model
python3 scripts/pipeline.py --sync-dataset                  # sync from dexm-one-page
python3 scripts/pipeline.py --autotune --profile auth       # requires OPENAI_API_KEY
python3 scripts/pipeline.py --all --profile auth --dry-run  # preview full pipeline
```

After the Space completes, run locally on your Mac:

```bash
./scripts/local_gguf_export.sh auth   # download merged, convert GGUF, push to Hub
```

Behavior:

- **`--all`** chains: preflight -> merge-next -> train -> eval -> eval-rag -> export-merged.
- **`--merge-next`** runs `scripts/merge_next.py` which validates `data/next/*.jsonl`, appends to
  train/eval (85/15 split), updates `metadata.json`, bumps `VERSION`, and clears `data/next/`.
- **`--export-merged`** runs `export_merged.py` (LoRA merge + push to Hub). GGUF conversion is local.
- **`--profile`**: `small` (Qwen 1.5B) or `auth` (Qwen 2.5 14B).
- **`HF_TOKEN`**: Hub pull/push. **`OPENAI_API_KEY`**: autotune judge.
- Large artifacts (adapters, merged models, GGUF) go to **HF Hub, not GitHub** (see `.gitignore`).

**HF Spaces / Gradio:** `app.py` mirrors the same flow in a UI with a "0. Merge Next Data" tab.

## Versioning

Model versions follow [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR**: breaking persona change, base model swap, or schema change.
- **MINOR**: new dataset entries, new training profiles, new capabilities.
- **PATCH**: hyperparameter tweaks, bug fixes, tooling improvements.

The current version lives in `VERSION` at the repo root. It is read by `train.py`
at startup and embedded in Hub commit messages. See [CHANGELOG.md](CHANGELOG.md)
for the full history.

### Preparing a new version

1. Drop new dataset entries into `data/next/` during development.
2. When ready, merge them into `data/train.jsonl` / `data/eval.jsonl` (via the dexm
   generator or manually).
3. Bump `VERSION`, add a `CHANGELOG.md` entry, retrain, and push.

## Repository structure

```
sft-wagmi/
├── data/
│   ├── train.jsonl                  # SFT training rows (JSONL chat format)
│   ├── eval.jsonl                   # Held-out eval rows
│   ├── tooling_email_calendar.jsonl # Auth-profile tool-calling examples
│   ├── metadata.json                # Counts + distribution + version
│   └── next/                        # Staging area for next-version entries
├── scripts/
│   ├── pipeline.py                  # CLI orchestration (--all, --train, --export-merged ...)
│   ├── merge_next.py                # Merge data/next/ into train/eval + bump VERSION
│   ├── export_ollama.py             # Standalone merge + GGUF + Ollama import
│   └── local_gguf_export.sh         # Local Mac GGUF conversion + quantization
├── baseline.py                      # Baseline eval (script)
├── train.py                         # Unsloth SFT (script)
├── autotune.py                      # Judge loop (script)
├── eval_sft.py                      # Post-training eval
├── eval_sft_rag.py                  # Eval with RAG-style context
├── eval_tool_calls.py               # Tool-calling eval
├── export_gguf.py                   # GGUF export + Hub (Space, legacy)
├── export_merged.py                 # LoRA merge + push merged model (Space)
├── retrain_step.py                  # Helper for autotune merge/retrain
├── app.py                           # Gradio Space entry
├── baseline.ipynb                   # Notebook alternative for baseline
├── train.ipynb                      # Notebook alternative for training
├── autotune.ipynb                   # Notebook alternative for autotune
├── requirements.txt
├── VERSION                          # Semver (e.g. 0.1.0)
├── CHANGELOG.md                     # Version history
└── README.md
```

## Attribution

| Component | License | Link |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | Apache 2.0 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Unsloth | Apache 2.0 | [unslothai/unsloth](https://github.com/unslothai/unsloth) |
| TRL | Apache 2.0 | [huggingface/trl](https://github.com/huggingface/trl) |
| llama.cpp | MIT | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Ollama | MIT | [ollama/ollama](https://github.com/ollama/ollama) |
| GPT-4o (judge) | Proprietary | [OpenAI](https://openai.com) |

The LoRA adapter and SFT dataset are original work by [Deal ex Machina](https://dealexmachina.com). Base model weights remain the property of the Qwen team (Alibaba Cloud).

## Related

- [dexm-one-page](https://github.com/DealExMachina/dexm-one-page) — production site and `scripts/generate-wagmi-sft-dataset.ts`
- [Dresser un petit modèle sur CPU](https://dealexmachina.com/fr/blog/dresser-petit-modele-cpu) — long-form write-up (FR)
- [SFT Wagmi, pipeline rudimentaire](https://dealexmachina.com/fr/blog/2026-04-06-sft-wagmi-rudimentary-pipeline-fr) — pipeline notes (FR)

---

<sub>Built with [Cursor](https://cursor.com) by [Deal ex Machina](https://dealexmachina.com) — *The optimal path between vision and results.*</sub>
