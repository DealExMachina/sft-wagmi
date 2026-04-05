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

Fine-tune [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) into **Wagmi**: a small quantized model that runs on CPU and answers questions about [Deal ex Machina](https://dealexmachina.com), its services, blog, and founder, in French and English, with guardrails against hallucination.

The stack has three layers:

1. **RAG** — BM25-style local retrieval injects verified facts into the system prompt at inference time in the site ([`local-rag.ts`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/local-rag.ts)).
2. **SFT** — JSONL chat examples generated from repo content (see below); identity, tone, uncertainty, and refusal reflexes.
3. **Autotune** — iterative judge–correct–retrain loop with GPT-4o scoring on six criteria until convergence (small profile only).

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

Requires `OPENAI_API_KEY` and `HF_TOKEN`. **`--profile auth`**: training and export are supported; **autotune is disabled for the auth profile** in `pipeline.py` (small only).

## Export to Ollama / GGUF

**Primary (pipeline):** `export_gguf.py` merges the LoRA adapter, builds GGUF, and can push to Hub; invoked via:

```bash
python3 scripts/pipeline.py --export
# or: python3 export_gguf.py
```

Profiles: `MODEL_PROFILE=small` (default) or `auth` (Mistral Small 24B path; separate Hub repos — see env defaults in `export_gguf.py`).

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

```bash
python3 scripts/pipeline.py --all
```

Useful variants:

```bash
python3 scripts/pipeline.py --preflight
python3 scripts/pipeline.py --sync-dataset
python3 scripts/pipeline.py --profile small --train
python3 scripts/pipeline.py --profile auth --train
python3 scripts/pipeline.py --eval
python3 scripts/pipeline.py --eval-rag
python3 scripts/pipeline.py --all --dry-run
```

Behavior:

- **Steps** prefer root Python scripts (`baseline.py`, `train.py`, `autotune.py`, `eval_sft.py`, `eval_sft_rag.py`, `export_gguf.py`). If a script is missing, the launcher **falls back** to the matching `.ipynb` via `jupyter nbconvert --execute` when Jupyter is installed.
- **`--sync-dataset`** runs `npm run dataset:wagmi:refresh` in `../dexm-one-page`.
- **`--profile`**: `small` (Qwen 1.5B) or `auth` (Mistral 24B-style path); also `MODEL_PROFILE` env.
- **`HF_TOKEN`**: Hub pull/push. **`OPENAI_API_KEY`**: autotune judge (and related evals if configured).

**HF Spaces / Gradio:** `app.py` mirrors the same flow in a UI (see Space config in the YAML header above).

## Repository structure

```
sft-wagmi/
├── data/
│   ├── train.jsonl           # SFT training rows (JSONL chat format)
│   ├── eval.jsonl            # Held-out eval rows
│   └── metadata.json         # Counts + distribution (from dexm generator)
├── scripts/
│   ├── pipeline.py           # CLI orchestration (sync, train, eval, export)
│   └── export_ollama.py      # Standalone merge + GGUF + Ollama import
├── baseline.py               # Baseline eval (script)
├── train.py                  # Unsloth SFT (script)
├── autotune.py               # Judge loop (script)
├── eval_sft.py               # Post-training eval
├── eval_sft_rag.py           # Eval with RAG-style context
├── export_gguf.py            # GGUF export + Hub (used by --export)
├── retrain_step.py           # Helper for autotune merge/retrain
├── app.py                    # Gradio Space entry
├── baseline.ipynb            # Notebook alternative for baseline
├── train.ipynb               # Notebook alternative for training
├── autotune.ipynb            # Notebook alternative for autotune
├── requirements.txt
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
