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
  <a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"><img src="https://img.shields.io/badge/Small-Qwen_2.5_1.5B-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="Qwen 2.5 1.5B"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct"><img src="https://img.shields.io/badge/Auth-Qwen_2.5_14B-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="Qwen 2.5 14B"></a>
  <a href="https://huggingface.co/unsloth/LFM2.5-1.2B-Instruct"><img src="https://img.shields.io/badge/Small-LFM2.5_1.2B-teal?logo=huggingface" alt="LFM2.5 1.2B Unsloth mirror"></a>
  <a href="https://huggingface.co/LiquidAI/LFM2-24B-A2B"><img src="https://img.shields.io/badge/Auth-LFM2_24B-teal?logo=huggingface" alt="LFM2 24B LiquidAI"></a>
  <a href="https://huggingface.co/collections/LiquidAI/lfm2-model-collection-67f8152be4674776f7de900e"><img src="https://img.shields.io/badge/Liquid_AI-LFM2-0d9488?logo=huggingface" alt="Liquid AI LFM2 collection"></a>
</p>
<p align="center">
  <a href="https://github.com/unslothai/unsloth"><img src="https://img.shields.io/badge/Training-Unsloth-orange?logo=github" alt="Unsloth"></a>
  <a href="https://huggingface.co/jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"><img src="https://img.shields.io/badge/HuggingFace-Adapter-yellow?logo=huggingface" alt="HuggingFace"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/Inference-Ollama-black?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IndoaXRlIi8+PC9zdmc+" alt="Ollama"></a>
  <a href="https://github.com/ggerganov/llama.cpp"><img src="https://img.shields.io/badge/llama.cpp-GGUF%20%26%20server_(eval)-654321?logo=github&logoColor=white" alt="llama.cpp"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/Judge-GPT--4o-412991?logo=openai&logoColor=white" alt="GPT-4o"></a>
  <a href="#attribution-and-licensing"><img src="https://img.shields.io/badge/License-multi--see_table-slategray" alt="Licenses"></a>
</p>

---

**Version `0.3.5`.** History: [CHANGELOG.md](CHANGELOG.md).

SFT and export for **Wagmi** (bilingual assistant for [Deal ex Machina](https://dealexmachina.com)). Base families: **Qwen 2.5** (`LLM_FAMILY=qwen`) and **Liquid LFM2** (`LLM_FAMILY=lfm2`). Profiles: **`small`** (anonymous tier), **`auth`** (tool-capable tier). Training: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl). Default Hub checkpoints for LFM2 small and Qwen3 training paths use **`unsloth/...`** IDs; lfm2 auth uses **`LiquidAI/LFM2-24B-A2B`** (TRL+PEFT path, no Unsloth mirror). See Training. Production inference is **Ollama** today; **llama.cpp** is used for GGUF conversion and is under evaluation as a direct OpenAI-compatible server ([dexm-one-page](https://github.com/DealExMachina/dexm-one-page): `LLM_RUNTIME=llamacpp`, `LLM_API_URL_*`).

Downstream site couples **RAG** ([`local-rag.ts`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/local-rag.ts)), **SFT** (this repo’s JSONL), and optional **autotune** (GPT-4o judge loop in `autotune.py`).

## Dataset

Authoritative row counts and distributions: **`data/metadata.json`** (updated by `merge_next.py` or dataset sync from dexm). Regenerate from the site repo:

```bash
cd ../dexm-one-page && pnpm run dataset:wagmi:refresh   # or pnpm run dataset:wagmi:sync
```

From here:

```bash
python3 scripts/pipeline.py --sync-dataset   # subprocess: npm run dataset:wagmi:refresh in ../dexm-one-page
```

## Pipeline

| File | Role |
| --- | --- |
| [`scripts/pipeline.py`](scripts/pipeline.py) | CLI: loads `.env`, sets `MODEL_PROFILE` / `LLM_FAMILY` for subprocesses. |
| [`config.py`](config.py) | `_REGISTRY`, `resolve_profile_config()`; overrides via `SMALL_*` / `AUTH_*` env (see module docstring). |

**`--all`:** `preflight` → `merge-next` (only if `data/next/*.jsonl` exist) → `train` → `eval` → `eval-rag` → `redteam` → `export-merged`. Does not run `baseline`, `autotune`, or `export-gguf` unless requested.

**Flags:** `--profile small|auth`, `--family qwen|lfm2` (defaults from env; **`qwen3`** exists in `config.py` only for direct `python3 train.py` / eval until CLI adds it), `--sync-dataset`, `--merge-next [--bump patch|minor|major]`, `--dry-run`, per-step toggles as in `--help`.

**Artifacts:** large files on Hugging Face Hub, not git ([`.gitignore`](.gitignore)). **Secrets:** `HF_TOKEN` (Hub), `OPENAI_API_KEY` (autotune).

```bash
python3 scripts/pipeline.py --all --profile auth
python3 scripts/pipeline.py --all --family lfm2 --profile auth
python3 scripts/pipeline.py --preflight --dry-run
```

After `export-merged` on the Space, local GGUF (typical): `./scripts/local_gguf_export.sh <profile>`. **Gradio:** [`app.py`](app.py) exposes the same steps (including merge-from-`data/next/`).

## Training

| Item | Value |
| --- | --- |
| Bases | `qwen`: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) / [14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct); `lfm2`: [unsloth/LFM2.5-1.2B-Instruct](https://huggingface.co/unsloth/LFM2.5-1.2B-Instruct) / [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B) |
| LoRA / LR / batch | Defaults in [`config.py`](config.py) `_REGISTRY`: **qwen** small/auth use r=32, α=64, LR `5e-5` / `2e-5`. **lfm2** small: r=16, α=32, LR `1e-4`, `per_device_batch` 2, `grad_accum` 8 (effective batch 16). **lfm2** auth: r=32, α=64, LR `2e-5`. Override with `SMALL_*` / `AUTH_*` env. |
| Other | Max seq 2048, bf16, cosine schedule, q/k/v/o + MLP targets (see `train.py`). Stack: Unsloth `FastLanguageModel` + TRL `SFTTrainer`. |

```bash
pip install -r requirements.txt
export HF_TOKEN=...

python3 train.py
MODEL_PROFILE=auth python3 train.py
LLM_FAMILY=lfm2 python3 train.py
LLM_FAMILY=lfm2 MODEL_PROFILE=auth python3 train.py
```

**Hub IDs:** Unsloth’s loader expects checkpoints it supports; `unsloth/*` repos are the supported builds. Upstream `LiquidAI/*` (or other) IDs on the same code path can raise `NotImplementedError` depending on Unsloth version. For plain Transformers/PEFT or for GGUF built outside Unsloth, use the license and files attached to the checkpoint you actually load. LFM weights: [LFM Open License v1.0](https://liquid.ai/lfm-license) (not Apache 2.0). Summary: [Attribution and licensing](#attribution-and-licensing).

**Auth tooling:** `train.py` merges `data/tooling_email_calendar.jsonl` with oversampling (`AUTH_TOOLING_MULTIPLIER`, default 3).

**Generation (eval / autotune / baseline):** `config.resolve_generation_kwargs()`: for `lfm2`, defaults are sampling-style (`temperature`, `top_p`, `top_k`, `repetition_penalty`); for `qwen` / `qwen3`, greedy unless env overrides. Env: `SMALL_TEMPERATURE`, `SMALL_TOP_P`, `SMALL_TOP_K`, `AUTH_*` analogues; `SMALL_TEMPERATURE=0` forces greedy LFM2. Training LR tweaks: use `SMALL_LEARNING_RATE` etc.; eval/train checkpoint selection stays in `train.py` / `retrain_step.py`.

## Hugging Face Space

**App:** [spaces/jeanbaptdzd/sft-wagmi](https://huggingface.co/spaces/jeanbaptdzd/sft-wagmi) (Gradio on port 7860). Git remote for pushes: `https://huggingface.co/spaces/jeanbaptdzd/sft-wagmi`. If the Space is **private**, unauthenticated HTTP checks return 401; open it while logged into Hugging Face.

After a git push to the Space branch, use a **factory rebuild** (not only Restart) so the image includes the current tree and `/app/VERSION` matches [`VERSION`](VERSION). Verify with `cat /app/VERSION` over SSH or logs. Optional: `bash scripts/hf_space_self_check.sh` from repo root.

**LFM2 small (1.2B Instruct), sequential commands:**

```bash
export LLM_FAMILY=lfm2 MODEL_PROFILE=small
python3 train.py
python3 eval_sft.py && python3 eval_sft_rag.py && python3 eval_redteam.py
# or: python3 scripts/pipeline.py --all --family lfm2 --profile small
```

## Cursor SDK recurring automations

The repo includes Cursor SDK scripts in `automation/cursor-sdk` for recurring maintenance and recurring training orchestration. Space-side / HF Jobs behaviour, `recurring_runner.py`, and **`configs/recurring_runs.json`** are documented in [`scripts/hf/README.md`](scripts/hf/README.md). Cadence and scope notes live in [`docs/RECURRING_TRAINING_PLAN.md`](docs/RECURRING_TRAINING_PLAN.md).

```bash
cd automation/cursor-sdk
npm install
```

HouseKeeper (docs and README hygiene):

```bash
CURSOR_API_KEY=... npm run run:housekeeper
CURSOR_API_KEY=... npm run run:housekeeper -- --area docs --instruction "focus on stale runbooks first"
```

Recurring training orchestrator trigger (optional **`--config`**, path relative to repo root; default **`configs/recurring_runs.json`**):

```bash
CURSOR_API_KEY=... npm run run:recurring -- --cadence daily --trigger scheduler
CURSOR_API_KEY=... npm run run:recurring -- --cadence weekly --trigger scheduler
```

GitHub Actions: [`.github/workflows/recurring-training-cursor-sdk.yml`](.github/workflows/recurring-training-cursor-sdk.yml).

Suggested cron shape (outside repo):

```bash
# nightly docs housekeeping
0 2 * * * cd /path/to/sft-wagmi/automation/cursor-sdk && CURSOR_API_KEY=... npm run run:housekeeper -- --area .
```

## Autotune

`autotune.py`: generate on eval set → GPT-4o scores six criteria (0–3) → corrector for failures → merge → retrain → Hub push; repeat until mean score > 2.5/3 or three iterations. Requires `OPENAI_API_KEY` and `HF_TOKEN`.

## Export (merged model, GGUF, Ollama)

1. **Space:** `export_merged.py` (LoRA merged to BF16, push Hub). Pipeline: `--export-merged` (included in `--all`).
2. **Local:** `scripts/local_gguf_export.sh`: pull merged weights, `convert_hf_to_gguf.py` + `llama-quantize` from llama.cpp, Q4_K_M, push GGUF + Modelfile.
3. **Optional:** `export_gguf.py` on-GPU merge+GGUF; pipeline `--export-gguf`.

**Dexm env names:** after `ollama create`, tags such as `wagmi-sft:latest` / `wagmi-sft-14b:latest` map to `LLM_MODEL` / `LLM_MODEL_AUTH`. Override in dexm and in this repo via `OLLAMA_MODEL` / `OLLAMA_MODEL_NAME` if your tags differ.

Modelfile generation uses [`scripts/ollama_qwen25_instruct_template.gotmpl`](scripts/ollama_qwen25_instruct_template.gotmpl) so Qwen-class exports follow the same tool template family as upstream `qwen2.5:*-instruct` images.

**Local recreate:** need a `.gguf` on disk (Hub download or export scripts). Then:

```bash
chmod +x scripts/recreate_ollama_wagmi.sh scripts/smoke_ollama_tools.sh
./scripts/recreate_ollama_wagmi.sh small  /path/to/wagmi-qwen2.5-1.5b-sft.q4_k_m.gguf
./scripts/recreate_ollama_wagmi.sh auth   /path/to/wagmi-qwen2.5-14b-sft.q4_k_m.gguf
./scripts/smoke_ollama_tools.sh wagmi-sft:latest
```

`ollama rm` on a missing model exits with an error; do not delete an existing model until a replacement GGUF and Modelfile exist.

**llama.cpp serving (evaluation):** GGUF tooling is production; serving via llama.cpp’s HTTP `/v1` API instead of Ollama is not fully validated (tools, chat templates Qwen vs LFM2, deployment). Dexm switches with `LLM_RUNTIME=llamacpp` when endpoints are ready.

## Versioning

Semver in [`VERSION`](VERSION): MAJOR = persona/base/schema break; MINOR = dataset or capability change; PATCH = hyperparameters or tooling. Release flow: add JSONL under `data/next/` → [`scripts/merge_next.py`](scripts/merge_next.py) → `pipeline.py --all` → [CHANGELOG.md](CHANGELOG.md) entry → commit.

## Prompt philosophy and retrain checklist

The downstream site ([dexm-one-page](https://github.com/DealExMachina/dexm-one-page)) ships **slim** system prompts split by tier. Encyclopedic Deal ex Machina facts (services, blog, stack, partners) are not inlined — they flow through `buildLocalRagContext()` from `src/lib/chat/wagmi-skills.md`, `SKILLS.md`, and `public/ai.txt`. SFT examples should match that contract:

| Profile | Site tier | Tools in production | Behavioural target |
| --- | --- | --- | --- |
| `small` | CPU / anonymous | **None** (enforced server-side) | Safety-first, factual, ~100 words, no tool JSON, point users to the on-page sign-in panel for auth. |
| `auth` | GPU / authenticated | `auth-user`, `email.send` (connected user only), `calendar.create_event` (JB only) | Fuller DxM knowledge in concise paragraphs, structured tool-call examples where present, same safety floor. |

Retrain checklist when a prompt-philosophy change lands in dexm-one-page:

1. Confirm `data/train.jsonl` / `data/eval.jsonl` system messages match the slim canonical form (no encyclopedic dumps).
2. For `small` profile rows: assistant turns must never call tools. Add hard-negatives in `data/next/` for any leaked tool JSON.
3. For `auth` profile rows: keep `data/tooling_email_calendar.jsonl` as the only path that produces tool calls; oversample via `AUTH_TOOLING_MULTIPLIER`.
4. Run `python3 scripts/pipeline.py --all --profile small` then `--profile auth` (per family if both `qwen` and `lfm2` ship).
5. Red-team: `pipeline.py --redteam` for both profiles; verify the small profile fails closed when prompted to invent tool calls or emails.
6. Bump `VERSION` (MINOR for behavioural change), update `CHANGELOG.md`, and tag the dexm-one-page commit that consumes the new tags.

## Red-team reports

```bash
python3 scripts/pipeline.py --profile auth --redteam
python3 scripts/pipeline.py --profile small --redteam
```

Outputs: `reports/redteam/v<version>/<profile>_redteam_<timestamp>.{json,md}` (release-gate style summary for internal traceability).

## Repository layout

```text
sft-wagmi/
├── data/                 train.jsonl, eval.jsonl, metadata.json, tooling_email_calendar.jsonl, next/
├── docs/                 HF model-card runbook; recurring-training overview (`RECURRING_TRAINING_PLAN.md`)
├── scripts/              pipeline.py, merge_next.py, export_ollama.py, local_gguf_export.sh,
│                         prepare_hf_model_cards_ai_act.py, hf_space_self_check.sh, …
├── train.py, autotune.py, eval_*.py, export_*.py, baseline.py, retrain_step.py, app.py
├── Dockerfile, docker-entrypoint.sh, requirements.txt, VERSION, CHANGELOG.md
└── reports/redteam/
```

## Attribution and licensing

Upstream licenses differ. **LFM** checkpoints (including `unsloth/` mirrors) are under **[LFM Open License v1.0](https://liquid.ai/lfm-license)** (Liquid AI; not interchangeable with Apache 2.0). **Qwen** bases used here are **Apache 2.0**. Read the license text and each Hub repo’s `LICENSE` before redistribution of weights, merges, or GGUF. Derivatives you train are still subject to the base model license. This section is informational, not legal advice.

| Component | License | References |
| --- | --- | --- |
| Qwen2.5-1.5B / 14B-Instruct | Apache 2.0 | [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| LFM2.5-1.2B (incl. `unsloth/` mirror) | LFM Open License v1.0 | [Terms](https://liquid.ai/lfm-license); [unsloth](https://huggingface.co/unsloth/LFM2.5-1.2B-Instruct), [LiquidAI](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| LFM2-24B-A2B | LFM Open License v1.0 | [Terms](https://liquid.ai/lfm-license); [LiquidAI](https://huggingface.co/LiquidAI/LFM2-24B-A2B) |
| Unsloth, TRL | Apache 2.0 | GitHub repos linked above |
| llama.cpp, Ollama | MIT | Project repos |
| GPT-4o (autotune) | Proprietary | OpenAI |

Dataset, adapters, and merge artifacts: Deal ex Machina; distribution rules follow the base license and your Hub terms of use.

## Compliance (summary)

Wagmi in production is a limited-scope chatbot (EU AI Act transparency tier; not high-risk as deployed). Internal practices include versioned training, red-team reports, no user chats in SFT data, RAG + SFT + optional autotune for quality. **Gaps:** no standalone formal risk register; release-time testing only; finite red-team vectors; incomplete public model cards; inference subprocessors and geography not fully contractually closed (see deployment docs on dexm / Koyeb).

AI Act model-card preparation workflow: [`docs/HF_MODEL_CARD_AI_ACT_RUNBOOK.md`](docs/HF_MODEL_CARD_AI_ACT_RUNBOOK.md) and `python3 scripts/prepare_hf_model_cards_ai_act.py`.

**GDPR (summary):** chat is positioned as voluntary information; no training on user messages; limited persistence by product design. **Gaps:** no published DPIA; infrastructure logging and DPA coverage for inference hosts not fully documented; autotune code path must stay isolated from live user traffic.

## Related

- [dexm-one-page](https://github.com/DealExMachina/dexm-one-page) (site, `scripts/generate-wagmi-sft-dataset.ts`)
- [FR: Dresser un petit modèle sur CPU](https://dealexmachina.com/fr/blog/dresser-petit-modele-cpu)
- [FR: Pipeline SFT](https://dealexmachina.com/fr/blog/2026-04-06-sft-wagmi-rudimentary-pipeline-fr)
