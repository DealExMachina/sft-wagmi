# Hugging Face Space helpers

The Space **Docker** image runs [`app.py`](../../app.py) (Gradio on `app_port` / `PORT`). **Batch scripts** below are separate: run them in the same container from a **shell** (`ssh.hf.space`, Cursor Remote-SSH, or Space terminal) with `cd /app` — they do not require the Gradio process to be up or healthy.

### HF MCP and `hf` CLI (expectations)

| Mechanism | What it can do |
| --- | --- |
| **Hub MCP** (`hub_repo_search`, `hub_repo_details`, …) | Find repos, metadata, links. **No** shell or `eval_redteam.py` execution inside your Space. |
| **`dynamic_space` MCP** | Invoke **catalog** Gradio Spaces (image gen, etc.). **Not** your private training Space. |
| **`hf` on your laptop** | `hf download`, `hf jobs run`, `hf auth`, … — talks to the **Hub API**. It does **not** SSH into a running Space or inject commands into the L40 pod. |
| **Space secrets** (`HF_TOKEN`, `OPENAI_API_KEY`, …) | Injected into the **Space container** environment. Any process you start there (SSH shell, `python3 eval_redteam.py`) inherits them — you usually **do not** need to `export` them again. |
| **[`hf jobs run`](https://huggingface.co/docs/huggingface_hub/guides/jobs)** | Runs a **separate** Hub Job (pick `--flavor l40sx1`, pass `--secrets HF_TOKEN` from your machine). Same **image**/deps as the Space unless you build one; not the same VM as the Space, but valid for batch eval if you script clone + install + `python3 eval_redteam.py`. |

**`eval_redteam.py`:** loads adapters with `HF_TOKEN` when hitting private Hub repos. **`OPENAI_API_KEY`** is **not** read by `eval_redteam` (used by `autotune.py` / pipeline). API-based eval uses **`EVAL_API_BASE_URL` / `EVAL_API_KEY` / `EVAL_API_MODEL`** only (see below).

## `run_redteam_auth_l40.sh`

Runs [`eval_redteam.py`](../../eval_redteam.py) twice (auth profile only):

1. `LLM_FAMILY=qwen` — Qwen2.5-14B + Wagmi auth adapter (local `output/...` or Hub id from `config.py`).
2. `LLM_FAMILY=lfm2` — LFM2-8B-A1B + Wagmi LFM2 auth adapter.

Requires **CUDA** (e.g. L40 on the Space). Set `HF_TOKEN` if `FastLanguageModel.from_pretrained` must pull a private adapter.

Artifacts: `reports/redteam/v<VERSION>/auth_redteam_<timestamp>.{json,md}`.

## Remote OpenAI-compatible mode (optional)

If the Space has no GPU but can reach Ollama / llama.cpp with OpenAI API:

```bash
export EVAL_API_BASE_URL="http://127.0.0.1:11434/v1"
export EVAL_API_KEY="ollama"
export EVAL_API_MODEL="wagmi-sft-14b:latest"
export LLM_FAMILY=qwen
export MODEL_PROFILE=auth
python3 eval_redteam.py
```

See [`eval_openai_compat.py`](../eval_openai_compat.py).

## Dexm Next.js `/api/chat` smoke (garde-fous route + JSON refus)

From any machine with network access to staging/prod:

```bash
export CHAT_API_BASE_URL="https://your-pages-or-staging.host"
python3 scripts/redteam_dexm_chat_api_smoke.py --max-cases 8
```

Uses [`guardrail_checks.py`](../../guardrail_checks.py) (same assertions as `eval_redteam`). Streamed LLM replies are checked heuristically on the raw stream body (line-oriented UI protocol); JSON `action_refused` / `blocked` bodies are preferred for deterministic cases.

## Recurring runner (Phase 2)

Use [`recurring_runner.py`](./recurring_runner.py) to orchestrate regular runs directly from the Space shell:

```bash
cd /app
python3 scripts/hf/recurring_runner.py --cadence daily --trigger manual
python3 scripts/hf/recurring_runner.py --cadence weekly --trigger manual
python3 scripts/hf/recurring_runner.py --run-id weekly-qwen-auth --trigger manual
```

Configuration lives in [`configs/recurring_runs.json`](../../configs/recurring_runs.json):

- `runs[].cadence`: `daily` or `weekly`
- `runs[].family` / `runs[].profile`: fed to `scripts/pipeline.py`
- `runs[].merge_next`: `auto|always|never`
- `runs[].skip_if_no_pending_data`: avoid expensive runs when `data/next/*.jsonl` is empty
- `runs[].timeout_minutes`: hard timeout per matrix run
- `runs[].env`: run-specific environment overrides
- `runs[].continue_on_failure`: stop matrix on critical run failure (unless `--allow-partial`)

Failure categorization:

- `infra`: OOM, timeout, network/runtime infra symptoms
- `data`: dataset/schema/JSON/missing-input failures
- `model_quality`: redteam/guardrail/eval quality-gate-type failures
- `unknown`: non-zero exit without a recognized signature

Reporting:

- Logs + summaries are written under `reports/recurring/<timestamp>/`
- If `gh` and `GH_TOKEN` (or `GITHUB_TOKEN`) are available, a comment is posted to the configured tracking issue (`issue_number`)
- Otherwise, the issue comment draft is saved locally in the same report folder

Backend mode:

- `backend=local`: execute `scripts/pipeline.py` directly in current environment
- `backend=hf_jobs`: submit `hf jobs run` using configured image/flavor/secrets (default in `configs/recurring_runs.json`)

## Cursor SDK trigger (no manual Space shell)

To avoid manual execution in the Space shell, this repo includes a Cursor SDK launcher:

- `automation/cursor-sdk/src/run-recurring.ts`
- GitHub Actions schedule: `.github/workflows/recurring-training-cursor-sdk.yml`

The SDK launcher asks a Cursor agent to execute:

```bash
python3 scripts/hf/recurring_runner.py --cadence <daily|weekly> --trigger <...>
```

With current config, this submits **HF Jobs** against the Space image (`hf.co/spaces/jeanbaptdzd/sft-wagmi`) rather than running training locally.
