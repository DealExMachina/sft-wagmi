# Changelog

Semver: `MAJOR.MINOR.PATCH`. **MAJOR:** persona / base model / schema break. **MINOR:** dataset, profile, or capability change. **PATCH:** hyperparameters, bugfix, tooling-only.

---

## 0.3.4 -- 2026-04-18

- `config.resolve_generation_kwargs()`: for `lfm2`, scripted `generate()` uses non-greedy defaults (`temperature`, `top_p`, `top_k`, `repetition_penalty`); `qwen` / `qwen3` stay greedy unless env overrides. Used from autotune, `eval_*`, `baseline.py`.
- `config.py` **lfm2 small** registry: LoRA r=16, α=32, LR `1e-4`, `per_device_batch` 2, `grad_accum` 8 (comments reference Liquid Unsloth fine-tuning guidance).
- TRL: `SFTTrainer` compatibility fix for tokenizer vs `processing_class` API drift.
- Docs: [`docs/HF_MODEL_CARD_AI_ACT_RUNBOOK.md`](docs/HF_MODEL_CARD_AI_ACT_RUNBOOK.md), `scripts/prepare_hf_model_cards_ai_act.py`.
- README: HF Space factory rebuild and `VERSION` check, `hf_space_self_check.sh`, LFM2 run commands, training table aligned with `_REGISTRY`; `data/metadata.json` `version` aligned with repo `VERSION` for this snapshot.

## 0.3.1 -- 2026-04-18

- README: licensing (LFM Open License vs Apache 2.0), pipeline and export consolidated; version fields aligned to `0.3.1` (corrects stray `0.3.2` on `VERSION` without changelog).
- No change to default training hyperparameters.

## 0.3.0 -- 2026-04-17

- `LLM_FAMILY=lfm2` in `config.py` / `scripts/pipeline.py` (`--family lfm2`): bases `unsloth/LFM2.5-1.2B-Instruct`, `unsloth/LFM2-8B-A1B`; Hub adapter/merged/GGUF IDs; Ollama names in registry.
- `config.py`: `ProfileConfig`; env overrides unchanged in spirit; autotune passes family/profile into `retrain_step` before Unsloth import; post-retrain adapter validation.
- Gradio Space: family selector; HF entry/version banner; SSH self-check; Gradio `app_port` alignment with README frontmatter.
- `eval_redteam.py`, `data/redteam_guardrail_cases.json`, versioned outputs under `reports/redteam/v<version>/`; `pipeline.py --redteam` and `--all` include red-team.
- Dataset row counts in `data/metadata.json` reflect merges into `data/*.jsonl` (e.g. 773 train / 136 eval in the snapshot at release; varies with `data/next/`).

## 0.2.0 -- 2026-04-07

- Dataset: +121 rows from `data/next/`, nine adversarial categories (injection, obfuscation, social engineering, multi-turn, abuse, harmful refusal, code refusal, etc.), FR/EN; total ~900 rows after merge.
- Smoke suite (28 vectors), excerpt:

| Model | PASS | SOFT | WARN | FAIL |
| --- | ---: | ---: | ---: | ---: |
| v0.1.0 auth (baseline) | 12 | 12 | 4 | 2 |
| v0.2.0 small (1.5B) | 23 | 3 | 1 | 1 |
| v0.2.0 auth (14B) | 24 | 4 | 0 | 0 |
- Pipeline: Gradio full-pipeline control; `scripts/merge_next.py`; version-aware logging in `train.py`, `export_merged.py`, `pipeline.py`.
- **Residual:** small model still weak on plain code requests and some urgency scenarios (track in later dataset work).

## 0.1.0 -- 2026-04-07

- Models: Qwen2.5 1.5B / 14B-Instruct, LoRA 32/64, Q4_K_M GGUF; Koyeb Ollama for auth path.
- Dataset: 780 examples (663 train / 117 eval), 29 sources; tooling file for auth with 3× oversampling.
- Pipeline: HF Space merge + local GGUF (llama.cpp); autotune (GPT-4o, six scores, threshold 2.5/3); `jeanbapt/ollama-wagmi` image.
- Smoke: 20/20 on identity, factuality, refusal, multilingual spot checks; ~14 tok/s auth Q4_K_M on 16 GB Mac.
