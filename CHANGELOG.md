# Changelog

All notable changes to the Wagmi SFT model are documented here.
Versioning follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking persona change, base model swap, or schema change.
- **MINOR**: new dataset entries, new training profiles, new capabilities.
- **PATCH**: hyperparameter tweaks, bug fixes, tooling improvements.

---

## 0.2.0 -- 2026-04-07

Security hardening release. Adds 121 adversarial training entries across
9 attack categories, significantly improving robustness against prompt
injection, social engineering, and harmful content generation.

### Dataset

- **+121 entries** merged from `data/next/` into the training set.
- 9 new categories: prompt injection resistance, encoding obfuscation,
  context manipulation, social engineering, information extraction,
  multi-turn deception, abuse resilience, harmful content refusal,
  code refusal.
- Bilingual (FR/EN) coverage for every category.
- Total dataset: ~900 examples after merge.

### Security smoke test (28 attack vectors)

| Model | PASS | SOFT | WARN | FAIL |
| --- | ---: | ---: | ---: | ---: |
| v0.1.0 auth (baseline) | 12 | 12 | 4 | 2 |
| v0.2.0 small (1.5B) | 23 | 3 | 1 | 1 |
| **v0.2.0 auth (14B)** | **24** | **4** | **0** | **0** |

Attack vectors tested: parroting, role-play hijack, fake system override,
Base64/ROT13/leetspeak obfuscation, "test mode" / "developer mode" pretexts,
flattery, urgency, child persona, system prompt extraction, training data
listing, PII requests, multi-turn escalation, direct insults, hostile
language, phishing/malware/scam generation, code generation refusal.

### Pipeline

- One-click "Full Pipeline" button in Gradio UI (`app.py`).
- `scripts/merge_next.py` for automated data staging with schema validation.
- Version-aware logging in `train.py`, `export_merged.py`, `pipeline.py`.

### Known gaps (targeted for v0.3.0)

- Small model (1.5B): fails on plain English code requests ("sort a list").
- Small model: weak urgency resistance under time pressure.

---

## 0.1.0 -- 2026-04-07

First versioned release.

### Model

- **small** profile: Qwen 2.5 1.5B-Instruct, LoRA r=32/a=64, Q4_K_M GGUF (~1 GB).
- **auth** profile: Qwen 2.5 14B-Instruct, LoRA r=32/a=64, Q4_K_M GGUF (~8.4 GB).
  Deployed locally and on Koyeb GPU Ollama.

### Dataset

- 780 examples (663 train / 117 eval), 29 sources.
- FR/EN bilingual (433 FR, 347 EN).
- Source mix: content, obsidian, synthetic guardrails, grounded QA,
  wagmi QA, hard negatives, multi-turn.
- Auth profile: +tooling examples (`tooling_email_calendar.jsonl`, 3x oversampling).

### Pipeline

- Hybrid export: LoRA merge on HF Space (L40), GGUF conversion local (llama.cpp).
- Autotune loop with GPT-4o judge (6 criteria, threshold 2.5/3.0).
- Custom Ollama Docker image for Koyeb deployment (`jeanbapt/ollama-wagmi`).

### Smoke test (20/20 passed)

- Identity, factual recall, uncertainty, refusal, jailbreak resistance,
  multilingual (FR/EN/ES), tone consistency, RAG context extraction.
- Performance: ~14 tok/s on 16 GB Mac (auth, Q4_K_M).
