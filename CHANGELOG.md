# Changelog

All notable changes to the Wagmi SFT model are documented here.
Versioning follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking persona change, base model swap, or schema change.
- **MINOR**: new dataset entries, new training profiles, new capabilities.
- **PATCH**: hyperparameter tweaks, bug fixes, tooling improvements.

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
