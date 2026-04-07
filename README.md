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
  <a href="https://github.com/unslothai/unsloth"><img src="https://img.shields.io/badge/Training-Unsloth-orange?logo=github" alt="Unsloth"></a>
  <a href="https://huggingface.co/jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"><img src="https://img.shields.io/badge/HuggingFace-Adapter-yellow?logo=huggingface" alt="HuggingFace"></a>
  <a href="https://ollama.com"><img src="https://img.shields.io/badge/Inference-Ollama-black?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9IndoaXRlIi8+PC9zdmc+" alt="Ollama"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/Judge-GPT--4o-412991?logo=openai&logoColor=white" alt="GPT-4o"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</p>

---

**Current version: `0.2.0`** (see [CHANGELOG.md](CHANGELOG.md))

Fine-tune [Qwen 2.5](https://huggingface.co/Qwen) into **Wagmi**: a quantized assistant that answers questions about [Deal ex Machina](https://dealexmachina.com) in French and English, with guardrails against hallucination, prompt injection, and harmful content.

Two profiles ship today:

- **small** (1.5B) -- CPU-friendly, ~1 GB Q4_K_M.
- **auth** (14B) -- GPU-deployed on Koyeb, ~8.4 GB Q4_K_M, with tool-calling capabilities.

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

Set `HF_TOKEN`, then:

```bash
python3 train.py                           # small profile (default)
MODEL_PROFILE=auth python3 train.py        # auth profile (14B)
```

The adapter is pushed to Hugging Face Hub with a version-tagged commit message.

## Autotune loop

`autotune.py` implements the self-improvement loop:

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

The default pipeline uses a **hybrid** approach:

1. **On HF Space (GPU):** `export_merged.py` merges the LoRA adapter into a
   full BF16 model and pushes it to Hub.
2. **Locally (Mac/Linux):** `scripts/local_gguf_export.sh` downloads the merged
   model, converts to GGUF with llama.cpp, quantizes to Q4_K_M, and pushes the
   GGUF + Modelfile back to Hub.

```bash
# After training completes on Space:
./scripts/local_gguf_export.sh auth
```

**Alternative (all-in-one on GPU):** `export_gguf.py` does merge + GGUF + push
in a single step on the Space, useful when llama.cpp is available in the
container. Invoked via `--export-gguf` in the pipeline.

For auth profile tool-calling specialization (email + calendar), `train.py`
auto-injects `data/tooling_email_calendar.jsonl` with 3x oversampling.

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

1. Drop new `.jsonl` files into `data/next/` during development (see `data/next/README.md`
   for the required schema).
2. Merge them with `python3 scripts/merge_next.py --bump minor` (validates schema,
   appends to train/eval, updates metadata, bumps VERSION).
3. Run the full pipeline: `python3 scripts/pipeline.py --all --profile auth`.
4. Add a `CHANGELOG.md` entry, commit, and push.

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
├── baseline.py                      # Baseline eval (pre-SFT)
├── train.py                         # Unsloth SFT
├── autotune.py                      # GPT-4o judge-correct-retrain loop
├── eval_sft.py                      # Post-training eval
├── eval_sft_rag.py                  # Eval with RAG context
├── eval_tool_calls.py               # Tool-calling eval (auth profile)
├── export_gguf.py                   # All-in-one GGUF export on Space (optional)
├── export_merged.py                 # LoRA merge + push merged model (default)
├── retrain_step.py                  # Helper for autotune merge/retrain
├── prompt_encode.py                 # Chat template utilities for baseline
├── app.py                           # Gradio Space entry
├── Dockerfile                       # CUDA + deps image for HF Spaces
├── docker-entrypoint.sh             # Container init (HOME, cache dirs)
├── requirements.txt
├── VERSION                          # Semver (current: 0.2.0)
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

## EU AI Act compliance

Wagmi is a **limited-risk AI system** under the [EU AI Act](https://artificialintelligenceact.eu/)
(Regulation 2024/1689). It is a customer-facing chatbot embedded in a commercial website, which
places it in the transparency-obligation tier (Article 50). It is not a high-risk system: it does
not make decisions affecting health, safety, employment, creditworthiness, or law enforcement.

### What we do today

| Principle | Implementation |
| --- | --- |
| **Transparency** | The chat widget explicitly identifies itself as an AI assistant ("I am Wagmi, the AI assistant for Deal ex Machina"). Every response carries this persona framing. The site footer links to the AI disclosure page (`/ai.txt`). |
| **Human oversight** | Wagmi operates in an advisory capacity only. It cannot trigger transactions, modify user data, or access backend systems without human-in-the-loop approval. Tool-calling (email, calendar) in the auth profile requires authenticated operator context. |
| **Robustness & security** | v0.2.0 adds 121 adversarial training entries covering OWASP LLM Top 10 categories. A 28-vector security smoke test runs at each release. The auth model (14B) achieved 0 FAIL / 0 WARN on the full suite. Prompt injection, social engineering, encoding obfuscation, and multi-turn deception vectors are explicitly trained against. |
| **Data governance** | Training data is sourced exclusively from Deal ex Machina's own content (blog, site pages, Obsidian notes) and hand-crafted synthetic examples. No user conversations are used for training. Dataset provenance is tracked in `data/metadata.json` with per-entry `source` and `id` fields. |
| **Accuracy & hallucination control** | Three-layer defence: RAG injects verified facts at inference time, SFT teaches uncertainty phrasing ("I don't have that information"), and the autotune loop penalises hallucination via GPT-4o scoring. |
| **Versioning & traceability** | Every model version is semantically versioned (`VERSION`), logged in `CHANGELOG.md`, and tagged in Hub commit messages. Training artifacts (adapters, merged models, GGUF) are stored on Hugging Face Hub with version-stamped commits. |
| **Bias mitigation** | Bilingual (FR/EN) dataset with balanced coverage. Guardrail entries explicitly train refusal of discriminatory, violent, or illegal content in both languages. |

### Honest gaps

- **No formal risk assessment document.** The AI Act will require a structured impact assessment
  for transparency-tier systems by 2027. We have smoke tests and changelogs, but no standalone
  risk register.
- **No automated monitoring in production.** We test at release time, not continuously. A
  production logging + flagging pipeline (e.g. Langfuse traces with anomaly detection) is needed.
- **Adversarial coverage is finite.** 28 attack vectors is a strong start, but new jailbreak
  techniques emerge regularly. The test suite must grow with each release.
- **No external audit.** All testing is internal. Third-party red-teaming would strengthen
  confidence in robustness claims.
- **Model card is incomplete.** Hugging Face model cards for the adapter and GGUF repos should
  document intended use, limitations, and evaluation results per the AI Act transparency
  requirements.

### Improvement path

1. Publish a formal **model card** on each Hugging Face repo (adapter, merged, GGUF) with
   intended use, limitations, evaluation metrics, and training data summary.
2. Add **production observability** (Langfuse or equivalent) to log inference traces, flag
   anomalous responses, and feed findings back into the training pipeline.
3. Maintain a **risk register** that maps each identified threat to its mitigation and test
   coverage.
4. Schedule periodic **external red-teaming** sessions, at minimum before major version bumps.
5. Expand the security smoke test to **50+ vectors** covering emerging attack patterns.

## GDPR compliance

Wagmi processes user messages at inference time to generate responses. This section documents
how the system relates to the [General Data Protection Regulation](https://gdpr.eu/)
(Regulation 2016/679), with an honest assessment of current state and gaps.

### What we do today

| Principle | Implementation |
| --- | --- |
| **Lawful basis** | The chat widget operates under **legitimate interest** (Art. 6(1)(f)): visitors initiate conversations voluntarily, and the assistant provides information about Deal ex Machina's services. No account creation or personal data collection is required to use the chat. |
| **Data minimisation** | Wagmi does not store conversation history server-side beyond the active session. Messages are held in browser memory and sent to the inference endpoint (Koyeb Ollama) per request. No conversation logs are persisted to disk or database in production. |
| **No training on user data** | The SFT dataset is composed exclusively of company-owned content and hand-crafted synthetic examples. User conversations are never fed back into training. This is a hard architectural boundary, not a policy toggle. |
| **No profiling** | Wagmi does not build user profiles, track users across sessions, or make automated decisions with legal or significant effects. |
| **Transparency** | The AI disclosure page (`/ai.txt`) and chat widget identify the system as AI-powered. Users know they are interacting with an automated system, not a human. |
| **Sub-processors** | Inference runs on [Koyeb](https://koyeb.com) (EU-available infrastructure). The autotune judge uses OpenAI GPT-4o (US-based), but only on synthetic eval data, never on user content. Hugging Face Hub stores model weights, not user data. |

### Honest gaps

- **No Data Protection Impact Assessment (DPIA).** While Wagmi is unlikely to trigger the DPIA
  threshold (no large-scale processing of personal data, no systematic monitoring), a lightweight
  DPIA would formalise the analysis and satisfy accountability obligations under Art. 35.
- **Inference endpoint logging is opaque.** Koyeb may retain request logs containing user messages
  at the infrastructure level. We have not audited Koyeb's data retention policy for GPU inference
  endpoints, nor established a Data Processing Agreement (DPA) specific to this workload.
- **No explicit consent mechanism.** The chat widget does not present a GDPR consent banner before
  the first message. While legitimate interest may suffice for a voluntary informational chatbot,
  a clear notice ("Your messages are processed by AI and not stored") would strengthen compliance.
- **No data subject rights workflow.** There is no self-service mechanism for users to request
  access to, correction of, or deletion of their chat data. Since we do not persist conversations,
  this is technically moot -- but the absence of documentation explaining this creates a
  transparency gap.
- **Cross-border data flows.** If Koyeb routes inference to non-EU nodes, user messages would
  transit outside the EEA. We have not verified the geographic pinning of our GPU instance or
  established Standard Contractual Clauses (SCCs) for this path.
- **Autotune sub-processor.** The GPT-4o judge processes synthetic data only, but the OpenAI API
  call path exists in production code. A misconfiguration could theoretically route user content
  to OpenAI. There is no runtime guardrail preventing this beyond code review.

### Improvement path

1. **Audit Koyeb's DPA** and verify that the GPU inference endpoint runs in an EU region.
   Establish SCCs or adequacy-basis documentation if it does not.
2. **Add a pre-chat notice** to the widget: "This is an AI assistant. Your messages are processed
   to generate responses and are not stored or used for training."
3. **Write a lightweight DPIA** covering the inference data flow (browser -> Koyeb -> Ollama ->
   response), even if the conclusion is that no high risk is identified.
4. **Document the "no persistence" architecture** in a public privacy notice, so data subjects
   understand that no conversation data is retained and therefore access/deletion requests are
   satisfied by design.
5. **Add a runtime assertion** in the inference path that prevents user-supplied content from
   being forwarded to external APIs (OpenAI, Anthropic) outside of explicitly operator-triggered
   autotune sessions.
6. **Review the cookie/local-storage footprint** of the chat widget to confirm it does not
   create persistent identifiers that would require ePrivacy consent.

## Related

- [dexm-one-page](https://github.com/DealExMachina/dexm-one-page) -- production site and `scripts/generate-wagmi-sft-dataset.ts`
- [Dresser un petit modele sur CPU](https://dealexmachina.com/fr/blog/dresser-petit-modele-cpu) -- long-form write-up (FR)
- [SFT Wagmi, pipeline rudimentaire](https://dealexmachina.com/fr/blog/2026-04-06-sft-wagmi-rudimentary-pipeline-fr) -- pipeline notes (FR)

---

<sub>Built with [Cursor](https://cursor.com) by [Deal ex Machina](https://dealexmachina.com) -- *The optimal path between vision and results.*</sub>
