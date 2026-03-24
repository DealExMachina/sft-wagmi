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

Fine-tune [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) into **Wagmi** -- a small model (986 MB quantized) that runs on CPU and answers questions about [Deal ex Machina](https://dealexmachina.com), its services, its blog, and its founder. In French and English, with guardrails against hallucination.

The pipeline has three layers:

1. **RAG** -- BM25-style local retrieval injects verified facts into the system prompt at inference time ([`local-rag.ts`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/local-rag.ts))
2. **SFT** -- 568 bilingual training examples generated from site content teach identity, tone, uncertainty, and refusal reflexes
3. **Autotune** -- iterative judge-correct-retrain loop with GPT-4o scoring on 6 criteria until convergence

## Dataset

| | Train | Eval | Total |
| --- | --- | --- | --- |
| Examples | 483 | 85 | **568** |
| EN | 285 | | |
| FR | 283 | | |

Sources: 9 blog posts (bilingual), [`wagmi-skills.md`](https://github.com/DealExMachina/dexm-one-page/blob/dev/src/lib/chat/wagmi-skills.md), [`ai.txt`](https://github.com/DealExMachina/dexm-one-page/blob/dev/public/ai.txt).

| Category | Count | Description |
| --- | --- | --- |
| Content-grounded | 368 | Summaries and citations from chunked site content |
| Guardrails | 112 | Identity, uncertainty, refusal, anti-hallucination |
| Direct Q&A | 62 | Company, services, founder, tech stack, contact |
| Cross-lingual | 54 | FR prompt frames on EN-only sources |
| Hard negatives | 18 | Corrections of observed v1 hallucinations |
| Multi-turn | 8 | Dialogue coherence across turns |

Generated from [dexm-one-page](https://github.com/DealExMachina/dexm-one-page) via:

```bash
npx tsx scripts/generate-wagmi-sft-dataset.ts
```

## Training

| Parameter | Value |
| --- | --- |
| Base model | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Method | LoRA (rank 32, alpha 64) |
| Target modules | q/k/v/o + gate/up/down proj |
| Max seq length | 2048 tokens |
| Learning rate | 2e-4 (cosine decay) |
| Epochs | 2 |
| Effective batch | 16 (4 per device, grad accum 4) |
| Precision | bf16 |
| Framework | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |
| Infrastructure | HuggingFace L40 (48 GB VRAM) |

### Run training

```bash
pip install -r requirements.txt
```

Set `HF_TOKEN` in environment, then execute `train.ipynb` top-to-bottom on a CUDA host (HF Spaces Gradio, Colab, or bare metal).

The adapter is pushed to [`jeanbaptdzd/wagmi-qwen2.5-1.5b-sft`](https://huggingface.co/jeanbaptdzd/wagmi-qwen2.5-1.5b-sft) on the Hub.

## Autotune loop

`autotune.ipynb` runs a Karpathy-style self-improvement loop:

```
[SFT model] --> inference on 50+ eval prompts (with RAG context)
     |
     v
[GPT-4o judge] --> scores on 6 criteria (0-3 each)
     |              factual_accuracy, language_match, tone_persona,
     |              guardrail_compliance, conciseness, hallucination_free
     v
[GPT-4o corrector] --> ideal responses for failures (total < 14/18)
     |
     v
[Merge corrections] --> retrain with Unsloth --> push to Hub
     |
     v
[Repeat] --> until mean score > 2.5/3.0 or 3 iterations
```

Requires `OPENAI_API_KEY` and `HF_TOKEN` in environment.

## Export to Ollama

After training, the LoRA adapter lives on the Hub. One local script handles the full export:

```bash
pip install torch transformers peft huggingface_hub
HF_TOKEN=hf_xxx python3.11 scripts/export_ollama.py
```

Pipeline: download adapter --> merge LoRA into base on CPU (~6 GB RAM) --> convert to GGUF via [llama.cpp](https://github.com/ggerganov/llama.cpp) --> quantize Q4_K_M --> create Ollama model with ChatML template.

Requires [Ollama](https://ollama.com) >= 0.5 and `brew install llama.cpp`.

```bash
ollama run wagmi-sft
```

Final model: **986 MB** (Q4_K_M), runs on any CPU.

## Regenerating the dataset

When site content changes:

```bash
cd ../dexm-one-page
npx tsx scripts/generate-wagmi-sft-dataset.ts
cp datasets/wagmi-sft/*.jsonl ../sft-wagmi/data/
cp datasets/wagmi-sft/metadata.json ../sft-wagmi/data/
```

## Repository structure

```
sft-wagmi/
├── data/
│   ├── train.jsonl              # 483 training examples (JSONL chat format)
│   ├── eval.jsonl               # 85 evaluation examples
│   └── metadata.json            # generation stats and distribution
├── scripts/
│   └── export_ollama.py         # merge + GGUF convert + Ollama import
├── train.ipynb                  # Unsloth SFT notebook
├── baseline.ipynb               # pre-training baseline evaluation
├── autotune.ipynb               # judge-correct-retrain loop
├── requirements.txt             # Python dependencies
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

The LoRA adapter and SFT dataset are original work by [Deal ex Machina](https://dealexmachina.com). The base model architecture and weights remain the intellectual property of the Qwen team at Alibaba Cloud.

## Related

- [dexm-one-page](https://github.com/DealExMachina/dexm-one-page) -- the website that serves Wagmi
- [Blog post: Dresser un petit modele sur CPU](https://dealexmachina.com/blog/dresser-petit-modele-cpu) -- full write-up of this pipeline (FR)

---

<sub>Built with [Cursor](https://cursor.com) by [Deal ex Machina](https://dealexmachina.com) -- *The optimal path between vision and results.*</sub>
