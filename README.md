# sft-wagmi

Supervised fine-tuning of **Qwen/Qwen2.5-1.5B-Instruct** on Deal ex Machina site content.  
Produces **Wagmi** — a grounded assistant that answers questions about the company, its blog, and its services, in French and English.

## Dataset

| Split | Examples |
|-------|----------|
| train | 374      |
| eval  | 66       |
| total | 440      |

Balanced EN/FR (222 / 218). Mix of:
- content-grounded summary & citation rows (blog posts + wagmi-skills + ai.txt)
- synthetic guardrail examples (identity, uncertainty, refusal, anti-hallucination)
- direct Q&A (company, services, founder, tech stack)
- multi-turn dialogues

Generated from [dexm-one-page](https://github.com/jeanbaptdzd/dexm-one-page) via:

```bash
npx tsx scripts/generate-wagmi-sft-dataset.ts
```

## Model config

| Parameter         | Value                          |
|-------------------|--------------------------------|
| Base model        | Qwen/Qwen2.5-1.5B-Instruct     |
| LoRA rank         | 32                             |
| LoRA alpha        | 64                             |
| Target modules    | q/k/v/o + gate/up/down proj    |
| Max seq length    | 2048                           |
| Learning rate     | 2e-4                           |
| Scheduler         | cosine                         |
| Epochs            | 2                              |
| Batch size        | 8 per device, grad accum 2     |
| Effective batch   | 16                             |
| Precision         | bf16 (no quantisation)         |
| Infra             | HuggingFace L40 (48 GB VRAM)   |

## Usage

### 1. Set up the environment

```bash
pip install -r requirements.txt
```

### 2. Configure

Open `train.ipynb` and adjust the variables at the top of **Cell 2**:

- `HUB_MODEL_ID` — your HuggingFace model repo (e.g. `yourname/wagmi-qwen2.5-1.5b-sft`)
- `PUSH_TO_HUB` — set to `True` to push the adapter automatically
- Set the `HF_TOKEN` environment variable (or run `huggingface-cli login`)
- Optionally set `WANDB_API_KEY` for W&B logging, or change `report_to` to `"none"`

### 3. Run

On HuggingFace Spaces (L40) or any CUDA host:

```
jupyter nbconvert --to notebook --execute train.ipynb --output train_executed.ipynb
```

Or open the notebook interactively and run top-to-bottom.

### 4. Output

- Local adapter: `wagmi-qwen2.5-1.5b-sft/`
- HF Hub: `https://huggingface.co/<HUB_MODEL_ID>` (private by default)

## Regenerating the dataset

When site content changes, regenerate from `dexm-one-page` and recopy:

```bash
cd ../dexm-one-page
npx tsx scripts/generate-wagmi-sft-dataset.ts
cp datasets/wagmi-sft/*.jsonl ../sft-wagmi/data/
cp datasets/wagmi-sft/metadata.json ../sft-wagmi/data/
```

## Autotune loop (Karpathy-style self-improvement)

`autotune.ipynb` runs an iterative improvement loop:

1. Inference on 50+ eval prompts (with RAG context)
2. Claude judge scores each response (6 criteria, 0-3 scale)
3. Claude corrector generates ideal responses for failures
4. Corrections merged into training set
5. Retrain with Unsloth, push to Hub
6. Repeat until convergence (mean score > 2.5/3.0 or 3 iterations)

Requires `ANTHROPIC_API_KEY` and `HF_TOKEN` in environment. Optionally set `WAGMI_SKILLS_PATH` to the `wagmi-skills.md` file (defaults to `../dexm-one-page/src/lib/chat/wagmi-skills.md`).

Outputs per iteration:
- `eval_scores_iterN.json` -- judge scores
- `corrections_iterN.jsonl` -- corrected SFT rows
- `data/train_iterN.jsonl` -- merged training set
- `autotune_history.json` -- convergence tracking

## Export to Ollama (local deployment)

After training on HF Spaces, the LoRA adapter is pushed to the Hub.
The GGUF export step of the Space may fail (interactive `apt-get` prompt) -- this is expected and safe to ignore.

To import the model into Ollama locally, run one script that handles everything (download adapter, merge on CPU, create Ollama model):

```bash
pip install torch transformers peft huggingface_hub
HF_TOKEN=hf_xxx python scripts/export_ollama.py
```

This merges the LoRA adapter into the base Qwen model on CPU (~6 GB RAM, takes 2-3 min), then creates the Ollama model with the right parameters.

Requires [Ollama](https://ollama.com) >= 0.5.

Once done: `ollama run wagmi-sft`

## Repository structure

```
sft-wagmi/
├── data/
│   ├── train.jsonl           # training examples
│   ├── eval.jsonl            # evaluation examples
│   └── metadata.json         # generation stats
├── scripts/
│   └── export_ollama.py      # Download adapter, merge on CPU, create Ollama model
├── train.ipynb               # Unsloth SFT notebook
├── baseline.ipynb            # Pre-training baseline eval
├── autotune.ipynb            # Autotune loop (judge + correct + retrain)
├── autotune_history.json     # Convergence tracking (generated)
├── requirements.txt          # Python dependencies
└── README.md
```
