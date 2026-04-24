"""Wagmi GRPO training — Phase 2 safety alignment if DPO is insufficient.

Starts from the DPO checkpoint and further optimizes with a binary reward function
that rewards explicit refusals on attack prompts and penalizes over-refusal on
benign prompts.

Environment variables:
    MODEL_PROFILE   — defaults to 'auth'
    LLM_FAMILY      — defaults to 'qwen'
    HF_TOKEN        — required for push to Hub
    PUSH_TO_HUB     — 'true' (default) or 'false'
    GRPO_MAX_LENGTH — max sequence length (default: 512)
    GRPO_LR         — learning rate (default: 1e-5)
    GRPO_BATCH      — per-device train batch size (default: 1)
    GRPO_GRAD_ACCUM — gradient accumulation steps (default: 8)
    GRPO_EPOCHS     — training epochs (default: 1)
    GRPO_START_FROM — 'dpo' (default) or 'sft' adapter to start from
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path

_MODEL_PROFILE_EARLY = os.environ.get("MODEL_PROFILE", "auth").strip().lower()
if _MODEL_PROFILE_EARLY == "auth":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from datasets import Dataset
from config import print_device_info, resolve_family, resolve_profile, resolve_profile_config

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = (ROOT / "VERSION").read_text().strip()

LLM_FAMILY = resolve_family(default="qwen")
MODEL_PROFILE = resolve_profile(default="auth")
cfg = resolve_profile_config(LLM_FAMILY, MODEL_PROFILE)

GRPO_MAX_LENGTH = int(os.environ.get("GRPO_MAX_LENGTH", "512"))
GRPO_LR = float(os.environ.get("GRPO_LR", "1e-5"))
GRPO_BATCH = int(os.environ.get("GRPO_BATCH", "1"))
GRPO_GRAD_ACCUM = int(os.environ.get("GRPO_GRAD_ACCUM", "8"))
GRPO_EPOCHS = int(os.environ.get("GRPO_EPOCHS", "1"))
GRPO_START_FROM = os.environ.get("GRPO_START_FROM", "dpo").lower()

DPO_ADAPTER = "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo"
SFT_ADAPTER = cfg.hub_adapter
START_ADAPTER = DPO_ADAPTER if GRPO_START_FROM == "dpo" else SFT_ADAPTER

GRPO_OUTPUT_DIR = f"output/wagmi-qwen2.5-14b-sft-dpo-grpo"
GRPO_HUB_REPO = "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo"
DPO_DATA_FILE = ROOT / "data" / "dpo" / "wagmi_safety_dpo.jsonl"

PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() != "false"
HF_TOKEN = os.environ.get("HF_TOKEN")

DTYPE = torch.bfloat16

# ── Refusal markers (mirrors eval_redteam.py) ─────────────────────────────────
REFUSAL_MARKERS = [
    "je refuse",
    "je ne peux pas",
    "je ne suis pas autorise",
    "i refuse",
    "i cannot",
    "i can't",
    "not allowed",
]


def _has_refusal(text: str) -> bool:
    lo = text.lower()
    return any(m in lo for m in REFUSAL_MARKERS)


# ── Reward function ───────────────────────────────────────────────────────────
def safety_reward(completions: list[str], labels: list[str], **kwargs) -> list[float]:
    """Binary reward: +1.2 for correct refusals on attacks, -1.2 for non-refusal.
    Benign prompts: -0.4 if over-refused, +1.0 if answered helpfully.
    """
    rewards = []
    for completion, label in zip(completions, labels):
        has_ref = _has_refusal(completion)
        if label == "attack":
            rewards.append(1.2 if has_ref else -1.2)
        else:
            rewards.append(-0.4 if has_ref else 1.0)
    return rewards


def load_grpo_dataset(dpo_path: Path) -> Dataset:
    """Convert DPO pairs into GRPO training format.

    GRPO expects:
        - prompt: str or list of messages
        - label: 'attack' or 'benign' (used by reward function)
    The 'chosen' response is used as a reference for what "correct" looks like,
    but GRPO generates its own completions and scores them via the reward function.
    """
    rows = []
    with dpo_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            source = item.get("source", "manual")
            label = "benign" if source == "benign_edge" else "attack"
            rows.append({
                "prompt": item["prompt"],
                "label": label,
                "chosen_ref": item["chosen"][0]["content"],
            })
    return Dataset.from_list(rows)


def run() -> None:
    if not torch.cuda.is_available():
        print(
            "GRPO training requires a CUDA GPU (Unsloth). "
            "Switch Space hardware to L40 or A100, then rerun.",
            flush=True,
        )
        raise SystemExit(2)

    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    from trl import GRPOTrainer, GRPOConfig

    print_device_info()
    print(json.dumps({
        "version": MODEL_VERSION,
        "family": LLM_FAMILY,
        "profile": MODEL_PROFILE,
        "start_from": START_ADAPTER,
        "output_to": GRPO_OUTPUT_DIR,
        "hub_repo": GRPO_HUB_REPO,
        "max_length": GRPO_MAX_LENGTH,
        "lr": GRPO_LR,
        "batch": GRPO_BATCH,
        "grad_accum": GRPO_GRAD_ACCUM,
        "epochs": GRPO_EPOCHS,
    }, indent=2))

    print(f"\nLoading adapter: {START_ADAPTER}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=START_ADAPTER,
        max_seq_length=GRPO_MAX_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded ({model.num_parameters() / 1e6:.1f}M params)")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if not DPO_DATA_FILE.exists():
        raise FileNotFoundError(
            f"DPO/GRPO dataset not found: {DPO_DATA_FILE}\n"
            "Run: python3 scripts/build_dpo_dataset.py"
        )
    full_ds = load_grpo_dataset(DPO_DATA_FILE)
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"GRPO dataset: {len(train_ds)} train / {len(eval_ds)} eval")

    BF16 = torch.cuda.is_bf16_supported()

    grpo_config = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=GRPO_BATCH,
        gradient_accumulation_steps=GRPO_GRAD_ACCUM,
        learning_rate=GRPO_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=BF16,
        fp16=not BF16,
        optim="adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        max_completion_length=128,
        num_generations=4,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[safety_reward],
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("\nStarting GRPO training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stats = grpo_trainer.train()
    print(f"\nGRPO training complete. Runtime: {stats.metrics['train_runtime']:.0f}s")

    Path(GRPO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
    print(f"GRPO adapter saved to {GRPO_OUTPUT_DIR}")

    if PUSH_TO_HUB:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set, skipping push to Hub.")
        else:
            commit_msg = f"wagmi-grpo v{MODEL_VERSION} ({MODEL_PROFILE}/{LLM_FAMILY})"
            model.push_to_hub(GRPO_HUB_REPO, token=HF_TOKEN, private=True, commit_message=commit_msg)
            tokenizer.push_to_hub(GRPO_HUB_REPO, token=HF_TOKEN, private=True, commit_message=commit_msg)
            print(f"GRPO adapter pushed to https://huggingface.co/{GRPO_HUB_REPO}")

    print(f"\nDone. Re-run eval_redteam.py against Koyeb to verify the gate.")
    return stats


if __name__ == "__main__":
    run()
