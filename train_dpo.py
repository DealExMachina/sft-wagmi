"""Wagmi DPO training — safety alignment from SFT adapter.

Starts from the latest SFT adapter (auth/qwen) and fine-tunes with DPO
using chosen/rejected preference pairs from data/dpo/wagmi_safety_dpo.jsonl.

Purpose: teach the model to emit explicit refusals ("I refuse." / "Je refuse.")
instead of soft deflections ("I don't…", "Je ne decode pas…"), which are
sufficient to pass the must_refuse gate in eval_redteam.py.

Environment variables (mirrors train.py conventions):
    MODEL_PROFILE   — defaults to 'auth'
    LLM_FAMILY      — defaults to 'qwen'
    HF_TOKEN        — required for push to Hub
    PUSH_TO_HUB     — 'true' (default) or 'false'
    DPO_BETA        — KL penalty weight (default: 0.15)
    DPO_EPOCHS      — number of training epochs (default: 2)
    DPO_MAX_LENGTH  — max sequence length for DPO (default: 512)
    DPO_LR          — learning rate (default: 5e-5)
    DPO_BATCH       — per-device train batch size (default: 1)
    DPO_GRAD_ACCUM  — gradient accumulation steps (default: 8)
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# Memory settings before Unsloth import (must come early)
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

# ── DPO hyperparams ───────────────────────────────────────────────────────────
DPO_BETA = float(os.environ.get("DPO_BETA", "0.15"))
DPO_EPOCHS = int(os.environ.get("DPO_EPOCHS", "2"))
DPO_MAX_LENGTH = int(os.environ.get("DPO_MAX_LENGTH", "512"))
DPO_LR = float(os.environ.get("DPO_LR", "5e-5"))
DPO_BATCH = int(os.environ.get("DPO_BATCH", "1"))
DPO_GRAD_ACCUM = int(os.environ.get("DPO_GRAD_ACCUM", "8"))

# ── Paths ─────────────────────────────────────────────────────────────────────
ADAPTER_DIR = cfg.hub_adapter          # load SFT adapter from Hub
DPO_OUTPUT_DIR = f"output/wagmi-{LLM_FAMILY}-{MODEL_PROFILE.replace('auth', '14b')}-sft-dpo"
DPO_HUB_REPO = f"jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo"
DPO_DATA_FILE = ROOT / "data" / "dpo" / "wagmi_safety_dpo.jsonl"

PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() != "false"
HF_TOKEN = os.environ.get("HF_TOKEN")

DTYPE = torch.bfloat16
LOAD_IN_4BIT = True


def load_dpo_dataset(path: Path) -> Dataset:
    """Load JSONL preference pairs and format for TRL DPOTrainer."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # TRL DPOTrainer with tokenize_row=True expects:
    #   prompt_input_ids / chosen_input_ids / rejected_input_ids
    # But the simplest path is to keep messages dicts and let the trainer
    # call apply_chat_template internally.  TRL ≥0.8 accepts this format:
    #   {"prompt": [{"role":...}], "chosen": [{"role":...}], "rejected": [{"role":...}]}
    # which is exactly what our JSONL contains.
    return Dataset.from_list(rows)


def run() -> None:
    if not torch.cuda.is_available():
        print(
            "DPO training requires a CUDA GPU (Unsloth). "
            "Current runtime has no GPU. "
            "Switch Space hardware to L40 or A100, then rerun.",
            flush=True,
        )
        raise SystemExit(2)

    # ── Patch DPO before importing TRL ────────────────────────────────────────
    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()
    from trl import DPOTrainer, DPOConfig

    print_device_info()
    print(json.dumps({
        "version": MODEL_VERSION,
        "family": LLM_FAMILY,
        "profile": MODEL_PROFILE,
        "adapter_from": ADAPTER_DIR,
        "output_to": DPO_OUTPUT_DIR,
        "hub_repo": DPO_HUB_REPO,
        "beta": DPO_BETA,
        "epochs": DPO_EPOCHS,
        "max_length": DPO_MAX_LENGTH,
        "lr": DPO_LR,
        "batch": DPO_BATCH,
        "grad_accum": DPO_GRAD_ACCUM,
        "load_in_4bit": LOAD_IN_4BIT,
    }, indent=2))

    # ── Load SFT adapter as starting point ────────────────────────────────────
    print(f"\nLoading SFT adapter: {ADAPTER_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=DPO_MAX_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded ({model.num_parameters() / 1e6:.1f}M params)")

    # ── Apply LoRA (fresh head on top of SFT weights) ─────────────────────────
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M trainable ({100 * trainable / total:.2f}%)")

    # ── Load DPO preference dataset ───────────────────────────────────────────
    if not DPO_DATA_FILE.exists():
        raise FileNotFoundError(
            f"DPO dataset not found: {DPO_DATA_FILE}\n"
            "Run: python3 scripts/build_dpo_dataset.py [--merge-giskard]"
        )
    full_ds = load_dpo_dataset(DPO_DATA_FILE)
    # Simple train/eval split: 90/10
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"DPO dataset: {len(train_ds)} train / {len(eval_ds)} eval")

    BF16 = torch.cuda.is_bf16_supported()

    # ── DPOConfig ─────────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH,
        per_device_eval_batch_size=DPO_BATCH,
        gradient_accumulation_steps=DPO_GRAD_ACCUM,
        learning_rate=DPO_LR,
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
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        beta=DPO_BETA,
        max_length=DPO_MAX_LENGTH,
        max_prompt_length=DPO_MAX_LENGTH // 2,
        # ref_model=None means Unsloth uses the frozen SFT base as reference
        # (implicit ref model via PatchDPOTrainer)
    )

    # ── DPO Trainer ───────────────────────────────────────────────────────────
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting DPO training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stats = dpo_trainer.train()
    print(f"\nDPO training complete. Runtime: {stats.metrics['train_runtime']:.0f}s")
    print(f"Train loss: {stats.metrics['train_loss']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(DPO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)
    print(f"DPO adapter saved to {DPO_OUTPUT_DIR}")

    if PUSH_TO_HUB:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set, skipping push to Hub.")
        else:
            commit_msg = f"wagmi-dpo v{MODEL_VERSION} ({MODEL_PROFILE}/{LLM_FAMILY})"
            model.push_to_hub(DPO_HUB_REPO, token=HF_TOKEN, private=True, commit_message=commit_msg)
            tokenizer.push_to_hub(DPO_HUB_REPO, token=HF_TOKEN, private=True, commit_message=commit_msg)
            print(f"DPO adapter pushed to https://huggingface.co/{DPO_HUB_REPO} [{commit_msg}]")

    print(f"\nDone. Next step: export GGUF and re-run eval_redteam.py against Koyeb.")
    return stats


if __name__ == "__main__":
    run()
