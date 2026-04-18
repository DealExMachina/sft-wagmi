"""Wagmi SFT training with profile switch (small/auth)."""

import json
import os
import warnings
from pathlib import Path

MODEL_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text().strip()

os.environ["PYTHONUNBUFFERED"] = "1"

# Auth (Qwen 2.5 14B 4-bit): keep compile disabled by default for stability.
# On A100 80GB you can set AUTH_ENABLE_TORCH_COMPILE=1.
# Apply compile-related env before importing Unsloth.
_MODEL_PROFILE_EARLY = os.environ.get("MODEL_PROFILE", "small").strip().lower()
if _MODEL_PROFILE_EARLY == "auth":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _auth_torch_compile = os.environ.get("AUTH_ENABLE_TORCH_COMPILE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not _auth_torch_compile:
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from unsloth import FastLanguageModel

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from config import print_device_info, resolve_family, resolve_profile, resolve_profile_config

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

LLM_FAMILY = resolve_family()
MODEL_PROFILE = resolve_profile()
cfg = resolve_profile_config(LLM_FAMILY, MODEL_PROFILE)

MODEL_ID = cfg.model_id
MAX_SEQ_LEN = cfg.max_seq_len
DTYPE = torch.bfloat16
LOAD_IN_4BIT = cfg["load_in_4bit"]
LORA_R = cfg.lora_r
LORA_ALPHA = cfg.lora_alpha
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = cfg.learning_rate
NUM_EPOCHS = cfg.num_epochs
PER_DEVICE_BATCH = cfg.per_device_batch
GRAD_ACCUM = cfg.grad_accum
DATASET_NUM_PROC = cfg.dataset_num_proc
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
OUTPUT_DIR = cfg.adapter_dir
HUB_MODEL_ID = cfg.hub_adapter
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() != "false"
HF_TOKEN = os.environ.get("HF_TOKEN")
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = not BF16

# ── Data ───────────────────────────────────────────────────────────────────
TRAIN_FILE = "data/train.jsonl"
EVAL_FILE  = "data/eval.jsonl"
AUTH_TOOLING_FILE = os.environ.get("AUTH_TOOLING_FILE", "data/tooling_email_calendar.jsonl")
AUTH_TOOLING_MULTIPLIER = int(os.environ.get("AUTH_TOOLING_MULTIPLIER", "3"))


def run():
    print_device_info()
    if MODEL_PROFILE == "auth":
        print(
            "Auth: default disables torch.compile (set AUTH_ENABLE_TORCH_COMPILE=1 on A100 80GB). "
            "Default AUTH_MAX_SEQ_LEN=2048 for Qwen2.5-14B. "
            "PYTORCH_CUDA_ALLOC_CONF may include expandable_segments."
        )
    print(json.dumps({
        "version": MODEL_VERSION,
        "family": LLM_FAMILY,
        "profile": MODEL_PROFILE,
        "model": MODEL_ID, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "lr": LEARNING_RATE, "epochs": NUM_EPOCHS,
        "load_in_4bit": LOAD_IN_4BIT,
        "max_seq_len": MAX_SEQ_LEN,
        "bf16": BF16,
        "fp16": FP16,
        "effective_batch": PER_DEVICE_BATCH * GRAD_ACCUM,
    }, indent=2))

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded ({model.num_parameters() / 1e6:.1f}M params)")

    # ── LoRA ───────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M trainable ({100 * trainable / total:.2f}%)")

    # ── Dataset ────────────────────────────────────────────────────────────
    raw = load_dataset("json", data_files={"train": TRAIN_FILE, "eval": EVAL_FILE})

    if MODEL_PROFILE == "auth":
        tooling_path = Path(AUTH_TOOLING_FILE)
        if tooling_path.exists():
            tooling_train = load_dataset("json", data_files={"train": str(tooling_path)})["train"]
            base_tooling_count = len(tooling_train)
            if AUTH_TOOLING_MULTIPLIER > 1:
                tooling_train = concatenate_datasets([tooling_train] * AUTH_TOOLING_MULTIPLIER)
            raw["train"] = concatenate_datasets([raw["train"], tooling_train]).shuffle(seed=42)
            print(
                "Auth tooling data enabled: "
                f"{base_tooling_count} base examples x {max(1, AUTH_TOOLING_MULTIPLIER)} "
                f"-> {len(tooling_train)} injected train rows."
            )
        else:
            print(f"Auth tooling data not found at {tooling_path}; continuing with base dataset only.")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = raw["train"].map(format_chat, remove_columns=raw["train"].column_names)
    eval_ds = raw["eval"].map(format_chat, remove_columns=raw["eval"].column_names)
    print(f"Dataset: {len(train_ds)} train / {len(eval_ds)} eval")

    dl_workers = 0 if MODEL_PROFILE == "auth" else 2

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=False,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_steps=max(1, int((len(train_ds) / max(1, PER_DEVICE_BATCH * GRAD_ACCUM)) * NUM_EPOCHS * WARMUP_RATIO)),
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            bf16=BF16,
            fp16=FP16,
            optim="adamw_8bit",
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            run_name=os.environ.get(
                "SMALL_RUN_NAME" if MODEL_PROFILE == "small" else "AUTH_RUN_NAME",
                cfg.run_name,
            ),
            seed=42,
            dataloader_num_workers=dl_workers,
            dataloader_pin_memory=(dl_workers > 0),
        ),
    )

    print("\nStarting training...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stats = trainer.train()
    print(f"\nTraining complete. Runtime: {stats.metrics['train_runtime']:.0f}s")
    print(f"Train loss: {stats.metrics['train_loss']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    if PUSH_TO_HUB:
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set, skipping push to Hub.")
        else:
            commit_msg = f"wagmi-sft v{MODEL_VERSION} ({MODEL_PROFILE})"
            model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN, private=True, commit_message=commit_msg)
            tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN, private=True, commit_message=commit_msg)
            print(f"Adapter pushed to https://huggingface.co/{HUB_MODEL_ID} [{commit_msg}]")

    print("\nDone. Run baseline.py with the adapter to verify outputs.")
    return stats


if __name__ == "__main__":
    run()
