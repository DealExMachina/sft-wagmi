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

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

LLM_FAMILY = os.environ.get("LLM_FAMILY", "qwen").strip().lower()
if LLM_FAMILY not in {"qwen", "lfm2"}:
    raise ValueError("Unsupported LLM_FAMILY. Expected one of: qwen, lfm2")


def _default_profile_config(profile: str) -> dict[str, str]:
    if LLM_FAMILY == "lfm2":
        if profile == "small":
            return {
                "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
                "output_dir": "output/wagmi-lfm2-small-sft",
                "hub_model_id": "jeanbaptdzd/wagmi-lfm2-small-sft",
                "run_name": "wagmi-lfm2-small",
            }
        return {
            "model_id": "LiquidAI/LFM2-8B-A1B",
            "output_dir": "output/wagmi-lfm2-auth-sft",
            "hub_model_id": "jeanbaptdzd/wagmi-lfm2-auth-sft",
            "run_name": "wagmi-lfm2-auth",
        }
    if profile == "small":
        return {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "output_dir": "output/wagmi-qwen2.5-1.5b-sft",
            "hub_model_id": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft",
            "run_name": "wagmi-qwen2.5-1.5b",
        }
    return {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "output_dir": "output/wagmi-qwen2.5-14b-sft",
        "hub_model_id": "jeanbaptdzd/wagmi-qwen2.5-14b-sft",
        "run_name": "wagmi-qwen2.5-14b",
    }


SMALL_DEFAULTS = _default_profile_config("small")
AUTH_DEFAULTS = _default_profile_config("auth")

PROFILES = {
    "small": {
        "model_id": os.environ.get("SMALL_MODEL_ID", SMALL_DEFAULTS["model_id"]),
        "max_seq_len": int(os.environ.get("SMALL_MAX_SEQ_LEN", "2048")),
        "load_in_4bit": os.environ.get("SMALL_LOAD_IN_4BIT", "false").lower() == "true",
        "lora_r": int(os.environ.get("SMALL_LORA_R", "32")),
        "lora_alpha": int(os.environ.get("SMALL_LORA_ALPHA", "64")),
        "learning_rate": float(os.environ.get("SMALL_LEARNING_RATE", "5e-5")),
        "num_epochs": int(os.environ.get("SMALL_NUM_EPOCHS", "3")),
        "per_device_batch": int(os.environ.get("SMALL_PER_DEVICE_BATCH", "4")),
        "grad_accum": int(os.environ.get("SMALL_GRAD_ACCUM", "2")),
        "dataset_num_proc": int(os.environ.get("SMALL_DATASET_NUM_PROC", "2")),
        "output_dir": os.environ.get("SMALL_OUTPUT_DIR", SMALL_DEFAULTS["output_dir"]),
        "hub_model_id": os.environ.get("SMALL_HUB_MODEL_ID", SMALL_DEFAULTS["hub_model_id"]),
        "run_name": os.environ.get("SMALL_RUN_NAME", SMALL_DEFAULTS["run_name"]),
    },
    "auth": {
        "model_id": os.environ.get("AUTH_MODEL_ID", AUTH_DEFAULTS["model_id"]),
        "max_seq_len": int(os.environ.get("AUTH_MAX_SEQ_LEN", "2048")),
        "load_in_4bit": os.environ.get("AUTH_LOAD_IN_4BIT", "true").lower() != "false",
        "lora_r": int(os.environ.get("AUTH_LORA_R", "32")),
        "lora_alpha": int(os.environ.get("AUTH_LORA_ALPHA", "64")),
        "learning_rate": float(os.environ.get("AUTH_LEARNING_RATE", "2e-5")),
        "num_epochs": int(os.environ.get("AUTH_NUM_EPOCHS", "2")),
        "per_device_batch": int(os.environ.get("AUTH_PER_DEVICE_BATCH", "1")),
        "grad_accum": int(os.environ.get("AUTH_GRAD_ACCUM", "8")),
        "dataset_num_proc": int(os.environ.get("AUTH_DATASET_NUM_PROC", "1")),
        "output_dir": os.environ.get("AUTH_OUTPUT_DIR", AUTH_DEFAULTS["output_dir"]),
        "hub_model_id": os.environ.get("AUTH_HUB_MODEL_ID", AUTH_DEFAULTS["hub_model_id"]),
        "run_name": os.environ.get("AUTH_RUN_NAME", AUTH_DEFAULTS["run_name"]),
    },
}

MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "small").strip().lower()
if MODEL_PROFILE not in PROFILES:
    raise ValueError(f"Unsupported MODEL_PROFILE={MODEL_PROFILE}. Expected one of: {', '.join(PROFILES.keys())}")

cfg = PROFILES[MODEL_PROFILE]
MODEL_ID = cfg["model_id"]
MAX_SEQ_LEN = int(cfg["max_seq_len"])
DTYPE = torch.bfloat16
LOAD_IN_4BIT = bool(cfg["load_in_4bit"])
LORA_R = int(cfg["lora_r"])
LORA_ALPHA = int(cfg["lora_alpha"])
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = float(cfg["learning_rate"])
NUM_EPOCHS = int(cfg["num_epochs"])
PER_DEVICE_BATCH = int(cfg["per_device_batch"])
GRAD_ACCUM = int(cfg["grad_accum"])
DATASET_NUM_PROC = int(cfg["dataset_num_proc"])
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
OUTPUT_DIR = str(cfg["output_dir"])
HUB_MODEL_ID = str(cfg["hub_model_id"])
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
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
            warmup_ratio=WARMUP_RATIO,
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
            run_name=cfg["run_name"],
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
