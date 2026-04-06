"""Retrain step — runs in a subprocess to isolate Unsloth monkey-patching.

Usage: python retrain_step.py <train_jsonl> <eval_jsonl> <iteration>
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "small").strip().lower()
PROFILES = {
    "small": {
        "base_model_id": os.environ.get("SMALL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
        "hub_adapter": os.environ.get("SMALL_HUB_MODEL_ID", "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"),
        "output_dir": os.environ.get("SMALL_OUTPUT_DIR", "output/wagmi-qwen2.5-1.5b-sft"),
        "max_seq_len": int(os.environ.get("SMALL_MAX_SEQ_LEN", "2048")),
        "load_in_4bit": os.environ.get("SMALL_LOAD_IN_4BIT", "false").lower() == "true",
        "lora_r": int(os.environ.get("SMALL_LORA_R", "32")),
        "lora_alpha": int(os.environ.get("SMALL_LORA_ALPHA", "64")),
        "learning_rate": float(os.environ.get("SMALL_LEARNING_RATE", "5e-5")),
        "num_epochs": int(os.environ.get("SMALL_NUM_EPOCHS", "3")),
        "per_device_batch": int(os.environ.get("SMALL_PER_DEVICE_BATCH", "4")),
        "grad_accum": int(os.environ.get("SMALL_GRAD_ACCUM", "2")),
        "dataset_num_proc": int(os.environ.get("SMALL_DATASET_NUM_PROC", "2")),
    },
    "auth": {
        "base_model_id": os.environ.get("AUTH_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct"),
        "hub_adapter": os.environ.get("AUTH_HUB_MODEL_ID", "jeanbaptdzd/wagmi-qwen2.5-14b-sft"),
        "output_dir": os.environ.get("AUTH_OUTPUT_DIR", "output/wagmi-qwen2.5-14b-sft"),
        "max_seq_len": int(os.environ.get("AUTH_MAX_SEQ_LEN", "2048")),
        "load_in_4bit": os.environ.get("AUTH_LOAD_IN_4BIT", "true").lower() != "false",
        "lora_r": int(os.environ.get("AUTH_LORA_R", "32")),
        "lora_alpha": int(os.environ.get("AUTH_LORA_ALPHA", "64")),
        "learning_rate": float(os.environ.get("AUTH_LEARNING_RATE", "2e-5")),
        "num_epochs": int(os.environ.get("AUTH_NUM_EPOCHS", "2")),
        "per_device_batch": int(os.environ.get("AUTH_PER_DEVICE_BATCH", "1")),
        "grad_accum": int(os.environ.get("AUTH_GRAD_ACCUM", "8")),
        "dataset_num_proc": int(os.environ.get("AUTH_DATASET_NUM_PROC", "1")),
    },
}
if MODEL_PROFILE not in PROFILES:
    raise RuntimeError(f"Unsupported MODEL_PROFILE={MODEL_PROFILE}")

cfg = PROFILES[MODEL_PROFILE]
BASE_MODEL_ID = cfg["base_model_id"]
HUB_ADAPTER = cfg["hub_adapter"]
OUTPUT_DIR = Path(cfg["output_dir"])
MAX_SEQ_LEN = int(cfg["max_seq_len"])
DTYPE = torch.bfloat16
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LORA_R = int(cfg["lora_r"])
LORA_ALPHA = int(cfg["lora_alpha"])
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = float(cfg["learning_rate"])
NUM_EPOCHS = int(cfg["num_epochs"])
PER_DEVICE_BATCH = int(cfg["per_device_batch"])
GRAD_ACCUM = int(cfg["grad_accum"])
DATASET_NUM_PROC = int(cfg["dataset_num_proc"])


def main():
    train_path = sys.argv[1]
    eval_path = sys.argv[2]
    iteration = int(sys.argv[3])

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Profile: {MODEL_PROFILE}")
    print(f"Retrain step: iteration {iteration}")
    print(f"Train data: {train_path}")
    print(f"Eval data: {eval_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID, max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE, load_in_4bit=bool(cfg["load_in_4bit"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=TARGET_MODULES, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M trainable")

    raw = load_dataset("json", data_files={"train": train_path, "eval": eval_path})

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )}

    train_ds = raw["train"].map(format_chat, remove_columns=raw["train"].column_names)
    eval_ds = raw["eval"].map(format_chat, remove_columns=raw["eval"].column_names)
    print(f"Dataset: {len(train_ds)} train / {len(eval_ds)} eval")

    out_dir = str(OUTPUT_DIR) + f"_iter{iteration}"

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=DATASET_NUM_PROC, packing=False,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0,
            bf16=True, fp16=False, optim="adamw_8bit",
            logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, report_to="none",
            run_name=f"wagmi-{MODEL_PROFILE}-autotune-iter{iteration}",
            seed=42,
            dataloader_num_workers=(0 if MODEL_PROFILE == "auth" else 2),
            dataloader_pin_memory=(MODEL_PROFILE != "auth"),
        ),
    )

    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nDone in {stats.metrics['train_runtime']:.0f}s, loss={stats.metrics['train_loss']:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Adapter saved to {OUTPUT_DIR}")

    if HF_TOKEN:
        model.push_to_hub(HUB_ADAPTER, token=HF_TOKEN, private=True,
                          commit_message=f"autotune iter {iteration}")
        tokenizer.push_to_hub(HUB_ADAPTER, token=HF_TOKEN, private=True,
                              commit_message=f"autotune iter {iteration}")
        print(f"Pushed to {HUB_ADAPTER}")

    print("Retrain step complete.")


if __name__ == "__main__":
    main()
