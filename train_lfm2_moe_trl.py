"""SFT training for LFM2-8B-A1B (MoE, custom_code) via TRL — Unsloth-free path.

LFM2-8B-A1B uses a non-standard MoE architecture (lfm2_moe) with custom_code=True.
Unsloth does not support it natively. This script uses vanilla TRL + PEFT + bitsandbytes
with trust_remote_code=True, which is the path recommended by Liquid AI docs for TRL.

Usage:
    LLM_FAMILY=lfm2 MODEL_PROFILE=auth python3 train_lfm2_moe_trl.py

Key env overrides (see config.py):
    AUTH_MODEL_ID          — base model (default: LiquidAI/LFM2-8B-A1B)
    AUTH_OUTPUT_DIR        — adapter output dir
    AUTH_HUB_MODEL_ID      — Hub repo for push
    AUTH_NUM_EPOCHS
    AUTH_LEARNING_RATE
    AUTH_PER_DEVICE_BATCH
    AUTH_GRAD_ACCUM
    PUSH_TO_HUB=false      — skip Hub push
    HF_TOKEN               — Hub token (required for private Hub adapter push)

Hardware: L40 (48 GB) or A100 40/80 GB. The MoE has 8B total / 1B active params;
4-bit QLoRA fits comfortably on an L40.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import resolve_family, resolve_profile, resolve_profile_config

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text().strip()

LLM_FAMILY = resolve_family()
MODEL_PROFILE = resolve_profile()

if LLM_FAMILY != "lfm2":
    raise SystemExit(f"This script is for LLM_FAMILY=lfm2 only (got {LLM_FAMILY!r})")
if MODEL_PROFILE != "auth":
    raise SystemExit(f"This script is for MODEL_PROFILE=auth only (got {MODEL_PROFILE!r})")

cfg = resolve_profile_config(LLM_FAMILY, MODEL_PROFILE)

MODEL_ID = cfg.model_id
MAX_SEQ_LEN = cfg.max_seq_len
LORA_R = cfg.lora_r
LORA_ALPHA = cfg.lora_alpha
LEARNING_RATE = cfg.learning_rate
NUM_EPOCHS = cfg.num_epochs
PER_DEVICE_BATCH = cfg.per_device_batch
GRAD_ACCUM = cfg.grad_accum
DATASET_NUM_PROC = cfg.dataset_num_proc
OUTPUT_DIR = cfg.adapter_dir
HUB_MODEL_ID = cfg.hub_adapter
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "true").lower() != "false"
HF_TOKEN = os.environ.get("HF_TOKEN")

TRAIN_FILE = "data/train.jsonl"
EVAL_FILE = "data/eval.jsonl"
AUTH_TOOLING_FILE = os.environ.get("AUTH_TOOLING_FILE", "data/tooling_email_calendar.jsonl")
AUTH_TOOLING_MULTIPLIER = int(os.environ.get("AUTH_TOOLING_MULTIPLIER", "3"))

# LFM2-8B-A1B target modules (MoE + attention).
# The MoE uses gate/expert projection layers; include them for alignment quality.
# Falls back gracefully if a module is absent (PEFT skips unknowns).
TARGET_MODULES = [
    # Attention projections (shared across all LFM2 MoE variants)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Standard FFN layers (non-expert, if present)
    "gate_proj", "up_proj", "down_proj",
    # MoE expert projections (lfm2_moe architecture: LFM2-8B-A1B, LFM2-24B-A2B)
    "w1", "w2", "w3",
]

BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_train_dataset():
    main = load_dataset("json", data_files=TRAIN_FILE, split="train")
    tooling_path = Path(AUTH_TOOLING_FILE)
    if tooling_path.exists() and AUTH_TOOLING_MULTIPLIER > 0:
        tooling = load_dataset("json", data_files=str(tooling_path), split="train")
        copies = [tooling] * AUTH_TOOLING_MULTIPLIER
        main = concatenate_datasets([main] + copies).shuffle(seed=42)
        print(f"Dataset: {len(main)} rows (tooling x{AUTH_TOOLING_MULTIPLIER})")
    else:
        print(f"Dataset: {len(main)} rows (no tooling file)")
    return main


def formatting_func(example: dict) -> str:
    """Convert messages list to a chat-template string using the tokenizer."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def run() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA GPU required for {MODEL_ID} QLoRA training.")

    print(f"=== LFM2 MoE SFT (TRL) — v{MODEL_VERSION} ===")
    print(f"Model:   {MODEL_ID}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")

    global tokenizer  # needed in formatting_func

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        # device_map="auto" distributes the model across all available GPUs.
        # LFM2-24B-A2B in bf16 is 44.42 GiB — just over one L40S (44.39 GiB).
        # Requires l40sx4 (or larger) so that loading fits across GPUs.
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        # modules_to_save: keep embed/lm_head full-precision if needed
    )

    train_ds = load_train_dataset()
    eval_ds = load_dataset("json", data_files=EVAL_FILE, split="train") if Path(EVAL_FILE).exists() else None

    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        bf16=BF16,
        fp16=not BF16,
        dataset_text_field=None,   # we use formatting_func
        packing=False,
        push_to_hub=False,         # manual push below
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}")

    if PUSH_TO_HUB:
        if not HF_TOKEN:
            print("WARN: HF_TOKEN not set — skipping Hub push.")
        else:
            from huggingface_hub import HfApi, create_repo
            create_repo(HUB_MODEL_ID, token=HF_TOKEN, private=True, repo_type="model", exist_ok=True)
            api = HfApi(token=HF_TOKEN)
            api.upload_folder(
                folder_path=OUTPUT_DIR,
                repo_id=HUB_MODEL_ID,
                repo_type="model",
                commit_message=f"sft lfm2-moe auth v{MODEL_VERSION}",
            )
            print(f"Adapter pushed to Hub: {HUB_MODEL_ID}")


if __name__ == "__main__":
    run()
