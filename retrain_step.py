"""Retrain step — runs in a subprocess to isolate Unsloth monkey-patching.

Usage:
    python retrain_step.py <train_jsonl> <eval_jsonl> <iteration> [--family F] [--profile P]

``--family`` / ``--profile`` are applied to the environment before Unsloth is
imported so the correct base model is always loaded (avoids import-order / env
merge issues with nested subprocesses).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autotune retrain subprocess (Unsloth SFT)")
    p.add_argument("train_path", type=Path, help="Training JSONL path")
    p.add_argument("eval_path", type=Path, help="Eval JSONL path")
    p.add_argument("iteration", type=int, help="Autotune iteration index")
    p.add_argument(
        "--family",
        default=None,
        metavar="F",
        help="LLM_FAMILY (qwen|qwen3|lfm2). When set, overrides env for this process.",
    )
    p.add_argument(
        "--profile",
        default=None,
        choices=("small", "auth"),
        metavar="P",
        help="MODEL_PROFILE. When set, overrides env for this process.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    _root = Path(__file__).resolve().parent
    _ver = (_root / "VERSION").read_text().strip() if (_root / "VERSION").is_file() else "?"
    print(
        f"sft-wagmi retrain_step VERSION={_ver} argv={sys.argv!r} "
        f"(expect --family/--profile before Unsloth import on 0.3.1+)",
        flush=True,
    )
    if args.family is not None:
        os.environ["LLM_FAMILY"] = str(args.family).strip().lower()
    if args.profile is not None:
        os.environ["MODEL_PROFILE"] = str(args.profile).strip().lower()

    # Heavy imports only after CLI/env are final (Unsloth patches on import).
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    from config import print_device_info, resolve_family, resolve_profile, resolve_profile_config

    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
        category=FutureWarning,
    )

    llm_family = resolve_family()
    model_profile = resolve_profile()
    cfg = resolve_profile_config(llm_family, model_profile)

    base_model_id = cfg.model_id
    hub_adapter = cfg.hub_adapter
    output_dir = Path(cfg.adapter_dir)
    max_seq_len = cfg.max_seq_len
    dtype = torch.bfloat16
    hf_token = os.environ.get("HF_TOKEN", "")

    lora_r = cfg.lora_r
    lora_alpha = cfg.lora_alpha
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    learning_rate = cfg.learning_rate
    num_epochs = cfg.num_epochs
    per_device_batch = cfg.per_device_batch
    grad_accum = cfg.grad_accum
    dataset_num_proc = cfg.dataset_num_proc

    train_path = str(args.train_path)
    eval_path = str(args.eval_path)
    iteration = args.iteration

    print_device_info()
    print(f"retrain_step: LLM_FAMILY={llm_family} MODEL_PROFILE={model_profile}")
    print(f"retrain_step: base model = {base_model_id}")
    print(f"Retrain step: iteration {iteration}")
    print(f"Train data: {train_path}")
    print(f"Eval data: {eval_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=bool(cfg.load_in_4bit),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M trainable")

    raw = load_dataset("json", data_files={"train": train_path, "eval": eval_path})

    def format_chat(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = raw["train"].map(format_chat, remove_columns=raw["train"].column_names)
    eval_ds = raw["eval"].map(format_chat, remove_columns=raw["eval"].column_names)
    print(f"Dataset: {len(train_ds)} train / {len(eval_ds)} eval")

    out_dir = str(output_dir) + f"_iter{iteration}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=max(
                1,
                int((len(train_ds) / max(1, per_device_batch * grad_accum)) * num_epochs * 0.05),
            ),
            weight_decay=0.01,
            max_grad_norm=1.0,
            bf16=True,
            fp16=False,
            optim="adamw_8bit",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            run_name=f"wagmi-{model_profile}-autotune-iter{iteration}",
            seed=42,
            dataloader_num_workers=(0 if model_profile == "auth" else 2),
            dataloader_pin_memory=(model_profile != "auth"),
        ),
    )

    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nDone in {stats.metrics['train_runtime']:.0f}s, loss={stats.metrics['train_loss']:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")

    if hf_token:
        model.push_to_hub(hub_adapter, token=hf_token, private=True, commit_message=f"autotune iter {iteration}")
        tokenizer.push_to_hub(hub_adapter, token=hf_token, private=True, commit_message=f"autotune iter {iteration}")
        print(f"Pushed to {hub_adapter}")

    print("Retrain step complete.")


if __name__ == "__main__":
    main()
