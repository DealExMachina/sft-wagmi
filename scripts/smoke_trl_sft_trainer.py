#!/usr/bin/env python3
"""CPU smoke test: ``build_sft_trainer`` must construct ``SFTTrainer`` (TRL old + new API).

Run from repo root in CI after installing torch (CPU) + trl + transformers + datasets.
Does not use Unsloth or GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from sft_trainer_compat import build_sft_trainer

    model_id = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = {"text": ["hello one", "hello two", "hello three", "hello four"]}
    train_ds = Dataset.from_dict(rows)
    eval_ds = Dataset.from_dict({"text": ["hello eval"]})

    out = ROOT / "output" / "_ci_smoke_trl"
    out.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out),
        max_steps=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        report_to="none",
        bf16=False,
        fp16=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        seed=42,
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_args=args,
        max_seq_len=32,
        dataset_num_proc=None,
        packing=False,
    )
    trainer.train()
    print("smoke_trl_sft_trainer: OK (SFTTrainer constructed + 1 train step)")


if __name__ == "__main__":
    main()
