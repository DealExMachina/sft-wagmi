"""Build ``SFTTrainer`` across TRL API variants.

Older TRL: ``tokenizer=``, ``dataset_text_field``, ``max_seq_length``, ``packing``, …
Newer TRL: ``processing_class=`` and SFT-specific fields live on ``SFTConfig`` (``max_length``, etc.).
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase, TrainingArguments


def build_sft_trainer(
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    max_seq_len: int,
    dataset_num_proc: int | None,
    dataset_text_field: str = "text",
    packing: bool = False,
) -> Any:
    from trl import SFTTrainer

    sig = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in sig.parameters:
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_len,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            args=training_args,
        )

    from packaging.version import Version
    from transformers import __version__ as transformers_version
    from trl import SFTConfig

    td = training_args.to_dict()
    hub_token = getattr(training_args, "hub_token", None)
    if hub_token is not None:
        td["hub_token"] = hub_token
    if Version(transformers_version) < Version("5.0.0"):
        td.pop("push_to_hub_token", None)

    td["dataset_text_field"] = dataset_text_field
    td["max_length"] = max_seq_len
    td["dataset_num_proc"] = dataset_num_proc
    td["packing"] = packing

    try:
        sft_args = SFTConfig(**td)
    except TypeError:
        valid = {f.name for f in fields(SFTConfig)}
        td = {k: v for k, v in td.items() if k in valid}
        sft_args = SFTConfig(**td)
    return SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
