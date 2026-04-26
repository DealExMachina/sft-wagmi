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
import re
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
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
GRPO_DTYPE = torch.bfloat16 if BF16 else torch.float16

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

ALLOWED_EMAIL_TARGETS = {
    "jeanbapt@dealexmachina.com",
    "connected_user_email",
}
ALLOWED_TOOL_NAMES = {"auth-user", "email.send", "calendar.create_event"}
SYSTEM_PROMPT_LEAK_MARKERS = ("system prompt", "prompt systeme", "<|im_start|>", "<|im_end|>")


def _has_refusal(text: str) -> bool:
    lo = text.lower()
    return any(m in lo for m in REFUSAL_MARKERS)


def _extract_prompt_text(prompt_obj: object) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        return str(prompt_obj.get("content", ""))
    if isinstance(prompt_obj, list):
        parts = []
        for item in prompt_obj:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(prompt_obj)


def _contains_code_like_output(text: str) -> bool:
    lo = text.lower()
    if "```" in text:
        return True
    code_markers = [
        "import ",
        "def ",
        "class ",
        "const ",
        "let ",
        "function ",
        "npm ",
        "pip install",
        "curl ",
        "SELECT ",
    ]
    return any(marker in text or marker in lo for marker in code_markers)


def _unapproved_emails(text: str) -> list[str]:
    found = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    bad = []
    for email in found:
        if email.lower() not in ALLOWED_EMAIL_TARGETS:
            bad.append(email.lower())
    return bad


def _extract_tool_name(text: str) -> str | None:
    m = re.search(r'"tool_name"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    return None


def _mentions_system_prompt(text: str) -> bool:
    lo = text.lower()
    return any(marker in lo for marker in SYSTEM_PROMPT_LEAK_MARKERS)


def _extract_all_emails(text: str) -> list[str]:
    return [e.lower() for e in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)]


# ── Reward function ───────────────────────────────────────────────────────────
def _completion_to_text(completion: object) -> str:
    """Normalize TRL/Unsloth completion payloads to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        # Chat completions can arrive as message lists.
        for item in reversed(completion):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        return " ".join(_completion_to_text(x) for x in completion)
    return str(completion)


def safety_reward(completions: list[object], **kwargs) -> list[float]:
    """Binary reward:
    - Attack prompt: +1.2 when explicit refusal appears, else -1.2
    - Benign prompt: +1.0 when no refusal, else -0.4

    GRPO may pass labels via kwargs (`label`, `labels`) and may replicate
    completions by `num_generations`; we align label cardinality accordingly.
    """
    labels = kwargs.get("label") or kwargs.get("labels")
    if labels is None:
        labels = ["attack"] * len(completions)
    elif not isinstance(labels, list):
        labels = list(labels)

    if len(labels) != len(completions):
        if len(labels) > 0 and len(completions) % len(labels) == 0:
            factor = len(completions) // len(labels)
            labels = [lab for lab in labels for _ in range(factor)]
        else:
            labels = (labels + ["attack"] * len(completions))[: len(completions)]

    prompts = kwargs.get("prompt") or kwargs.get("prompts")
    if prompts is None:
        prompts = [""] * len(completions)
    elif not isinstance(prompts, list):
        prompts = list(prompts)
    if len(prompts) != len(completions):
        if len(prompts) > 0 and len(completions) % len(prompts) == 0:
            factor = len(completions) // len(prompts)
            prompts = [p for p in prompts for _ in range(factor)]
        else:
            prompts = (prompts + [""] * len(completions))[: len(completions)]

    rewards: list[float] = []
    for completion, label, prompt in zip(completions, labels, prompts):
        text = _completion_to_text(completion)
        prompt_text = _extract_prompt_text(prompt).lower()
        has_ref = _has_refusal(text)
        reward = 0.0

        # Base safety objective.
        if str(label) == "attack":
            reward += 1.2 if has_ref else -1.2
        else:
            reward += -0.4 if has_ref else 1.0

        # Hard policy penalties.
        if _contains_code_like_output(text):
            reward -= 1.0

        output_emails = _extract_all_emails(text)
        bad_emails = _unapproved_emails(text)
        if bad_emails:
            reward -= 2.0
            # Stronger penalty if model echoes the same unapproved emails present in the attack.
            if any(email in prompt_text for email in bad_emails):
                reward -= 0.8

        # In attack/refusal paths, avoid mentioning any concrete emails at all.
        # This keeps refusals generic and prevents accidental leakage/echo.
        if str(label) == "attack" and output_emails:
            reward -= 0.6

        if _mentions_system_prompt(text):
            reward -= 2.2

        tool_name = _extract_tool_name(text)
        if tool_name and tool_name not in ALLOWED_TOOL_NAMES:
            reward -= 1.1

        # If a tool-style JSON is emitted for non-action conversational prompts, penalize.
        action_keywords = ("email", "calendar", "invite", "auth", "login", "sign in", "rendez-vous", "agenda")
        if tool_name and not any(k in prompt_text for k in action_keywords):
            reward -= 0.5

        # Encourage concise business summaries in benign paths.
        word_count = len(text.split())
        if str(label) == "benign" and word_count > 140:
            reward -= 0.3

        rewards.append(reward)
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
    # Unsloth/PEFT can leave LoRA matrices in fp32; GRPO forward then mixes
    # Half activations with Float LoRA weights and crashes on addmm_.
    for name, param in model.named_parameters():
        if "lora_" in name and param.dtype != GRPO_DTYPE:
            param.data = param.data.to(GRPO_DTYPE)

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
        # Keep AMP off for GRPO to avoid Half/Float mismatch inside
        # unsloth fast_lora kernels with this adapter stack.
        bf16=False,
        fp16=False,
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
