"""Central configuration for the sft-wagmi pipeline.

Single source of truth for model families, profiles, and training defaults.
All scripts import from here — do not duplicate family/profile logic elsewhere.

Adding a new family (e.g. qwen4, lfm3):
    1. Add an entry to _REGISTRY below with 'small' and 'auth' sub-dicts.
    2. Add the key to VALID_FAMILIES.
    3. That's it — every script picks it up automatically.

Adding a new profile:
    Add the profile key to every family entry in _REGISTRY and to VALID_PROFILES.

Runtime environment overrides (highest priority, applied on top of registry defaults):
    MODEL_PROFILE, LLM_FAMILY — family / profile selection
    SMALL_MODEL_ID / AUTH_MODEL_ID — override base model ID
    SMALL_OUTPUT_DIR / AUTH_OUTPUT_DIR — override local adapter dir
    SMALL_HUB_MODEL_ID / AUTH_HUB_MODEL_ID — override Hub adapter repo
    SMALL_MAX_SEQ_LEN / AUTH_MAX_SEQ_LEN
    SMALL_LOAD_IN_4BIT / AUTH_LOAD_IN_4BIT
    SMALL_LORA_R / AUTH_LORA_R
    SMALL_LORA_ALPHA / AUTH_LORA_ALPHA
    SMALL_LEARNING_RATE / AUTH_LEARNING_RATE
    SMALL_NUM_EPOCHS / AUTH_NUM_EPOCHS
    SMALL_PER_DEVICE_BATCH / AUTH_PER_DEVICE_BATCH
    SMALL_GRAD_ACCUM / AUTH_GRAD_ACCUM
    SMALL_DATASET_NUM_PROC / AUTH_DATASET_NUM_PROC
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def read_package_version() -> str:
    """Semver from the VERSION file next to this module (HF / SSH sanity checks)."""
    vpath = _PACKAGE_ROOT / "VERSION"
    return vpath.read_text().strip() if vpath.is_file() else "?"


# ── Supported families and profiles ───────────────────────────────────────────

VALID_FAMILIES: frozenset[str] = frozenset({"qwen", "qwen3", "lfm2"})
VALID_PROFILES: frozenset[str] = frozenset({"small", "auth"})

# ── Registry ──────────────────────────────────────────────────────────────────
# Each entry: family -> profile -> defaults dict.
# All integer/float/bool fields are stored as their native Python types.
# Training note — Qwen3: thinking mode is enabled by default in Qwen3 base
# models. When using these for SFT with a chat template, pass
# enable_thinking=False to the tokenizer / apply_chat_template call to prevent
# <think> token leakage into training examples.

_REGISTRY: dict[str, dict[str, dict]] = {
    # ── Qwen 2.5 ──────────────────────────────────────────────────────────────
    "qwen": {
        "small": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter_dir": "output/wagmi-qwen2.5-1.5b-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft",
            "hub_merged": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-gguf",
            "ollama_name": "wagmi-sft",
            "max_seq_len": 2048,
            "load_in_4bit": False,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "per_device_batch": 4,
            "grad_accum": 2,
            "dataset_num_proc": 2,
        },
        "auth": {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "adapter_dir": "output/wagmi-qwen2.5-14b-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-qwen2.5-14b-sft",
            "hub_merged": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-gguf",
            "ollama_name": "wagmi-sft-14b",
            "max_seq_len": 2048,
            "load_in_4bit": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 2e-5,
            "num_epochs": 2,
            "per_device_batch": 1,
            "grad_accum": 8,
            "dataset_num_proc": 1,
        },
    },

    # ── Qwen 3 ────────────────────────────────────────────────────────────────
    # Qwen3 models ship without -Instruct suffix; instruction-following is
    # built in. Use unsloth/ IDs for Unsloth-based training.
    # Verify exact model IDs on https://huggingface.co/Qwen before first run.
    # Enable thinking=False in apply_chat_template for standard SFT.
    "qwen3": {
        "small": {
            "model_id": "unsloth/Qwen3-0.6B",
            "adapter_dir": "output/wagmi-qwen3-0.6b-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-qwen3-0.6b-sft",
            "hub_merged": "jeanbaptdzd/wagmi-qwen3-0.6b-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-qwen3-0.6b-sft-gguf",
            "ollama_name": "wagmi-qwen3-small",
            "max_seq_len": 2048,
            "load_in_4bit": False,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "per_device_batch": 4,
            "grad_accum": 2,
            "dataset_num_proc": 2,
        },
        "auth": {
            "model_id": "unsloth/Qwen3-8B",
            "adapter_dir": "output/wagmi-qwen3-8b-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-qwen3-8b-sft",
            "hub_merged": "jeanbaptdzd/wagmi-qwen3-8b-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-qwen3-8b-sft-gguf",
            "ollama_name": "wagmi-qwen3-auth",
            "max_seq_len": 2048,
            "load_in_4bit": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 2e-5,
            "num_epochs": 2,
            "per_device_batch": 2,
            "grad_accum": 4,
            "dataset_num_proc": 1,
        },
    },

    # ── Liquid AI LFM2 ────────────────────────────────────────────────────────
    # Use unsloth/ IDs when training with Unsloth (FastLanguageModel).
    # Upstream LiquidAI/ IDs are valid for plain Transformers/PEFT only.
    "lfm2": {
        # LFM2.5 small: align with Liquid Unsloth quick-start (r=16, alpha=32) and Unsloth LoRA
        # guide (LR ~1e-4–2e-4 band, effective batch ~16). See docs.liquid.ai/lfm/fine-tuning/unsloth
        "small": {
            "model_id": "unsloth/LFM2.5-1.2B-Instruct",
            "adapter_dir": "output/wagmi-lfm2-small-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-lfm2-small-sft",
            "hub_merged": "jeanbaptdzd/wagmi-lfm2-small-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-lfm2-small-sft-gguf",
            "ollama_name": "wagmi-lfm2-small",
            "max_seq_len": 2048,
            "load_in_4bit": False,
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "per_device_batch": 2,
            "grad_accum": 8,
            "dataset_num_proc": 2,
        },
        "auth": {
            "model_id": "unsloth/LFM2-8B-A1B",
            "adapter_dir": "output/wagmi-lfm2-auth-sft",
            "hub_adapter": "jeanbaptdzd/wagmi-lfm2-auth-sft",
            "hub_merged": "jeanbaptdzd/wagmi-lfm2-auth-sft-merged",
            "hub_gguf": "jeanbaptdzd/wagmi-lfm2-auth-sft-gguf",
            "ollama_name": "wagmi-lfm2-auth",
            "max_seq_len": 2048,
            "load_in_4bit": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 2e-5,
            "num_epochs": 2,
            "per_device_batch": 1,
            "grad_accum": 8,
            "dataset_num_proc": 1,
        },
    },
}

# ── Typed config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProfileConfig:
    """Resolved, fully-typed profile configuration. Immutable after construction."""
    family: str
    profile: str
    # Model
    model_id: str
    # Local adapter / output
    adapter_dir: str
    # Hub identifiers
    hub_adapter: str
    hub_merged: str
    hub_gguf: str
    # Inference
    ollama_name: str
    run_name: str
    # Training hyperparameters
    max_seq_len: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    learning_rate: float
    num_epochs: int
    per_device_batch: int
    grad_accum: int
    dataset_num_proc: int

    def asdict(self) -> dict:
        """Return a plain dict — useful for JSON logging."""
        return dataclasses.asdict(self)


# ── Resolver helpers ───────────────────────────────────────────────────────────

def resolve_family(default: str = "qwen") -> str:
    """Read LLM_FAMILY from env, validate, and return it."""
    val = os.environ.get("LLM_FAMILY", default).strip().lower()
    if val not in VALID_FAMILIES:
        raise ValueError(
            f"Unsupported LLM_FAMILY={val!r}. "
            f"Expected one of: {', '.join(sorted(VALID_FAMILIES))}"
        )
    return val


def resolve_profile(default: str = "small") -> str:
    """Read MODEL_PROFILE from env, validate, and return it."""
    val = os.environ.get("MODEL_PROFILE", default).strip().lower()
    if val not in VALID_PROFILES:
        raise ValueError(
            f"Unsupported MODEL_PROFILE={val!r}. "
            f"Expected one of: {', '.join(sorted(VALID_PROFILES))}"
        )
    return val


def resolve_profile_config(
    family: str | None = None,
    profile: str | None = None,
) -> ProfileConfig:
    """Return a fully-resolved, immutable ProfileConfig.

    Precedence (highest to lowest):
        1. Per-profile env vars  (SMALL_MODEL_ID, AUTH_LORA_R, …)
        2. Registry defaults     (_REGISTRY[family][profile])
    """
    f = family if family is not None else resolve_family()
    p = profile if profile is not None else resolve_profile()

    raw = dict(_REGISTRY[f][p])
    prefix = "SMALL_" if p == "small" else "AUTH_"

    def _s(key: str, field: str) -> str:
        return os.environ.get(f"{prefix}{key}", raw[field])

    def _i(key: str, field: str) -> int:
        return int(os.environ.get(f"{prefix}{key}", raw[field]))

    def _f(key: str, field: str) -> float:
        return float(os.environ.get(f"{prefix}{key}", raw[field]))

    def _b(key: str, field: str) -> bool:
        raw_val = os.environ.get(f"{prefix}{key}")
        if raw_val is None:
            return bool(raw[field])
        return raw_val.strip().lower() not in ("false", "0", "no")

    return ProfileConfig(
        family=f,
        profile=p,
        model_id=_s("MODEL_ID", "model_id"),
        adapter_dir=_s("OUTPUT_DIR", "adapter_dir"),
        hub_adapter=_s("HUB_MODEL_ID", "hub_adapter"),
        hub_merged=os.environ.get(f"{prefix}HUB_MERGED_REPO", raw["hub_merged"]),
        hub_gguf=os.environ.get(f"{prefix}HUB_GGUF_REPO", raw["hub_gguf"]),
        ollama_name=os.environ.get("OLLAMA_MODEL_NAME", raw["ollama_name"]),
        run_name=os.environ.get(f"{prefix}RUN_NAME", f"wagmi-{f}-{p}"),
        max_seq_len=_i("MAX_SEQ_LEN", "max_seq_len"),
        load_in_4bit=_b("LOAD_IN_4BIT", "load_in_4bit"),
        lora_r=_i("LORA_R", "lora_r"),
        lora_alpha=_i("LORA_ALPHA", "lora_alpha"),
        learning_rate=_f("LEARNING_RATE", "learning_rate"),
        num_epochs=_i("NUM_EPOCHS", "num_epochs"),
        per_device_batch=_i("PER_DEVICE_BATCH", "per_device_batch"),
        grad_accum=_i("GRAD_ACCUM", "grad_accum"),
        dataset_num_proc=_i("DATASET_NUM_PROC", "dataset_num_proc"),
    )


def print_device_info() -> None:
    """Print GPU name and VRAM, or a CPU-only notice — safe on any platform."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("GPU: none detected (CPU-only mode)")
    except Exception:
        print("GPU: could not query device info")


def dump_registry() -> None:
    """Print the full registry — useful for verifying a new family was added."""
    import json
    print(json.dumps(
        {f: {p: {k: str(v) for k, v in d.items()} for p, d in profiles.items()}
         for f, profiles in _REGISTRY.items()},
        indent=2,
    ))
