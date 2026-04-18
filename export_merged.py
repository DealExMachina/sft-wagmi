"""Export merged model to HuggingFace Hub (Space-side, no llama.cpp).

Merges the LoRA adapter into the base model and pushes the full merged
safetensors to Hub.  GGUF conversion is done locally — see scripts/local_gguf_export.sh.
"""

import os
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text().strip()

import torch
from huggingface_hub import HfApi, create_repo
from unsloth import FastLanguageModel

MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "small").strip().lower()
LLM_FAMILY = os.environ.get("LLM_FAMILY", "qwen").strip().lower()
if LLM_FAMILY not in {"qwen", "lfm2"}:
    raise ValueError("Unsupported LLM_FAMILY. Expected one of: qwen, lfm2")


def _defaults(profile: str) -> dict[str, str]:
    if LLM_FAMILY == "lfm2":
        if profile == "small":
            return {
                "base_model_id": "unsloth/LFM2.5-1.2B-Instruct",
                "hub_adapter": "jeanbaptdzd/wagmi-lfm2-small-sft",
                "adapter_dir": "output/wagmi-lfm2-small-sft",
                "hub_merged_repo": "jeanbaptdzd/wagmi-lfm2-small-sft-merged",
                "artifact_prefix": "wagmi-lfm2-small-sft",
            }
        return {
            "base_model_id": "unsloth/LFM2-8B-A1B",
            "hub_adapter": "jeanbaptdzd/wagmi-lfm2-auth-sft",
            "adapter_dir": "output/wagmi-lfm2-auth-sft",
            "hub_merged_repo": "jeanbaptdzd/wagmi-lfm2-auth-sft-merged",
            "artifact_prefix": "wagmi-lfm2-auth-sft",
        }
    if profile == "small":
        return {
            "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "hub_adapter": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft",
            "adapter_dir": "output/wagmi-qwen2.5-1.5b-sft",
            "hub_merged_repo": "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft-merged",
            "artifact_prefix": "wagmi-qwen2.5-1.5b-sft",
        }
    return {
        "base_model_id": "Qwen/Qwen2.5-14B-Instruct",
        "hub_adapter": "jeanbaptdzd/wagmi-qwen2.5-14b-sft",
        "adapter_dir": "output/wagmi-qwen2.5-14b-sft",
        "hub_merged_repo": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-merged",
        "artifact_prefix": "wagmi-qwen2.5-14b-sft",
    }


SMALL_DEFAULTS = _defaults("small")
AUTH_DEFAULTS = _defaults("auth")
PROFILE_CONFIG = {
    "small": {
        "base_model_id": os.environ.get("SMALL_MODEL_ID", SMALL_DEFAULTS["base_model_id"]),
        "hub_adapter": os.environ.get("SMALL_HUB_MODEL_ID", SMALL_DEFAULTS["hub_adapter"]),
        "adapter_dir": os.environ.get("SMALL_OUTPUT_DIR", SMALL_DEFAULTS["adapter_dir"]),
        "hub_merged_repo": os.environ.get("SMALL_HUB_MERGED_REPO", SMALL_DEFAULTS["hub_merged_repo"]),
        "artifact_prefix": os.environ.get("SMALL_ARTIFACT_PREFIX", SMALL_DEFAULTS["artifact_prefix"]),
    },
    "auth": {
        "base_model_id": os.environ.get("AUTH_MODEL_ID", AUTH_DEFAULTS["base_model_id"]),
        "hub_adapter": os.environ.get("AUTH_HUB_MODEL_ID", AUTH_DEFAULTS["hub_adapter"]),
        "adapter_dir": os.environ.get("AUTH_OUTPUT_DIR", AUTH_DEFAULTS["adapter_dir"]),
        "hub_merged_repo": os.environ.get("AUTH_HUB_MERGED_REPO", AUTH_DEFAULTS["hub_merged_repo"]),
        "artifact_prefix": os.environ.get("AUTH_ARTIFACT_PREFIX", AUTH_DEFAULTS["artifact_prefix"]),
    },
}
if MODEL_PROFILE not in PROFILE_CONFIG:
    raise ValueError(f"Unsupported MODEL_PROFILE={MODEL_PROFILE}. Expected: {', '.join(PROFILE_CONFIG.keys())}")

cfg = PROFILE_CONFIG[MODEL_PROFILE]
BASE_MODEL_ID = cfg["base_model_id"]
HUB_ADAPTER = cfg["hub_adapter"]
ADAPTER_DIR = cfg["adapter_dir"]
HUB_MERGED_REPO = cfg["hub_merged_repo"]
MAX_SEQ_LEN = 2048
DTYPE = torch.bfloat16

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MERGED_DIR = Path(f"output/{MODEL_PROFILE}-merged")


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (merge works, just slower)")

    adapter_path = ADAPTER_DIR if Path(ADAPTER_DIR).exists() else HUB_ADAPTER
    print(f"\nProfile:  {MODEL_PROFILE}")
    print(f"Family:   {LLM_FAMILY}")
    print(f"Base:     {BASE_MODEL_ID}")
    print(f"Adapter:  {adapter_path}")
    print(f"Merged →  {HUB_MERGED_REPO}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded ({model.num_parameters() / 1e6:.1f}M params)")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Merging LoRA → saving merged 16-bit model")
    print(f"{'='*60}\n")

    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )

    safetensors = sorted(MERGED_DIR.glob("*.safetensors"))
    total_mb = sum(f.stat().st_size for f in safetensors) / 1e6
    print(f"\nMerged model saved: {len(safetensors)} file(s), {total_mb:.0f} MB total")

    if not HF_TOKEN:
        print("\nHF_TOKEN not set — skipping Hub push.")
        print(f"Files are in {MERGED_DIR}/")
        return

    print(f"\nPushing to {HUB_MERGED_REPO} ...")
    try:
        create_repo(HUB_MERGED_REPO, token=HF_TOKEN, private=True, repo_type="model", exist_ok=True)
        print(f"  Repo ready: {HUB_MERGED_REPO}")
    except Exception as e:
        print(f"  WARNING: create_repo: {e}")
        traceback.print_exc()

    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_folder(
            folder_path=str(MERGED_DIR),
            repo_id=HUB_MERGED_REPO,
            repo_type="model",
            commit_message=f"wagmi-sft v{MODEL_VERSION} ({MODEL_PROFILE}) merged bf16",
        )
        print(f"  Upload complete.")
    except Exception as e:
        print(f"  ERROR uploading: {e}")
        traceback.print_exc()
        return

    print(f"""
{'='*60}
  DONE — merged model on Hub
{'='*60}

  Repo: https://huggingface.co/{HUB_MERGED_REPO}

  Next step — run locally on your Mac:

    cd sft-wagmi
    ./scripts/local_gguf_export.sh {MODEL_PROFILE}

  This will download the merged model, convert to GGUF Q4_K_M,
  generate the Ollama Modelfile, and push the GGUF to Hub.

{'='*60}
""")


if __name__ == "__main__":
    run()
