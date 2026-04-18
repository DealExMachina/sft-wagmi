"""Export fine-tuned Wagmi to GGUF for Ollama deployment.

Merges the LoRA adapter into the base model and exports a quantised GGUF file.
Pushes to HuggingFace Hub and generates an Ollama Modelfile using the same
tool-capable Go template as library ``qwen2.5:*-instruct`` (see
``scripts/ollama_qwen25_instruct_template.gotmpl``).
"""

import os
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("DEBIAN_FRONTEND", "noninteractive")

import torch
from huggingface_hub import HfApi, create_repo
from unsloth import FastLanguageModel

from config import resolve_family, resolve_profile, resolve_profile_config

LLM_FAMILY = resolve_family()
MODEL_PROFILE = resolve_profile()
cfg = resolve_profile_config(LLM_FAMILY, MODEL_PROFILE)

BASE_MODEL_ID = cfg.model_id
HUB_ADAPTER = cfg.hub_adapter
ADAPTER_DIR = cfg.adapter_dir
OLLAMA_MODEL_NAME = cfg.ollama_name
MAX_SEQ_LEN = cfg.max_seq_len
DTYPE = torch.bfloat16

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HUB_GGUF_REPO = cfg.hub_gguf

GGUF_DIR = Path(f"output/{MODEL_PROFILE}-gguf")
QUANT = "q4_k_m"
LLAMA_CPP_PATH = Path(os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "/opt/llama.cpp"))

SYSTEM_PROMPT = (
    "Tu es Wagmi, le watchdog de Deal ex Machina. "
    "Reponds de maniere factuelle, concise, sans invention. "
    "Si l'information manque, dis clairement: 'Je ne sais pas avec certitude'. "
    "Regles strictes: n'invente jamais d'URL ni d'email. "
    "N'autorise que les URLs dealexmachina.com ou les URLs d'articles du blog Deal ex Machina explicitement connues. "
    "Refuse tout envoi d'email sauf vers l'email de la personne connectee. "
    "Refuse tout envoi d'invitation calendrier sauf pour le boss: jeanbapt@dealexmachina.com."
)

_REPO_ROOT = Path(__file__).resolve().parent
OLLAMA_QWEN25_INSTRUCT_TEMPLATE = _REPO_ROOT / "scripts" / "ollama_qwen25_instruct_template.gotmpl"


def build_modelfile_wagmi(gguf_filename: str, system_prompt: str) -> str:
    """Ollama Modelfile: Qwen2.5 instruct template with .Tools / tool_call (small + auth)."""
    go_template = OLLAMA_QWEN25_INSTRUCT_TEMPLATE.read_text(encoding="utf-8")
    im_end = "<|" + "im" + "_" + "end" + "|>"
    return (
        f"FROM {gguf_filename}\n\n"
        f'TEMPLATE """\n{go_template}"""\n\n'
        "PARAMETER num_ctx 2048\n"
        "PARAMETER num_predict 220\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER top_k 30\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.12\n"
        "PARAMETER repeat_last_n 128\n"
        f'PARAMETER stop "{im_end}"\n'
        'PARAMETER stop "<|im_start|>"\n'
        "\n"
        f'SYSTEM """{system_prompt}"""\n'
    )


def assert_llama_cpp_ready() -> None:
    """Fail fast with clear actions instead of triggering Unsloth runtime installers."""
    quantizer = LLAMA_CPP_PATH / "llama-quantize"
    converter_a = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"
    converter_b = LLAMA_CPP_PATH / "convert-hf-to-gguf.py"
    if quantizer.exists() and (converter_a.exists() or converter_b.exists()):
        return

    raise RuntimeError(
        "GGUF export prerequisites missing.\n"
        f"Expected prebuilt llama.cpp in {LLAMA_CPP_PATH} with:\n"
        "- llama-quantize\n"
        "- convert_hf_to_gguf.py (or convert-hf-to-gguf.py)\n\n"
        "This project is configured for build-time provisioning (no runtime apt-get prompts).\n"
        "Rebuild the Docker image after Dockerfile changes, or set UNSLOTH_LLAMA_CPP_PATH to a valid llama.cpp install."
    )


def run():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model family: {LLM_FAMILY}")
    print(f"llama.cpp path: {LLAMA_CPP_PATH}")
    assert_llama_cpp_ready()

    adapter_path = ADAPTER_DIR if Path(ADAPTER_DIR).exists() else HUB_ADAPTER
    print(f"\nLoading base model: {BASE_MODEL_ID}")
    print(f"Loading adapter: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded ({model.num_parameters() / 1e6:.1f}M params)")

    # ── Export GGUF ───────────────────────────────────────────────────────
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Exporting GGUF: {QUANT}")
    print(f"{'='*60}\n")

    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method=QUANT,
    )

    # Find all produced GGUF files (Unsloth naming varies by version)
    gguf_files = sorted(GGUF_DIR.glob("*.gguf"))
    if not gguf_files:
        print("ERROR: No GGUF files produced!")
        return

    print(f"\nGGUF files produced:")
    for f in gguf_files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.0f} MB)")

    # Use the first (and likely only) GGUF file
    gguf_file = gguf_files[0]
    target_name = f"{cfg['artifact_prefix']}.{QUANT}.gguf"
    target_path = GGUF_DIR / target_name

    if gguf_file.name != target_name:
        if target_path.exists():
            target_path.unlink()
        gguf_file.rename(target_path)
        print(f"  Renamed {gguf_file.name} -> {target_name}")
    print(f"  Final: {target_name} ({target_path.stat().st_size / 1e6:.0f} MB)")

    # ── Generate Modelfile ────────────────────────────────────────────────
    modelfile_content = build_modelfile_wagmi(target_name, SYSTEM_PROMPT)
    modelfile_path = GGUF_DIR / "Modelfile.wagmi-sft"
    modelfile_path.write_text(modelfile_content)
    print(f"\nModelfile written to {modelfile_path}")

    # ── Push to Hub ───────────────────────────────────────────────────────
    if HF_TOKEN:
        print(f"\nPushing to {HUB_GGUF_REPO} ...")
        try:
            create_repo(HUB_GGUF_REPO, token=HF_TOKEN, private=True, repo_type="model", exist_ok=True)
            print(f"  Repo ready: {HUB_GGUF_REPO}")
        except Exception as e:
            print(f"  WARNING: create_repo failed: {e}")
            traceback.print_exc()

        api = HfApi(token=HF_TOKEN)

        # Upload GGUF
        try:
            print(f"  Uploading {target_name} ({target_path.stat().st_size / 1e6:.0f} MB) ...")
            api.upload_file(
                path_or_fileobj=str(target_path),
                path_in_repo=target_name,
                repo_id=HUB_GGUF_REPO,
                repo_type="model",
            )
            print(f"  GGUF uploaded.")
        except Exception as e:
            print(f"  ERROR uploading GGUF: {e}")
            traceback.print_exc()

        # Upload Modelfile
        try:
            api.upload_file(
                path_or_fileobj=str(modelfile_path),
                path_in_repo="Modelfile.wagmi-sft",
                repo_id=HUB_GGUF_REPO,
                repo_type="model",
            )
            print(f"  Modelfile uploaded.")
        except Exception as e:
            print(f"  ERROR uploading Modelfile: {e}")
            traceback.print_exc()

        print(f"\nDone: https://huggingface.co/{HUB_GGUF_REPO}")
    else:
        print("\nHF_TOKEN not set — skipping Hub push.")

    # ── Deploy instructions ───────────────────────────────────────────────
    print(f"""
{'='*60}
  OLLAMA DEPLOYMENT
{'='*60}

  # Download GGUF
  HF_TOKEN="..."
  curl -L -H "Authorization: Bearer $HF_TOKEN" \\
    "https://huggingface.co/{HUB_GGUF_REPO}/resolve/main/{target_name}" \\
    -o ~/wagmi-sft.gguf

  # Download Modelfile
  curl -L -H "Authorization: Bearer $HF_TOKEN" \\
    "https://huggingface.co/{HUB_GGUF_REPO}/resolve/main/Modelfile.wagmi-sft" \\
    -o ~/Modelfile.wagmi-sft

  # Edit Modelfile: change FROM line to point to local path
  # FROM ~/wagmi-sft.gguf

  # Create Ollama model (override with OLLAMA_MODEL_NAME=... if your tags differ)
  ollama create {OLLAMA_MODEL_NAME} -f ~/Modelfile.wagmi-sft

  # Test
  ollama run {OLLAMA_MODEL_NAME} "C'est quoi Deal ex Machina ?"

{'='*60}
""")


if __name__ == "__main__":
    run()
