"""Download LoRA adapter from Hub, merge into base model on CPU, convert to GGUF, create Ollama model.

Single script that runs locally on Mac/Linux. No GPU needed (1.5B model fits in ~6 GB RAM).

    pip install torch transformers peft huggingface_hub
    HF_TOKEN=hf_xxx python3.11 scripts/export_ollama.py

Requires:
  - Ollama >= 0.5  (https://ollama.com)
  - llama.cpp      (brew install llama.cpp)
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ADAPTER_HUB_ID = os.environ.get(
    "ADAPTER_HUB_ID", "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"
)
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "wagmi-sft")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
QUANTIZATION = os.environ.get("QUANTIZATION", "Q4_K_M")
CACHE_DIR = Path(".cache/wagmi-merge")

CONVERT_SCRIPT = "/opt/homebrew/bin/convert_hf_to_gguf.py"
QUANTIZE_BIN = "llama-quantize"

SYSTEM_PROMPT = (
    "Tu es Wagmi, le watchdog de Deal ex Machina. "
    "Réponds de manière factuelle, concise, sans invention. "
    "Si l'information manque, dis clairement: 'Je ne sais pas avec certitude'."
)

def build_modelfile(gguf_path: str) -> str:
    lines = [
        f"FROM {gguf_path}",
        "",
        'TEMPLATE """{{- if .System }}<|im_start|>system',
        "{{ .System }}<|im_end|>",
        "{{ end }}<|im_start|>user",
        "{{ .Prompt }}<|im_end|>",
        "<|im_start|>assistant",
        '{{ .Response }}<|im_end|>',
        '"""',
        "",
        "PARAMETER num_ctx 2048",
        "PARAMETER num_predict 220",
        "PARAMETER temperature 0.2",
        "PARAMETER top_k 30",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.12",
        "PARAMETER repeat_last_n 128",
        'PARAMETER stop "<|im_end|>"',
        'PARAMETER stop "<|im_start|>"',
        "",
        f'SYSTEM """{SYSTEM_PROMPT}"""',
    ]
    return "\n".join(lines) + "\n"

SMOKE_PROMPT = "C'est quoi Deal ex Machina ?"


def check_deps():
    missing = []
    for pkg in ["torch", "transformers", "peft", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"Missing packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    for tool, hint in [
        ("ollama", "Install from https://ollama.com"),
        (QUANTIZE_BIN, "brew install llama.cpp"),
    ]:
        r = subprocess.run(["which", tool], capture_output=True)
        if r.returncode != 0:
            print(f"{tool} not found. {hint}", file=sys.stderr)
            sys.exit(1)

    if not Path(CONVERT_SCRIPT).exists():
        print(f"{CONVERT_SCRIPT} not found. brew install llama.cpp", file=sys.stderr)
        sys.exit(1)

    r = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    print(f"Ollama: {r.stdout.strip()}")
    print(f"llama-quantize: {shutil.which(QUANTIZE_BIN)}")


def merge_model() -> Path:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = CACHE_DIR / "merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Downloading adapter from {ADAPTER_HUB_ID} ...")
    adapter_dir = snapshot_download(
        repo_id=ADAPTER_HUB_ID,
        token=HF_TOKEN or None,
        local_dir=str(CACHE_DIR / "adapter"),
    )

    print(f"[2/5] Loading base model {BASE_MODEL_ID} (CPU) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map="cpu",
        token=HF_TOKEN or None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN or None,
    )

    print("[2/5] Applying LoRA and merging weights ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = model.merge_and_unload()

    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    param_count = sum(p.numel() for p in merged.parameters()) / 1e6
    print(f"      Merged: {param_count:.0f}M parameters -> {merged_dir}")

    del merged, model, base_model
    return merged_dir


def convert_to_gguf(merged_dir: Path) -> Path:
    f16_gguf = CACHE_DIR / "wagmi-f16.gguf"
    quantized_gguf = CACHE_DIR / f"wagmi-{QUANTIZATION.lower()}.gguf"

    print(f"\n[3/5] Converting safetensors -> GGUF (f16) ...")
    result = subprocess.run(
        [sys.executable, CONVERT_SCRIPT, str(merged_dir),
         "--outfile", str(f16_gguf), "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout[-500:])
    if result.returncode != 0:
        print(f"convert_hf_to_gguf STDERR:\n{result.stderr[-1000:]}", file=sys.stderr)
        sys.exit(1)
    print(f"      {f16_gguf} ({f16_gguf.stat().st_size / 1e6:.0f} MB)")

    print(f"[4/5] Quantizing -> {QUANTIZATION} ...")
    subprocess.run(
        [QUANTIZE_BIN, str(f16_gguf), str(quantized_gguf), QUANTIZATION],
        check=True,
    )
    print(f"      {quantized_gguf} ({quantized_gguf.stat().st_size / 1e6:.0f} MB)")

    f16_gguf.unlink(missing_ok=True)
    return quantized_gguf


def create_ollama_model(gguf_path: Path):
    content = build_modelfile(str(gguf_path.resolve()))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False, encoding="utf-8") as f:
        f.write(content)
        modelfile_path = f.name

    try:
        print(f"\n[5/5] Creating Ollama model '{OLLAMA_MODEL_NAME}' ...")
        subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path],
            check=True,
        )
        print(f"      Model '{OLLAMA_MODEL_NAME}' created.")
    finally:
        os.unlink(modelfile_path)


def smoke_test():
    print(f"\nSmoke test: '{SMOKE_PROMPT}'")
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL_NAME, SMOKE_PROMPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"Response:\n{result.stdout.strip()}")
    else:
        err = result.stderr.strip() or result.stdout.strip() or "(no output)"
        print(f"Smoke test issue: {err}", file=sys.stderr)
        print("Try manually: ollama run wagmi-sft")


def cleanup():
    if CACHE_DIR.exists():
        size_mb = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()) / 1e6
        print(f"\nCleaning up {CACHE_DIR} ({size_mb:.0f} MB) ...")
        shutil.rmtree(CACHE_DIR)


def run():
    check_deps()

    if HF_TOKEN:
        print(f"HF_TOKEN: {HF_TOKEN[:8]}...")
    else:
        print("WARNING: HF_TOKEN not set — will fail on private repos.")

    subprocess.run(["ollama", "rm", OLLAMA_MODEL_NAME],
                   capture_output=True)

    merged_dir = merge_model()
    gguf_path = convert_to_gguf(merged_dir)
    create_ollama_model(gguf_path)
    smoke_test()
    cleanup()

    print(f"\n{'=' * 60}")
    print(f"  ollama run {OLLAMA_MODEL_NAME}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
