"""Minimal Gradio UI for triggering baseline eval and SFT training."""

import os
import subprocess
import sys

# Match Dockerfile: Triton/torchao must not use /.triton when HOME is / (e.g. HF Spaces).
cache_base = os.environ.get("CACHE_BASE_DIR", "/data")
if not os.access(cache_base, os.W_OK):
    cache_base = "/tmp"
os.environ.setdefault("TRITON_CACHE_DIR", f"{cache_base}/triton_cache")
os.environ.setdefault("XDG_CACHE_HOME", f"{cache_base}/.cache")
if not os.environ.get("HOME") or os.environ.get("HOME") == "/":
    os.environ["HOME"] = "/tmp"
for _cache_dir in (os.environ["TRITON_CACHE_DIR"], os.environ["XDG_CACHE_HOME"]):
    try:
        os.makedirs(_cache_dir, exist_ok=True)
    except OSError:
        pass

import gradio as gr
import torch

os.environ["PYTHONUNBUFFERED"] = "1"
PROFILE_CHOICES = ["small", "auth"]


def gpu_info():
    if not torch.cuda.is_available():
        return "No GPU detected."
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} — {mem:.1f} GB VRAM"


def run_script(script: str, profile: str = "small"):
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "MODEL_PROFILE": profile},
    )
    output = ""
    for line in iter(proc.stdout.readline, ""):
        output += line
        yield output
    proc.wait()
    if proc.returncode != 0:
        output += f"\n\nProcess exited with code {proc.returncode}"
    else:
        output += "\n\nDone."
    yield output


def run_baseline(profile: str):
    yield from run_script("baseline.py", profile)


def run_training(profile: str):
    yield from run_script("train.py", profile)


def run_eval_sft(profile: str):
    yield from run_script("eval_sft.py", profile)


def run_eval_rag(profile: str):
    yield from run_script("eval_sft_rag.py", profile)


def run_eval_tools(profile: str):
    yield from run_script("eval_tool_calls.py", profile)


def run_autotune(profile: str):
    if profile == "auth":
        yield (
            "Autotune is currently configured for the small profile only "
            "(Qwen judge-correct-retrain loop). "
            "For auth profile, run: Train -> Eval SFT -> Eval SFT+RAG.\n\n"
            "If needed, we can add an auth-specific autotune loop next."
        )
        return
    yield from run_script("autotune.py", profile)


def run_export_gguf(profile: str):
    yield from run_script("export_gguf.py", profile)


with gr.Blocks(title="sft-wagmi") as demo:
    gr.Markdown("# sft-wagmi")
    gr.Markdown(f"**GPU:** {gpu_info()}")
    profile = gr.Radio(
        choices=PROFILE_CHOICES,
        value="small",
        label="Training profile",
        info="small = Qwen 1.5B (non-auth), auth = Qwen 2.5 14B (authenticated tier)",
    )
    gr.Markdown(
        "Note: this pipeline is text SFT only. You do not need to upload any image for training."
    )

    with gr.Tab("1. Baseline"):
        gr.Markdown("Run baseline eval for the selected profile model (small or auth).")
        baseline_btn = gr.Button("Run baseline", variant="primary")
        baseline_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        baseline_btn.click(fn=run_baseline, inputs=profile, outputs=baseline_out)

    with gr.Tab("2. Train"):
        gr.Markdown(
            "Fine-tune with LoRA/QLoRA on the Wagmi dataset. "
            "small: Qwen 1.5B (fast). auth: Qwen 2.5 14B (stronger tool-calling)."
        )
        train_btn = gr.Button("Run training", variant="primary")
        train_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        train_btn.click(fn=run_training, inputs=profile, outputs=train_out)

    with gr.Tab("3. Eval SFT"):
        gr.Markdown("Run fine-tuned model on same prompts and compare with baseline.")
        eval_btn = gr.Button("Run eval", variant="primary")
        eval_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_btn.click(fn=run_eval_sft, inputs=profile, outputs=eval_out)

    with gr.Tab("4. Eval SFT+RAG"):
        gr.Markdown("Run fine-tuned model WITH RAG context injection (simulates production). Compares SFT vs SFT+RAG.")
        eval_rag_btn = gr.Button("Run eval with RAG", variant="primary")
        eval_rag_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_rag_btn.click(fn=run_eval_rag, inputs=profile, outputs=eval_rag_out)

    with gr.Tab("5. Eval Tool Calls"):
        gr.Markdown(
            "Evaluate tool-calling quality on email/calendar scenarios "
            "(JSON validity, tool-name match, required-arg coverage)."
        )
        eval_tools_btn = gr.Button("Run tool-calling eval", variant="primary")
        eval_tools_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_tools_btn.click(fn=run_eval_tools, inputs=profile, outputs=eval_tools_out)

    with gr.Tab("6. Autotune"):
        gr.Markdown(
            "**Karpathy-style self-improvement loop (small profile only).** Claude scores each response (6 criteria), "
            "generates ideal answers for failures, merges into training set, and retrains. "
            "Up to 3 iterations. Requires `OPENAI_API_KEY` in Space variables."
        )
        autotune_btn = gr.Button("Run autotune loop", variant="primary")
        autotune_out = gr.Textbox(label="Output", lines=40, max_lines=120, autoscroll=True)
        autotune_btn.click(fn=run_autotune, inputs=profile, outputs=autotune_out)

    with gr.Tab("7. Export GGUF"):
        gr.Markdown(
            "**Export to Ollama.** Merges LoRA adapter into base model, exports GGUF (Q4_K_M + Q8_0), "
            "pushes to HuggingFace Hub, and generates an Ollama Modelfile. "
            "Run this after training or autotune completes."
        )
        export_btn = gr.Button("Export GGUF", variant="primary")
        export_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        export_btn.click(fn=run_export_gguf, inputs=profile, outputs=export_out)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
