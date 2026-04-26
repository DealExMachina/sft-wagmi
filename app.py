"""Minimal Gradio UI for triggering baseline eval and SFT training.

Hugging Face Docker Space constraints (see hub docs: Spaces / Docker):
- The process must listen on the port declared as ``app_port`` in README front matter
  (here: 7860). Gradio also reads ``GRADIO_SERVER_PORT`` / ``PORT`` if set — keep them
  aligned with ``app_port`` or the proxy will not reach the app.
- Bind on ``0.0.0.0`` (``GRADIO_SERVER_NAME``) so the platform can route inbound traffic.
- Overlapping restarts can leave stale listeners; we probe/cleanup before bind.
"""

import os
import socket
import subprocess
import sys
import time

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
FAMILY_CHOICES = ["qwen", "lfm2", "qwen3"]


def gpu_info():
    if not torch.cuda.is_available():
        return "No GPU detected."
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} — {mem:.1f} GB VRAM"


def run_script(script: str, profile: str = "small", family: str = "qwen"):
    extra_env = {}
    if profile == "auth" and family == "qwen":
        # Keep 14B auth runs stable on L40 (44GB) in Spaces.
        extra_env["AUTH_MAX_SEQ_LEN"] = os.environ.get("AUTH_MAX_SEQ_LEN", "1024")
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            **extra_env,
            "PYTHONUNBUFFERED": "1",
            "MODEL_PROFILE": profile,
            "LLM_FAMILY": family,
        },
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


def run_baseline(profile: str, family: str):
    yield from run_script("baseline.py", profile, family)


def run_training(profile: str, family: str):
    yield from run_script("train.py", profile, family)


def run_dpo_training(profile: str, family: str):
    yield from run_script("train_dpo.py", profile, family)


def run_grpo_training(profile: str, family: str):
    yield from run_script("train_grpo.py", profile, family)


def run_eval_sft(profile: str, family: str):
    yield from run_script("eval_sft.py", profile, family)


def run_eval_rag(profile: str, family: str):
    yield from run_script("eval_sft_rag.py", profile, family)


def run_eval_tools(profile: str, family: str):
    yield from run_script("eval_tool_calls.py", profile, family)


def run_eval_redteam(profile: str, family: str):
    yield from run_script("eval_redteam.py", profile, family)


def run_autotune(profile: str, family: str):
    yield from run_script("autotune.py", profile, family)


def run_export_merged(profile: str, family: str):
    yield from run_script("export_merged.py", profile, family)


def run_export_dpo_merged(profile: str, family: str):
    """Export merged model using DPO adapter instead of SFT adapter."""
    extra_env = {
        "AUTH_HUB_MODEL_ID": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo",
        "AUTH_HUB_MERGED_REPO": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-merged",
    }
    if profile == "auth" and family == "qwen":
        extra_env["AUTH_MAX_SEQ_LEN"] = os.environ.get("AUTH_MAX_SEQ_LEN", "1024")
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    proc = subprocess.Popen(
        [sys.executable, "-u", "export_merged.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            **extra_env,
            "PYTHONUNBUFFERED": "1",
            "MODEL_PROFILE": profile,
            "LLM_FAMILY": family,
        },
    )
    output = ""
    for line in iter(proc.stdout.readline, ""):
        output += line
        yield output
    proc.wait()
    output += f"\n\nProcess exited with code {proc.returncode}" if proc.returncode != 0 else "\n\nDone."
    yield output


def run_export_grpo_merged(profile: str, family: str):
    """Export merged model using GRPO adapter (DPO+GRPO checkpoint)."""
    extra_env = {
        "AUTH_HUB_MODEL_ID": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo",
        "AUTH_HUB_MERGED_REPO": "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo-merged",
    }
    if profile == "auth" and family == "qwen":
        extra_env["AUTH_MAX_SEQ_LEN"] = os.environ.get("AUTH_MAX_SEQ_LEN", "1024")
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    proc = subprocess.Popen(
        [sys.executable, "-u", "export_merged.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            **extra_env,
            "PYTHONUNBUFFERED": "1",
            "MODEL_PROFILE": profile,
            "LLM_FAMILY": family,
        },
    )
    output = ""
    for line in iter(proc.stdout.readline, ""):
        output += line
        yield output
    proc.wait()
    output += f"\n\nProcess exited with code {proc.returncode}" if proc.returncode != 0 else "\n\nDone."
    yield output


def run_export_gguf(profile: str, family: str):
    yield from run_script("export_gguf.py", profile, family)


def run_merge_next():
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/merge_next.py", "--bump", "minor"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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


def run_full_pipeline(profile: str, family: str):
    extra_env = {}
    if profile == "auth" and family == "qwen":
        extra_env["AUTH_MAX_SEQ_LEN"] = os.environ.get("AUTH_MAX_SEQ_LEN", "1024")
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/pipeline.py", "--all", "--profile", profile, "--family", family],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            **extra_env,
            "PYTHONUNBUFFERED": "1",
            "LLM_FAMILY": family,
            "MODEL_PROFILE": profile,
        },
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


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    try:
        with open(version_file) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "?"


def _docker_space_listen_port() -> int:
    """Port Gradio binds to — must match README ``app_port`` for Docker Spaces."""
    for key in ("GRADIO_SERVER_PORT", "PORT"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return int(str(raw).strip(), 10)
    return 7860


def _wait_until_port_free(port: int, *, host: str = "0.0.0.0", timeout_s: float = 90.0) -> bool:
    """Best-effort wait for an available TCP port before Gradio launch."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            return True
        except OSError:
            time.sleep(0.75)
    return False


def _list_listening_pids(port: int) -> list[int]:
    """Return PIDs listening on ``port`` (best effort; empty on failure)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids: list[int] = []
    for raw in out.splitlines():
        raw = raw.strip()
        if raw.isdigit():
            pid = int(raw)
            if pid != os.getpid():
                pids.append(pid)
    return pids


def _terminate_stale_listeners(port: int) -> None:
    """Terminate stale listeners to break HF restart loops (best effort)."""
    pids = _list_listening_pids(port)
    if not pids:
        return
    print(f"Port {port}: terminating stale listeners {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, 15)  # SIGTERM
        except OSError:
            pass
    time.sleep(2.0)
    for pid in _list_listening_pids(port):
        try:
            os.kill(pid, 9)  # SIGKILL
        except OSError:
            pass


with gr.Blocks(title="sft-wagmi") as demo:
    gr.Markdown(f"# sft-wagmi v{get_version()}")
    gr.Markdown(f"**GPU:** {gpu_info()}")
    profile = gr.Radio(
        choices=PROFILE_CHOICES,
        value="small",
        label="Training profile",
        info="small = anon-tier model, auth = tool-capable authenticated-tier model",
    )
    family = gr.Radio(
        choices=FAMILY_CHOICES,
        value="qwen",
        label="Model family",
        info="qwen = Qwen 2.5 path, qwen3 = Qwen 3 path, lfm2 = Liquid AI LFM2/LFM2.5 path",
    )
    gr.Markdown(
        "Note: this pipeline is text SFT only. You do not need to upload any image for training."
    )

    with gr.Tab("FULL PIPELINE"):
        gr.Markdown(
            "**One-click full pipeline:** preflight → merge data/next/ → train → eval → eval+RAG → export merged.\n\n"
            "Select the profile above, then hit the button. Sit back."
        )
        full_btn = gr.Button("Run full pipeline", variant="primary", size="lg")
        full_out = gr.Textbox(label="Output", lines=40, max_lines=200, autoscroll=True)
        full_btn.click(fn=run_full_pipeline, inputs=[profile, family], outputs=full_out)

    with gr.Tab("0. Merge Next Data"):
        gr.Markdown(
            "**Merge `data/next/*.jsonl` into the training dataset.** "
            "Validates schema, splits into train/eval (85/15), updates metadata, "
            "bumps VERSION (minor). Run this before training if new entries are staged."
        )
        merge_btn = gr.Button("Merge & bump version", variant="primary")
        merge_out = gr.Textbox(label="Output", lines=20, max_lines=50, autoscroll=True)
        merge_btn.click(fn=run_merge_next, outputs=merge_out)

    with gr.Tab("1. Baseline"):
        gr.Markdown("Run baseline eval for the selected profile model (small or auth).")
        baseline_btn = gr.Button("Run baseline", variant="primary")
        baseline_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        baseline_btn.click(fn=run_baseline, inputs=[profile, family], outputs=baseline_out)

    with gr.Tab("2. Train (SFT)"):
        gr.Markdown(
            "Fine-tune with LoRA/QLoRA on the Wagmi dataset. "
            "Model family and profile are selected above. "
            "small = anon-tier base model, auth = tool-capable authenticated-tier model."
        )
        train_btn = gr.Button("Run SFT training", variant="primary")
        train_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        train_btn.click(fn=run_training, inputs=[profile, family], outputs=train_out)

    with gr.Tab("2b. Train (DPO)"):
        gr.Markdown(
            "**DPO safety alignment** — starts from the last SFT adapter and trains on "
            "`data/dpo/wagmi_safety_dpo.jsonl` (81 chosen/rejected pairs). "
            "Use after SFT when `must_refuse` gate fails. ~25 min on L40."
        )
        dpo_btn = gr.Button("Run DPO training", variant="primary")
        dpo_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        dpo_btn.click(fn=run_dpo_training, inputs=[profile, family], outputs=dpo_out)

    with gr.Tab("2c. Train (GRPO)"):
        gr.Markdown(
            "**GRPO safety alignment** — Phase 2 fallback if DPO is insufficient. "
            "Starts from the DPO checkpoint. Uses a binary reward function that rewards "
            "explicit refusals on attacks. ~90 min on L40."
        )
        grpo_btn = gr.Button("Run GRPO training", variant="primary")
        grpo_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        grpo_btn.click(fn=run_grpo_training, inputs=[profile, family], outputs=grpo_out)

    with gr.Tab("3. Eval SFT"):
        gr.Markdown("Run fine-tuned model on same prompts and compare with baseline.")
        eval_btn = gr.Button("Run eval", variant="primary")
        eval_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_btn.click(fn=run_eval_sft, inputs=[profile, family], outputs=eval_out)

    with gr.Tab("4. Eval SFT+RAG"):
        gr.Markdown("Run fine-tuned model WITH RAG context injection (simulates production). Compares SFT vs SFT+RAG.")
        eval_rag_btn = gr.Button("Run eval with RAG", variant="primary")
        eval_rag_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_rag_btn.click(fn=run_eval_rag, inputs=[profile, family], outputs=eval_rag_out)

    with gr.Tab("5. Eval Tool Calls"):
        gr.Markdown(
            "Evaluate tool-calling quality on email/calendar scenarios "
            "(JSON validity, tool-name match, required-arg coverage)."
        )
        eval_tools_btn = gr.Button("Run tool-calling eval", variant="primary")
        eval_tools_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_tools_btn.click(fn=run_eval_tools, inputs=[profile, family], outputs=eval_tools_out)

    with gr.Tab("5b. Eval Red Team"):
        gr.Markdown(
            "**Guardrail / red-team gate** — runs `eval_redteam.py` (must_refuse, URLs, emails). "
            "Set `EVAL_API_*` in Space secrets to evaluate a remote Koyeb endpoint; otherwise loads the local adapter."
        )
        eval_red_btn = gr.Button("Run red-team eval", variant="primary")
        eval_red_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        eval_red_btn.click(fn=run_eval_redteam, inputs=[profile, family], outputs=eval_red_out)

    with gr.Tab("6. Autotune"):
        gr.Markdown(
            "**Karpathy-style self-improvement loop (small + auth).** GPT-4o scores each response (6 criteria), "
            "generates ideal answers for failures, merges into training set, and retrains. "
            "Up to 3 iterations. Requires `OPENAI_API_KEY` in Space variables."
        )
        autotune_btn = gr.Button("Run autotune loop", variant="primary")
        autotune_out = gr.Textbox(label="Output", lines=40, max_lines=120, autoscroll=True)
        autotune_btn.click(fn=run_autotune, inputs=[profile, family], outputs=autotune_out)

    with gr.Tab("7. Export Merged (SFT)"):
        gr.Markdown(
            "**Merge SFT LoRA + push to Hub.** Merges the SFT adapter into the base model "
            "and uploads merged safetensors to HuggingFace Hub. "
            "GGUF conversion runs locally on your Mac via `scripts/local_gguf_export.sh`."
        )
        export_merged_btn = gr.Button("Export Merged Model (SFT)", variant="primary")
        export_merged_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        export_merged_btn.click(fn=run_export_merged, inputs=[profile, family], outputs=export_merged_out)

    with gr.Tab("7c. Export Merged (DPO)"):
        gr.Markdown(
            "**Merge DPO adapter + push to Hub.** Uses `jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo` "
            "as the adapter source and pushes merged safetensors to `wagmi-qwen2.5-14b-sft-dpo-merged`. "
            "Run this after DPO training. Then run `scripts/local_gguf_export.sh auth-dpo` locally."
        )
        export_dpo_btn = gr.Button("Export Merged Model (DPO)", variant="primary")
        export_dpo_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        export_dpo_btn.click(fn=run_export_dpo_merged, inputs=[profile, family], outputs=export_dpo_out)

    with gr.Tab("7d. Export Merged (GRPO)"):
        gr.Markdown(
            "**Merge GRPO adapter + push to Hub.** Uses `jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo` "
            "as the adapter source and pushes merged safetensors to "
            "`wagmi-qwen2.5-14b-sft-dpo-grpo-merged`. "
            "Run this after GRPO training. Then run `scripts/local_gguf_export.sh auth-grpo` locally."
        )
        export_grpo_btn = gr.Button("Export Merged Model (GRPO)", variant="primary")
        export_grpo_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        export_grpo_btn.click(fn=run_export_grpo_merged, inputs=[profile, family], outputs=export_grpo_out)

    with gr.Tab("7b. Export GGUF (legacy)"):
        gr.Markdown(
            "**Legacy: full GGUF export on Space.** Requires llama.cpp in the Docker image. "
            "Prefer tab 7 (merged export) + local GGUF conversion instead."
        )
        export_btn = gr.Button("Export GGUF", variant="secondary")
        export_out = gr.Textbox(label="Output", lines=30, max_lines=80, autoscroll=True)
        export_btn.click(fn=run_export_gguf, inputs=[profile, family], outputs=export_out)


if __name__ == "__main__":
    listen_host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0").strip() or "0.0.0.0"
    listen_port = _docker_space_listen_port()
    print(
        f"Gradio launch: server_name={listen_host!r} server_port={listen_port} "
        f"(HF Docker: align with README app_port; optional env GRADIO_SERVER_PORT / PORT)",
        flush=True,
    )
    if not _wait_until_port_free(listen_port, host=listen_host):
        print(
            f"Port {listen_port} still busy after initial wait; attempting stale-listener cleanup.",
            flush=True,
        )
        _terminate_stale_listeners(listen_port)
        if not _wait_until_port_free(listen_port, host=listen_host, timeout_s=20.0):
            print(
                f"Port {listen_port} remains busy; launching anyway and letting Gradio surface the error.",
                flush=True,
            )
    demo.queue().launch(server_name=listen_host, server_port=listen_port)
