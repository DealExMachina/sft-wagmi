"""One-command launcher for the script-based sft-wagmi pipeline.

This orchestrates the same flow as the HF Gradio app:
1) optional dataset sync from dexm-one-page
2) baseline.py
3) train.py
4) autotune.py
5) export_gguf.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / ".cache" / "runs"

DATA_FILES = ["train.jsonl", "eval.jsonl", "metadata.json"]
SCRIPT_STEPS = {
    "baseline": ROOT / "baseline.py",
    "train": ROOT / "train.py",
    "autotune": ROOT / "autotune.py",
    "eval": ROOT / "eval_sft.py",
    "eval-rag": ROOT / "eval_sft_rag.py",
    "eval-tools": ROOT / "eval_tool_calls.py",
    "export": ROOT / "export_gguf.py",
}
NOTEBOOKS = {
    "baseline": ROOT / "baseline.ipynb",
    "train": ROOT / "train.ipynb",
    "autotune": ROOT / "autotune.ipynb",
}
DEXM_REPO = ROOT.parent / "dexm-one-page"
ENV_FILE = ROOT / ".env"


def command_exists(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_cmd(
    command: list[str],
    cwd: Path | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> int:
    location = str(cwd) if cwd else str(ROOT)
    printable = " ".join(command)
    print(f"[cmd] ({location}) {printable}")
    if dry_run:
        return 0
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(command, cwd=str(cwd or ROOT), env=env)
    return result.returncode


def check_required_files(paths: Iterable[Path], label: str) -> bool:
    ok = True
    for file_path in paths:
        if not file_path.exists():
            print(f"[missing] {label}: {file_path}")
            ok = False
        else:
            print(f"[ok] {label}: {file_path.name}")
    return ok


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def run_notebook(notebook_key: str, dry_run: bool, profile: str) -> int:
    notebook = NOTEBOOKS[notebook_key]
    output_name = f"{notebook.stem}.executed.ipynb"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return run_cmd(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output",
            output_name,
            "--output-dir",
            str(RUNS_DIR),
        ],
        cwd=ROOT,
        dry_run=dry_run,
        extra_env={"MODEL_PROFILE": profile},
    )

def run_python_step(step_key: str, dry_run: bool, profile: str) -> int:
    script = SCRIPT_STEPS[step_key]
    return run_cmd(
        ["python3", str(script)],
        cwd=ROOT,
        dry_run=dry_run,
        extra_env={"MODEL_PROFILE": profile},
    )

def run_pipeline_step(step_key: str, dry_run: bool, profile: str) -> int:
    if SCRIPT_STEPS[step_key].exists():
        return run_python_step(step_key, dry_run, profile)
    if step_key in NOTEBOOKS and NOTEBOOKS[step_key].exists():
        print(f"[warn] {SCRIPT_STEPS[step_key].name} missing, falling back to notebook execution.")
        if not command_exists("jupyter"):
            print("Cannot fallback to notebook: jupyter is not installed.")
            return 1
        return run_notebook(step_key, dry_run, profile)
    print(f"[missing] No runnable artifact for step '{step_key}'")
    return 1


def preflight() -> bool:
    print_header("Preflight checks")
    load_env_file(ENV_FILE)
    ok = True

    data_paths = [DATA_DIR / file_name for file_name in DATA_FILES]
    if not check_required_files(data_paths, "dataset file"):
        ok = False

    script_paths = [SCRIPT_STEPS["baseline"], SCRIPT_STEPS["train"], SCRIPT_STEPS["autotune"], SCRIPT_STEPS["export"]]
    if not check_required_files(script_paths, "pipeline script"):
        print("[warn] Some scripts are missing; notebook fallback may be attempted where available.")
        ok = False

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[ok] HF_TOKEN detected")
    else:
        print("[warn] HF_TOKEN is not set (required for Hub pull/push)")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("[ok] OPENAI_API_KEY detected")
    else:
        print("[warn] OPENAI_API_KEY is not set (required for autotune judge)")

    if command_exists("python3"):
        print("[ok] python3 available")
    else:
        print("[missing] python3 not found in PATH")
        ok = False

    if command_exists("jupyter"):
        print("[ok] jupyter available for fallback notebook execution")
    else:
        print("[warn] jupyter not found (not needed if script pipeline is present)")

    if command_exists("npm"):
        print("[ok] npm available for dataset sync")
    else:
        print("[warn] npm not found (cross-repo dataset sync disabled)")

    return ok


def sync_dataset_from_dexm(dry_run: bool) -> int:
    if not DEXM_REPO.exists():
        print(f"[skip] dexm-one-page repo not found at: {DEXM_REPO}")
        return 0
    return run_cmd(
        ["npm", "run", "dataset:wagmi:refresh"],
        cwd=DEXM_REPO,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sft-wagmi pipeline steps")
    parser.add_argument(
        "--profile",
        choices=["small", "auth"],
        default=os.environ.get("MODEL_PROFILE", "small"),
        help="Model profile to run (small or auth)",
    )
    parser.add_argument("--preflight", action="store_true", help="Run environment and file checks")
    parser.add_argument("--sync-dataset", action="store_true", help="Sync dataset from ../dexm-one-page")
    parser.add_argument("--baseline", action="store_true", help="Execute baseline step")
    parser.add_argument("--train", action="store_true", help="Execute training step")
    parser.add_argument("--autotune", action="store_true", help="Execute autotune step")
    parser.add_argument("--eval", action="store_true", help="Execute eval_sft.py")
    parser.add_argument("--eval-rag", action="store_true", help="Execute eval_sft_rag.py")
    parser.add_argument("--eval-tools", action="store_true", help="Execute eval_tool_calls.py")
    parser.add_argument("--export", action="store_true", help="Execute export step")
    parser.add_argument("--all", action="store_true", help="Run sync + baseline + train + autotune + eval + eval-rag + export")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = args.profile
    os.environ["MODEL_PROFILE"] = profile

    run_all = args.all
    do_preflight = args.preflight or run_all
    do_sync = args.sync_dataset or run_all
    do_baseline = args.baseline or run_all
    do_train = args.train or run_all
    do_autotune = args.autotune or run_all
    do_eval = args.eval or run_all
    do_eval_rag = args.eval_rag or run_all
    do_eval_tools = args.eval_tools or run_all
    do_export = args.export or run_all

    if not any([do_preflight, do_sync, do_baseline, do_train, do_autotune, do_eval, do_eval_rag, do_eval_tools, do_export]):
        print("No step selected. Use --preflight, --all, or explicit step flags.")
        return 1

    if do_preflight:
        preflight_ok = preflight()
        if not preflight_ok:
            print("\nPreflight found blocking issues.")
            if not args.dry_run:
                return 1
            print("Continuing due to --dry-run.")

    if do_sync:
        print_header("Dataset sync")
        if sync_dataset_from_dexm(args.dry_run) != 0:
            return 1

    if do_baseline:
        print_header(f"Baseline evaluation ({profile})")
        if run_pipeline_step("baseline", args.dry_run, profile) != 0:
            return 1

    if do_train:
        print_header(f"SFT training ({profile})")
        if run_pipeline_step("train", args.dry_run, profile) != 0:
            return 1

    if do_autotune:
        print_header(f"Autotune ({profile})")
        if run_pipeline_step("autotune", args.dry_run, profile) != 0:
            return 1

    if do_eval:
        print_header(f"Eval SFT ({profile})")
        if run_pipeline_step("eval", args.dry_run, profile) != 0:
            return 1

    if do_eval_rag:
        print_header(f"Eval SFT + RAG ({profile})")
        if run_pipeline_step("eval-rag", args.dry_run, profile) != 0:
            return 1

    if do_eval_tools:
        print_header(f"Eval Tool Calls ({profile})")
        if run_pipeline_step("eval-tools", args.dry_run, profile) != 0:
            return 1

    if do_export:
        print_header(f"GGUF export ({profile})")
        if run_pipeline_step("export", args.dry_run, profile) != 0:
            return 1

    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
