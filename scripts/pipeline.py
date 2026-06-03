"""One-command launcher for the sft-wagmi pipeline.

Steps are selected with explicit flags or with `--all`. Available steps:
`preflight`, dataset `sync-dataset`, `merge-next` (bumps VERSION when
`data/next/*.jsonl` exist), `baseline`, `train`, `autotune` (needs
`OPENAI_API_KEY`), `eval`, `eval-rag`, `eval-tools`, `redteam`,
`export-merged`, `export-gguf`.

`--all` enables: preflight → merge-next → train → eval → eval-rag → redteam →
export-merged. It does **not** run `sync-dataset`, `baseline`, `autotune`,
`eval-tools`, or `export-gguf` unless those flags are added.

Local GGUF after Hub export: `scripts/local_gguf_export.sh` (manual).

Usage:
  python3 scripts/pipeline.py --all --profile auth
  python3 scripts/pipeline.py --merge-next --train --export-merged --profile auth
  python3 scripts/pipeline.py --preflight --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NEXT_DIR = DATA_DIR / "next"
RUNS_DIR = ROOT / ".cache" / "runs"
VERSION_FILE = ROOT / "VERSION"

DATA_FILES = ["train.jsonl", "eval.jsonl", "metadata.json"]
SCRIPT_STEPS = {
    "baseline": ROOT / "baseline.py",
    "train": ROOT / "train.py",
    "autotune": ROOT / "autotune.py",
    "eval": ROOT / "eval_sft.py",
    "eval-rag": ROOT / "eval_sft_rag.py",
    "eval-tools": ROOT / "eval_tool_calls.py",
    "redteam": ROOT / "eval_redteam.py",
    "export-merged": ROOT / "export_merged.py",
    "export-gguf": ROOT / "export_gguf.py",
}
NOTEBOOKS = {
    "baseline": ROOT / "baseline.ipynb",
    "train": ROOT / "train.ipynb",
    "autotune": ROOT / "autotune.ipynb",
}
DEXM_REPO = ROOT.parent / "dexm-one-page"
ENV_FILE = ROOT / ".env"


def get_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    return "unknown"


def command_exists(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def print_header(title: str) -> None:
    version = get_version()
    print(f"\n{'=' * 72}\n  [{version}] {title}\n{'=' * 72}")


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
            print(f"  [missing] {label}: {file_path}")
            ok = False
        else:
            print(f"  [ok] {label}: {file_path.name}")
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


def run_notebook(notebook_key: str, dry_run: bool, profile: str, family: str) -> int:
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
        extra_env={"MODEL_PROFILE": profile, "LLM_FAMILY": family},
    )


def run_python_step(step_key: str, dry_run: bool, profile: str, family: str) -> int:
    script = SCRIPT_STEPS[step_key]
    return run_cmd(
        ["python3", str(script)],
        cwd=ROOT,
        dry_run=dry_run,
        extra_env={"MODEL_PROFILE": profile, "LLM_FAMILY": family},
    )


def run_pipeline_step(step_key: str, dry_run: bool, profile: str, family: str) -> int:
    if SCRIPT_STEPS[step_key].exists():
        return run_python_step(step_key, dry_run, profile, family)
    if step_key in NOTEBOOKS and NOTEBOOKS[step_key].exists():
        print(f"  [warn] {SCRIPT_STEPS[step_key].name} missing, falling back to notebook.")
        if not command_exists("jupyter"):
            print("  Cannot fallback to notebook: jupyter is not installed.")
            return 1
        return run_notebook(step_key, dry_run, profile, family)
    print(f"  [missing] No runnable artifact for step '{step_key}'")
    return 1


def preflight(profile: str) -> bool:
    print_header(f"Preflight checks (profile={profile})")
    load_env_file(ENV_FILE)
    ok = True

    data_paths = [DATA_DIR / f for f in DATA_FILES]
    if not check_required_files(data_paths, "dataset"):
        ok = False

    next_files = list(NEXT_DIR.glob("*.jsonl"))
    if next_files:
        total = sum(1 for f in next_files for _ in f.read_text().strip().split("\n") if _.strip())
        print(f"  [info] data/next/: {len(next_files)} file(s), {total} entries pending merge")
    else:
        print(f"  [info] data/next/: empty (nothing to merge)")

    core_scripts = ["train", "eval", "export-merged"]
    for key in core_scripts:
        p = SCRIPT_STEPS[key]
        if p.exists():
            print(f"  [ok] {p.name}")
        else:
            print(f"  [missing] {p.name}")
            ok = False

    hf_token = os.environ.get("HF_TOKEN")
    print(f"  [{'ok' if hf_token else 'warn'}] HF_TOKEN {'detected' if hf_token else 'NOT set'}")

    openai_key = os.environ.get("OPENAI_API_KEY")
    print(f"  [{'ok' if openai_key else 'warn'}] OPENAI_API_KEY {'detected' if openai_key else 'NOT set (autotune disabled)'}")

    if command_exists("python3"):
        print("  [ok] python3")
    else:
        print("  [missing] python3")
        ok = False

    return ok


def merge_next(dry_run: bool, bump: str = "minor") -> int:
    merge_script = ROOT / "scripts" / "merge_next.py"
    if not merge_script.exists():
        print("  [missing] scripts/merge_next.py")
        return 1

    next_files = list(NEXT_DIR.glob("*.jsonl"))
    if not next_files:
        print("  No files in data/next/ — skipping merge.")
        return 0

    cmd = ["python3", str(merge_script), "--bump", bump]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd, cwd=ROOT, dry_run=False)


def sync_dataset_from_dexm(dry_run: bool) -> int:
    if not DEXM_REPO.exists():
        print(f"  [skip] dexm-one-page repo not found at: {DEXM_REPO}")
        return 0
    return run_cmd(
        ["npm", "run", "dataset:wagmi:refresh"],
        cwd=DEXM_REPO,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sft-wagmi pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --all --profile auth          # full pipeline (L40)
  %(prog)s --all --profile auth --family lfm2
  %(prog)s --merge-next --train          # merge new data + retrain
  %(prog)s --profile auth --redteam      # generate AI Act red-team report
  %(prog)s --preflight --dry-run         # check environment
  %(prog)s --train --export-merged       # train + push merged model
""",
    )
    parser.add_argument("--profile", choices=["small", "auth"],
                        default=os.environ.get("MODEL_PROFILE", "small"),
                        help="Model profile (default: small)")
    parser.add_argument("--family", choices=["qwen", "lfm2"],
                        default=os.environ.get("LLM_FAMILY", "qwen"),
                        help="Base model family defaults (default: qwen)")
    parser.add_argument("--preflight", action="store_true", help="Run environment checks")
    parser.add_argument("--sync-dataset", action="store_true", help="Sync dataset from ../dexm-one-page")
    parser.add_argument("--merge-next", action="store_true", help="Merge data/next/ into training set")
    parser.add_argument("--bump", choices=["patch", "minor", "major"], default="minor",
                        help="Version bump type when merging (default: minor)")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation")
    parser.add_argument("--train", action="store_true", help="Run SFT training")
    parser.add_argument("--autotune", action="store_true", help="Run autotune judge loop")
    parser.add_argument("--eval", action="store_true", help="Run eval_sft.py")
    parser.add_argument("--eval-rag", action="store_true", help="Run eval_sft_rag.py")
    parser.add_argument("--eval-tools", action="store_true", help="Run eval_tool_calls.py")
    parser.add_argument("--redteam", action="store_true", help="Run eval_redteam.py and generate versioned report")
    parser.add_argument("--export-merged", action="store_true", help="Export merged model to Hub")
    parser.add_argument("--export-gguf", action="store_true", help="Export GGUF (legacy, on-Space)")
    parser.add_argument("--all", action="store_true",
                        help="Full pipeline: preflight -> merge-next -> train -> eval -> eval-rag -> redteam -> export-merged")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = args.profile
    family = args.family
    os.environ["MODEL_PROFILE"] = profile
    os.environ["LLM_FAMILY"] = family
    load_env_file(ENV_FILE)

    run_all = args.all
    do_preflight = args.preflight or run_all
    do_sync = args.sync_dataset
    do_merge = args.merge_next or run_all
    do_baseline = args.baseline
    do_train = args.train or run_all
    do_autotune = args.autotune
    do_eval = args.eval or run_all
    do_eval_rag = args.eval_rag or run_all
    do_eval_tools = args.eval_tools
    do_redteam = args.redteam or run_all
    do_export_merged = args.export_merged or run_all
    do_export_gguf = args.export_gguf

    steps = [do_preflight, do_sync, do_merge, do_baseline, do_train,
             do_autotune, do_eval, do_eval_rag, do_eval_tools, do_redteam,
             do_export_merged, do_export_gguf]
    if not any(steps):
        print("No step selected. Use --all or explicit step flags. See --help.")
        return 1

    print(f"\nsft-wagmi pipeline v{get_version()} | family={family} | profile={profile}")

    if do_preflight:
        if not preflight(profile):
            print("\nPreflight found blocking issues.")
            if not args.dry_run:
                return 1
            print("Continuing due to --dry-run.")

    if do_sync:
        print_header("Dataset sync from dexm-one-page")
        if sync_dataset_from_dexm(args.dry_run) != 0:
            return 1

    if do_merge:
        print_header("Merge data/next/ into training set")
        rc = merge_next(args.dry_run, args.bump)
        if rc != 0:
            return rc

    if do_baseline:
        print_header(f"Baseline evaluation ({profile})")
        if run_pipeline_step("baseline", args.dry_run, profile, family) != 0:
            return 1

    if do_train:
        print_header(f"SFT training ({profile})")
        if run_pipeline_step("train", args.dry_run, profile, family) != 0:
            return 1

    if do_autotune:
        print_header(f"Autotune ({profile})")
        if run_pipeline_step("autotune", args.dry_run, profile, family) != 0:
            return 1

    if do_eval:
        print_header(f"Eval SFT ({profile})")
        if run_pipeline_step("eval", args.dry_run, profile, family) != 0:
            return 1

    if do_eval_rag:
        print_header(f"Eval SFT + RAG ({profile})")
        if run_pipeline_step("eval-rag", args.dry_run, profile, family) != 0:
            return 1

    if do_eval_tools:
        print_header(f"Eval Tool Calls ({profile})")
        if run_pipeline_step("eval-tools", args.dry_run, profile, family) != 0:
            return 1

    if do_redteam:
        print_header(f"Red Team Guardrails ({profile})")
        if run_pipeline_step("redteam", args.dry_run, profile, family) != 0:
            return 1

    if do_export_merged:
        print_header(f"Export merged model ({profile})")
        if run_pipeline_step("export-merged", args.dry_run, profile, family) != 0:
            return 1

    if do_export_gguf:
        print_header(f"Export GGUF — legacy ({profile})")
        if run_pipeline_step("export-gguf", args.dry_run, profile, family) != 0:
            return 1

    version = get_version()
    print(f"\n{'=' * 72}")
    print(f"  Pipeline completed — v{version} ({profile})")
    print(f"{'=' * 72}")
    if do_export_merged and not args.dry_run:
        print(f"\n  Next step (local Mac):")
        print(f"    ./scripts/local_gguf_export.sh {profile}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
