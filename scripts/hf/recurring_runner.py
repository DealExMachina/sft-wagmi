"""Recurring orchestrator for sft-wagmi training on HF Space.

Phase 2 goals:
- run configurable daily/weekly matrices from JSON config
- execute scripts/pipeline.py with explicit profile/family flags
- support per-run env and timeout control
- classify failures (infra / data / model_quality / unknown)
- persist logs + JSON summaries under reports/recurring/
- optionally post run summary comments to a GitHub issue via gh CLI
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "recurring_runs.json"
DEFAULT_TEMPLATE_PATH = ROOT / "scripts" / "hf" / "templates" / "recurring_issue_comment.md"
REPORTS_DIR = ROOT / "reports" / "recurring"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    cadence: str
    family: str
    profile: str
    name: str
    enabled: bool
    skip_if_no_pending_data: bool
    sync_dataset: bool
    merge_next: str
    bump: str
    pipeline_flags: list[str]
    timeout_minutes: int | None
    env: dict[str, str]
    continue_on_failure: bool
    backend: str
    hf_job_image: str | None
    hf_job_flavor: str | None
    hf_job_secrets: list[str]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ts_slug(ts: dt.datetime) -> str:
    return ts.strftime("%Y%m%d-%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_specs(config_path: Path) -> tuple[dict[str, Any], list[RunSpec]]:
    raw = read_json(config_path)
    runs_raw = raw.get("runs", [])
    specs: list[RunSpec] = []
    for item in runs_raw:
        specs.append(
            RunSpec(
                run_id=item["id"],
                cadence=item["cadence"],
                family=item["family"],
                profile=item["profile"],
                name=str(item.get("name", item["id"])),
                enabled=bool(item.get("enabled", True)),
                skip_if_no_pending_data=bool(item.get("skip_if_no_pending_data", False)),
                sync_dataset=bool(item.get("sync_dataset", False)),
                merge_next=str(item.get("merge_next", "auto")).lower(),
                bump=str(item.get("bump", "minor")).lower(),
                pipeline_flags=list(item.get("pipeline_flags", [])),
                timeout_minutes=(
                    int(item["timeout_minutes"])
                    if item.get("timeout_minutes") is not None
                    else None
                ),
                env={str(k): str(v) for k, v in dict(item.get("env", {})).items()},
                continue_on_failure=bool(item.get("continue_on_failure", True)),
                backend=str(item.get("backend", "local")).lower(),
                hf_job_image=(
                    str(item["hf_job_image"]).strip()
                    if item.get("hf_job_image")
                    else None
                ),
                hf_job_flavor=(
                    str(item["hf_job_flavor"]).strip()
                    if item.get("hf_job_flavor")
                    else None
                ),
                hf_job_secrets=[str(x) for x in list(item.get("hf_job_secrets", []))],
            )
        )
    return raw, specs


def pending_next_stats() -> tuple[int, int]:
    next_dir = ROOT / "data" / "next"
    files = sorted(next_dir.glob("*.jsonl"))
    line_count = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                line_count += 1
    return len(files), line_count


def should_merge(spec: RunSpec, has_pending: bool) -> bool:
    if spec.merge_next == "always":
        return True
    if spec.merge_next == "never":
        return False
    return has_pending


def build_pipeline_cmd(spec: RunSpec, has_pending: bool) -> list[str]:
    cmd = [
        "python3",
        "scripts/pipeline.py",
        "--profile",
        spec.profile,
        "--family",
        spec.family,
        "--preflight",
    ]
    if spec.sync_dataset:
        cmd.append("--sync-dataset")
    if should_merge(spec, has_pending):
        cmd.extend(["--merge-next", "--bump", spec.bump])
    cmd.extend(spec.pipeline_flags)
    return cmd


def build_hf_jobs_cmd(spec: RunSpec, has_pending: bool) -> list[str]:
    if not spec.hf_job_image:
        raise ValueError(
            f"Run {spec.run_id} uses backend=hf_jobs but hf_job_image is not configured."
        )
    pipeline_cmd = build_pipeline_cmd(spec, has_pending)
    cmd = ["hf", "jobs", "run"]
    if spec.hf_job_flavor:
        cmd.extend(["--flavor", spec.hf_job_flavor])
    if spec.timeout_minutes is not None:
        cmd.extend(["--timeout", f"{spec.timeout_minutes}m"])
    for secret_key in spec.hf_job_secrets:
        cmd.extend(["--secrets", f"{secret_key}=${secret_key}"])
    cmd.append(spec.hf_job_image)
    cmd.extend(pipeline_cmd)
    return cmd


def run_command(
    cmd: list[str],
    dry_run: bool,
    timeout_minutes: int | None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        return 0, f"[dry-run] {printable}\n"
    run_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if extra_env:
        run_env.update(extra_env)
    timeout_seconds = timeout_minutes * 60 if timeout_minutes is not None else None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            env=run_env,
            timeout=timeout_seconds,
        )
        output = f"$ {printable}\n\n{proc.stdout}\n{proc.stderr}".strip() + "\n"
        return proc.returncode, output
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        output = (
            f"$ {printable}\n\n{out}\n{err}\n"
            f"[timeout] command exceeded {timeout_minutes} minute(s)\n"
        )
        return 124, output


def select_specs(specs: list[RunSpec], cadence: str | None, run_id: str | None) -> list[RunSpec]:
    selected: list[RunSpec] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if cadence and spec.cadence != cadence:
            continue
        if run_id and spec.run_id != run_id:
            continue
        selected.append(spec)
    return selected


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_template(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (
        "## Recurring Training Run - {run_timestamp}\n\n"
        "- Trigger: `{trigger}`\n"
        "- Cadence: `{cadence}`\n"
        "- Runs: `{run_count}`\n"
        "- Overall: `{overall_status}`\n\n"
        "{run_rows}\n"
    )


def row_for_summary(result: dict[str, Any]) -> str:
    status = result["status"]
    icon = {
        "success": "PASS",
        "failed": "FAIL",
        "skipped": "SKIP",
    }.get(status, status.upper())
    category = result.get("failure_category")
    category_chunk = f" [{category}]" if category else ""
    return (
        f"- `{result['run_id']}` [{icon}] "
        f"(family={result['family']}, profile={result['profile']}){category_chunk} - "
        f"{result.get('note', '')} "
        f"[log]({result['log_relpath']})"
    ).strip()


def classify_failure(log_text: str, exit_code: int) -> str:
    lower = log_text.lower()
    infra_markers = [
        "cuda out of memory",
        "runtimeerror: cuda",
        "nccl",
        "timed out",
        "[timeout]",
        "temporary failure",
        "connection reset",
        "connection refused",
        "name resolution",
        "no space left on device",
        "killed",
    ]
    data_markers = [
        "jsondecodeerror",
        "file not found",
        "filenotfounderror",
        "missing] dataset",
        "schema",
        "invalid json",
        "unsupported model_profile",
        "unsupported llm_family",
    ]
    model_quality_markers = [
        "red team",
        "redteam",
        "guardrail",
        "must_refuse",
        "regression",
        "quality gate",
        "eval failed",
        "threshold",
    ]

    if any(marker in lower for marker in infra_markers):
        return "infra"
    if any(marker in lower for marker in data_markers):
        return "data"
    if any(marker in lower for marker in model_quality_markers):
        return "model_quality"

    if exit_code in (124, 137, 143):
        return "infra"
    return "unknown"


def render_issue_comment(
    template: str,
    run_timestamp: str,
    trigger: str,
    cadence: str,
    results: list[dict[str, Any]],
    summary_relpath: str,
) -> str:
    run_rows = "\n".join(row_for_summary(r) for r in results)
    overall_status = "success"
    if any(r["status"] == "failed" for r in results):
        overall_status = "failed"
    elif all(r["status"] == "skipped" for r in results):
        overall_status = "skipped"
    failure_buckets: dict[str, int] = {}
    for result in results:
        bucket = result.get("failure_category")
        if bucket:
            failure_buckets[bucket] = failure_buckets.get(bucket, 0) + 1
    if failure_buckets:
        failure_breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(failure_buckets.items()))
    else:
        failure_breakdown = "none"

    body = template.format(
        run_timestamp=run_timestamp,
        trigger=trigger,
        cadence=cadence,
        run_count=len(results),
        overall_status=overall_status,
        run_rows=run_rows,
        summary_relpath=summary_relpath,
        failure_breakdown=failure_breakdown,
    )
    if "summary_relpath" not in template:
        body += f"\n\n- Summary JSON: `{summary_relpath}`\n"
    return body


def can_post_with_gh() -> bool:
    return bool(shutil.which("gh")) and bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))


def post_issue_comment(repo: str, issue_number: int, body_path: Path, dry_run: bool) -> tuple[bool, str]:
    cmd = [
        "gh",
        "issue",
        "comment",
        str(issue_number),
        "--repo",
        repo,
        "--body-file",
        str(body_path),
    ]
    if dry_run:
        return True, f"[dry-run] {' '.join(shlex.quote(x) for x in cmd)}"
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    ok = proc.returncode == 0
    msg = (proc.stdout + "\n" + proc.stderr).strip()
    return ok, msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recurring runner for HF Space training cadence.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to recurring run config JSON.")
    parser.add_argument("--cadence", choices=["daily", "weekly"], help="Execute only one cadence class.")
    parser.add_argument("--run-id", help="Execute only one configured run id.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Continue matrix even if a run marked continue_on_failure=false fails.",
    )
    parser.add_argument("--trigger", default="manual", help="Trigger label for reporting.")
    parser.add_argument("--issue-number", type=int, default=None, help="Override issue number for posting.")
    parser.add_argument("--no-issue-comment", action="store_true", help="Disable GitHub issue commenting.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and produce reports without running.")
    return parser.parse_args()


def build_execution_cmd(spec: RunSpec, has_pending: bool) -> list[str]:
    if spec.backend == "local":
        return build_pipeline_cmd(spec, has_pending)
    if spec.backend == "hf_jobs":
        return build_hf_jobs_cmd(spec, has_pending)
    raise ValueError(f"Unsupported backend={spec.backend!r} for run {spec.run_id}.")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Missing config file: {config_path}")
        return 1

    cfg_raw, specs = load_specs(config_path)
    selected = select_specs(specs, args.cadence, args.run_id)
    if not selected:
        print("No runs selected from config. Check --cadence/--run-id filters.")
        return 1

    files_count, pending_rows = pending_next_stats()
    has_pending = files_count > 0 and pending_rows > 0
    run_ts = utc_now()
    run_tag = ts_slug(run_ts)
    run_dir = REPORTS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recurring run {run_tag} | selected={len(selected)} | pending_next_files={files_count} rows={pending_rows}")

    results: list[dict[str, Any]] = []
    for spec in selected:
        log_path = run_dir / f"{spec.run_id}.log"
        cmd = build_execution_cmd(spec, has_pending)
        note = ""
        status = "success"
        failure_category: str | None = None

        if spec.skip_if_no_pending_data and not has_pending:
            status = "skipped"
            note = "Skipped (no pending data in data/next)."
            output = f"[skip] {note}\n"
            rc = 0
        else:
            rc, output = run_command(
                cmd,
                dry_run=args.dry_run,
                timeout_minutes=spec.timeout_minutes,
                extra_env=spec.env,
            )
            if rc != 0:
                status = "failed"
                note = f"Pipeline exited with code {rc}."
                failure_category = classify_failure(output, rc)
            else:
                note = (
                    "HF job submitted."
                    if spec.backend == "hf_jobs"
                    else "Pipeline completed."
                )

        write_text(log_path, output)
        results.append(
            {
                "run_id": spec.run_id,
                "cadence": spec.cadence,
                "family": spec.family,
                "profile": spec.profile,
                "backend": spec.backend,
                "status": status,
                "exit_code": rc,
                "note": note,
                "name": spec.name,
                "command": cmd,
                "timeout_minutes": spec.timeout_minutes,
                "env_keys": sorted(spec.env.keys()),
                "failure_category": failure_category,
                "log_relpath": str(log_path.relative_to(ROOT)),
            }
        )
        fail_tag = f", category={failure_category}" if failure_category else ""
        print(f"- {spec.run_id}: {status} ({note}{fail_tag})")
        if status == "failed" and (not spec.continue_on_failure) and (not args.allow_partial):
            print(
                f"Stopping matrix after {spec.run_id} failure "
                "(continue_on_failure=false; override with --allow-partial)."
            )
            break

    overall = "success"
    if any(item["status"] == "failed" for item in results):
        overall = "failed"
    elif all(item["status"] == "skipped" for item in results):
        overall = "skipped"

    summary = {
        "timestamp_utc": run_ts.isoformat(),
        "run_tag": run_tag,
        "trigger": args.trigger,
        "selected_count": len(selected),
        "pending_next_files": files_count,
        "pending_next_rows": pending_rows,
        "overall_status": overall,
        "failure_breakdown": {
            bucket: sum(1 for r in results if r.get("failure_category") == bucket)
            for bucket in ("infra", "data", "model_quality", "unknown")
            if any(r.get("failure_category") == bucket for r in results)
        },
        "results": results,
    }
    summary_path = run_dir / "summary.json"
    write_text(summary_path, json.dumps(summary, indent=2))
    print(f"Summary: {summary_path.relative_to(ROOT)}")

    issue_number = args.issue_number
    if issue_number is None:
        raw_issue = cfg_raw.get("issue_number")
        issue_number = int(raw_issue) if raw_issue is not None else None
    repo = str(cfg_raw.get("repo", "DealExMachina/sft-wagmi"))

    if args.no_issue_comment or issue_number is None:
        print("Issue comment step skipped (--no-issue-comment or no issue number).")
    else:
        template = load_template(DEFAULT_TEMPLATE_PATH)
        comment_md = render_issue_comment(
            template=template,
            run_timestamp=run_ts.isoformat(),
            trigger=args.trigger,
            cadence=args.cadence or "mixed",
            results=results,
            summary_relpath=str(summary_path.relative_to(ROOT)),
        )
        comment_path = run_dir / "issue_comment.md"
        write_text(comment_path, comment_md)
        if can_post_with_gh():
            ok, message = post_issue_comment(
                repo=repo,
                issue_number=issue_number,
                body_path=comment_path,
                dry_run=args.dry_run,
            )
            print("Issue comment:", "posted" if ok else "failed")
            if message:
                print(message)
        else:
            print(
                "Issue comment not posted (missing gh CLI or GH_TOKEN/GITHUB_TOKEN). "
                f"Comment draft saved: {comment_path.relative_to(ROOT)}"
            )

    return 1 if overall == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
