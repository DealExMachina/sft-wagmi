# Recurring training (design notes)

**Implemented.** Operational steps, Cursor SDK launchers, and reporting live in:

- [`scripts/hf/README.md`](../scripts/hf/README.md)
- [`configs/recurring_runs.json`](../configs/recurring_runs.json)
- [`.github/workflows/recurring-training-cursor-sdk.yml`](../.github/workflows/recurring-training-cursor-sdk.yml)
- [`automation/cursor-sdk/README.md`](../automation/cursor-sdk/README.md)

This file keeps the original intent and gates; it does not duplicate command examples from the README above.

## Objective

Run `sft-wagmi` on a **daily** (lighter) and **weekly** (heavier) cadence using Hugging Face GPU jobs as the execution plane and Cursor agents / GitHub Actions as the control plane, with auditable outcomes.

## Constraints

- Heavy training and evaluation run remotely (HF Jobs / Space image), not on a laptop.
- Job stdout and local disks are ephemeral unless artifacts are pushed to the Hub or logged (e.g. issue comments).
- Pipeline behavior stays anchored on **`scripts/pipeline.py`** and flags defined there.

## Operating model

1. Scheduler or manual trigger starts the Cursor SDK recurring launcher (or `recurring_runner.py` in a Space shell).
2. `recurring_runner.py` expands the matrix from `configs/recurring_runs.json`, optionally skips empty `data/next/`, submits HF Jobs or runs locally per `backend`.
3. Success requires stage exits consistent with configured gates (train, eval, redteam, export as configured).
4. Outcomes roll up to the configured GitHub issue when `gh` + token are available.

## Configured cadence (current defaults)

See `configs/recurring_runs.json` for the source of truth. At a glance:

| Run id | Cadence | Matrix | Notes |
| --- | --- | --- | --- |
| `daily-qwen-small` | daily | `qwen` / `small` | `skip_if_no_pending_data`; `merge_next: auto`; HF Jobs backend |
| `weekly-qwen-auth` | weekly | `qwen` / `auth` | Longer timeout; stricter `continue_on_failure` |
| `weekly-lfm2-auth` | weekly | `lfm2` / `auth` | Present but **disabled** by default (`enabled: false`) |

Pipeline flags in config focus on `--train`, `--eval`, `--redteam`, `--export-merged` plus dataset sync / merge behavior per run — not necessarily the full local `--all` graph.

## Release gates

Treat a run as successful only when configured stages complete with exit code `0` and expected artifacts exist (merged weights on Hub where export is enabled). On failure: record stage, excerpt, and avoid promoting the build as a release candidate.

## Risks (concise)

- **OOM / timeout** on auth or large-seq runs — mitigate with timeouts, seq caps (`AUTH_MAX_SEQ_LEN` in config), and staged retries.
- **Missing secrets** — validate in preflight before submitting remote jobs.
- **False success** — rely on explicit per-stage checks in the runner, not a single blanket exit code from partial work.
- **Empty `data/next/`** — use `skip_if_no_pending_data` on light cadences to avoid pointless GPU use.
