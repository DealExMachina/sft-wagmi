# Recurring training (design overview)

High-level goals, cadence, and release gates for scheduled runs. **How to run** this today: [`scripts/hf/README.md`](../scripts/hf/README.md) (`recurring_runner.py`, `configs/recurring_runs.json`, Cursor SDK, GitHub Actions).

## Objective

Run `sft-wagmi` training on a regular cadence with:

- Cursor agents as orchestration/control plane
- Hugging Face (e.g. Hub Jobs / Space image) as execution plane
- Clear release gates (`eval`, `redteam`, `export-merged`) and auditable outcomes

## Constraints

- Heavy training and evaluation run remotely on Hugging Face, not on a local workstation, unless intentionally local.
- Job outputs are ephemeral unless persisted to the Hub or another durable store.
- Entrypoint remains [`scripts/pipeline.py`](../scripts/pipeline.py) so behavior matches manual runs.

## Cadence (target)

### Daily (light)

- `qwen` / `small`
- Typical steps: `--preflight`, optional `--sync-dataset`, conditional `--merge-next`, then `--train` → `--eval` → `--redteam` → `--export-merged` (or the equivalent via `recurring_runner.py` matrix)

### Weekly (heavy)

- `qwen` / `auth` (and optionally `lfm2` / `auth` when stable)
- Same gates as daily, with longer timeouts and stricter post-run checks.

## Release gates

A run is successful only if training exits `0`, `eval` completes, `redteam` meets the bar, and `export-merged` publishes the expected artifact. On failure: record the stage, log excerpt, and do not treat the model as a release candidate.

## Implementation in this repo

| Area | Location |
| --- | --- |
| Orchestrator | [`scripts/hf/recurring_runner.py`](../scripts/hf/recurring_runner.py) |
| Run matrix & backends | [`configs/recurring_runs.json`](../configs/recurring_runs.json) |
| Operational runbook | [`scripts/hf/README.md`](../scripts/hf/README.md) |
| Cursor SDK launcher | [`automation/cursor-sdk`](../automation/cursor-sdk/README.md) |
| Scheduled workflow (example) | `.github/workflows/recurring-training-cursor-sdk.yml` |

## Risks (unchanged)

- OOM/timeout on `auth` profiles → profile-specific timeouts, optional seq-length fallback, retries for transient infra.
- Missing secrets → preflight validation before remote submit.
- False-positive “success” → explicit per-stage checks and gate evaluation.
