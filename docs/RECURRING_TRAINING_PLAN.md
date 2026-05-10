# Recurring Training Plan (Cursor Agents + HF ml-intern)

## Implementation status

The items under [Minimum technical deliverables](#minimum-technical-deliverables) and the Phase 1–2 automation described below are **implemented** in-repo:

- Runner: [`scripts/hf/recurring_runner.py`](../scripts/hf/recurring_runner.py)
- Matrix config: [`configs/recurring_runs.json`](../configs/recurring_runs.json)
- Operational runbook: [`scripts/hf/README.md`](../scripts/hf/README.md) (recurring runner, HF Jobs, Cursor SDK)
- Cursor SDK package: [`automation/cursor-sdk/README.md`](../automation/cursor-sdk/README.md)
- Schedule: [`.github/workflows/recurring-training-cursor-sdk.yml`](../.github/workflows/recurring-training-cursor-sdk.yml)

Use the runbook for day-to-day operations; keep this document as the design reference (objectives, cadence, gates, risks).

## Objective

Run `sft-wagmi` training on a regular cadence with:

- Cursor agents as orchestration/control plane
- Hugging Face infrastructure (`ml-intern` target) as execution plane
- Clear release gates (`eval`, `redteam`) and auditable outcomes

## Constraints

- Training and heavy evaluation execute remotely on Hugging Face, not on local workstation.
- Job outputs are ephemeral unless explicitly persisted to the Hub or another durable store.
- Existing repo entrypoint remains `scripts/pipeline.py` to avoid divergence in training behavior.

## Target Operating Model

1. Cursor scheduled agent starts a run.
2. Agent determines run type (daily light vs weekly heavy).
3. Agent submits remote HF execution with required secrets.
4. Remote job runs the configured training command (in practice `scripts/hf/recurring_runner.py` invoking `scripts/pipeline.py` with matrix flags).
5. Agent collects outcome signals (success, metrics, redteam status, artifact refs).
6. Agent updates a tracking issue and flags failures for human review.

## Cadence

### Daily (light)

- `qwen/small`
- Steps (equivalent to `scripts/pipeline.py --all` for the chosen profile/family, after optional preflight/sync/merge):
  - `--preflight`
  - optional `--sync-dataset`
  - conditional `--merge-next`
  - `train`, `eval`, `eval-rag`, `redteam`, `export-merged`

### Weekly (heavy)

- `qwen/auth` (+ optionally `lfm2/auth` once stable)
- Same steps as daily, with longer timeout and stricter post-run checks.

## Minimum technical deliverables

These are **in place** (paths above). When extending automation, update `configs/recurring_runs.json` and/or `recurring_runner.py` rather than forking the training entrypoint (`scripts/pipeline.py`).

| Deliverable | Location |
| --- | --- |
| Orchestrator / matrix | [`scripts/hf/recurring_runner.py`](../scripts/hf/recurring_runner.py) |
| Run profile config | [`configs/recurring_runs.json`](../configs/recurring_runs.json) |
| Issue reporting template | [`scripts/hf/templates/recurring_issue_comment.md`](../scripts/hf/templates/recurring_issue_comment.md) |
| Operational runbook | [`scripts/hf/README.md`](../scripts/hf/README.md) |

## Release gates

A run is considered successful only if:

- training exits with code `0`
- `eval` and `eval-rag` complete
- `redteam` completes with acceptable threshold
- `export-merged` completes and target Hub repo receives the expected artifact

If any gate fails:

- mark run as failed
- include failing stage and key log excerpt in issue update
- do not mark model as release candidate

## Phase plan

Phase 1–2 scope (daily `qwen/small`, weekly `qwen/auth`, configurable matrix, failure categories) is implemented via `recurring_runner.py` and `configs/recurring_runs.json`. Remaining items are hardening (Phase 3) and process, not greenfield build-out.

### Phase 3: Hardening (ongoing)

- Add retry policy for transient infra failures
- Add timeout policies per run class
- Add post-run summary artifact in-repo or Hub dataset

## Risks and Mitigations

- **OOM or timeout on auth profile**
  - Mitigation: profile-specific timeouts, lower seq length fallback, staged retries.
- **Missing secret/token**
  - Mitigation: preflight secret validation before remote submit.
- **False-positive success (partial pipeline)**
  - Mitigation: explicit stage-level success checks and gate evaluation.
- **No pending data but wasted GPU run**
  - Mitigation: conditional merge/run logic, skip policy for empty `data/next`.

## What's next

Track operational improvements (pause/resume, richer summaries, Hub artifacts) in issues and in [`scripts/hf/README.md`](../scripts/hf/README.md). For a new automation surface, extend `recurring_runner.py` / `configs/recurring_runs.json` so training behavior stays on `scripts/pipeline.py`.
