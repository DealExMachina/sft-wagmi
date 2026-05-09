# Recurring Training Plan (Cursor Agents + HF ml-intern)

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
4. Remote job runs `scripts/pipeline.py` with profile/family flags.
5. Agent collects outcome signals (success, metrics, redteam status, artifact refs).
6. Agent updates a tracking issue and flags failures for human review.

## Cadence

### Daily (light)

- `qwen/small`
- Steps:
  - `--preflight`
  - optional `--sync-dataset`
  - conditional `--merge-next`
  - `--train --eval --redteam --export-merged`

### Weekly (heavy)

- `qwen/auth` (+ optionally `lfm2/auth` once stable)
- Same steps as daily, with longer timeout and stricter post-run checks.

## Minimum Technical Deliverables

1. **Orchestrator script**
   - Add `scripts/hf/recurring_runner.py` (or similar) that:
     - builds run matrix
     - submits HF run commands
     - records job IDs/URLs
     - formats run summary payload

2. **Run profile config**
   - Add `configs/recurring_runs.json` (or env-driven equivalent) with:
     - cadence
     - family/profile
     - timeout
     - optional step flags

3. **Issue reporting template**
   - Add issue body/comment template for:
     - run date + trigger
     - job URL
     - pipeline command
     - pass/fail gates
     - links to reports/model artifacts

4. **Operational runbook**
   - Expand `scripts/hf/README.md` with:
     - how scheduler triggers agent
     - how to pause/resume
     - how to retry failed runs

## Release Gates

A run is considered successful only if:

- training exits with code `0`
- `eval` completes
- `redteam` completes with acceptable threshold
- `export-merged` completes and target Hub repo receives the expected artifact

If any gate fails:

- mark run as failed
- include failing stage and key log excerpt in issue update
- do not mark model as release candidate

## Phase Plan

### Phase 1: Single-path automation (target: 1-2 days)

- Automate daily `qwen/small`
- Manual trigger fallback kept available
- Tracking issue updated on each run

### Phase 2: Multi-profile expansion (target: +2-3 days)

- Add weekly `qwen/auth`
- Add configurable matrix support in orchestrator
- Introduce failure categorization (infra vs data vs model quality)

### Phase 3: Hardening (target: +2 days)

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

## Immediate Next Steps

1. Implement Phase 1 orchestrator skeleton and config file.
2. Create/initialize tracking issue for recurring run logs.
3. Run one dry execution path and validate reporting shape.
4. Enable schedule after first successful manual run.
