# Recurring training (overview)

Operational runbook for commands, backends, scheduling, and failure handling lives in **`scripts/hf/README.md`**. Implementations touched by that doc:

| Piece | Role |
| --- | --- |
| [`scripts/hf/recurring_runner.py`](../scripts/hf/recurring_runner.py) | Matrix runner: `configs/recurring_runs.json`, pipelines, timeouts, summaries, optional GitHub issue comments |
| [`configs/recurring_runs.json`](../configs/recurring_runs.json) | Cadence (`daily`/`weekly`), family/profile, `pipeline_flags`, `backend` (`local`/`hf_jobs`), HF Job image/flavor |
| [`scripts/hf/templates/recurring_issue_comment.md`](../scripts/hf/templates/recurring_issue_comment.md) | Template for recurring run summaries |
| [`automation/cursor-sdk`](../automation/cursor-sdk) (`npm run run:recurring`) | Delegates execution to Cursor Cloud Agent (`CURSOR_API_KEY`) |
| [`.github/workflows/recurring-training-cursor-sdk.yml`](../.github/workflows/recurring-training-cursor-sdk.yml) | Cron + manual dispatch invoking the SDK launcher |

High-level cadence tracked in **`configs/recurring_runs.json`**: typically daily `qwen`/`small` and weekly `qwen`/`auth`, with **`weekly-lfm2-auth`** gated off (`enabled: false`) until validated.

Pipeline composition for scheduled runs differs from **`python3 scripts/pipeline.py --all`**: **`recurring_runner.py`** injects **`--preflight`**, optional **`--sync-dataset`**, and conditional **`--merge-next`**; **`pipeline_flags`** in the JSON then append stages. The default **`pipeline_flags`** in-repo run **`train`**, **`eval`**, **`redteam`**, and **`export-merged`** — they omit **`eval-rag`** (and **`eval-tools`**) to save time versus an interactive **`--all`**. Use **`pipeline.py --eval-rag`** in **`pipeline_flags`** if you need parity with **`--all`**.

## Constraints (still true)

- Training and heavy evaluation are meant to execute on HF (Space shell or **`hf jobs`**, per config), not on a laptop.
- Job outputs outside Hub / explicit artifacts are ephemeral.
- Keep **`scripts/pipeline.py`** the single behavioural entrypoint; recurring automation should orchestrate flags, not fork training scripts.

## Release gates

Treat a recurring run as successful when the configured **`pipeline_flags`** stages all exit `0`: typically training, eval, redteam, merged export plus any flags you added. Extend the checklist in **`scripts/hf/README.md`** when you widen **`pipeline_flags`**.

If a gate fails: mark the matrix row failed (see runner summaries under **`reports/recurring/`**), capture logs / issue-comment draft, do not ship the model as release-ready until rerun passes.

## Remaining backlog (hardening)

- Stronger transient-retry policies per run class (**`continue_on_failure`** + manual reruns partially cover this today).
- Optional post-run summary persisted as Hub dataset artifact (beyond local **`reports/recurring/`**).
