# Cursor SDK automation

This folder contains local Cursor SDK launchers used as control-plane automations for `sft-wagmi`.

## Prerequisites

- Node 20+
- `CURSOR_API_KEY` in environment

Install:

```bash
npm install
```

## Scripts

- `npm run run:housekeeper`
  - Runs a local Cursor agent focused on repository housekeeping:
    - update stale docs and README sections
    - remove clearly redundant documents
    - keep references consistent after edits/deletions
- `npm run run:recurring -- --cadence daily|weekly`
  - Triggers the existing recurring training runner through a Cursor agent.

Operational detail (run matrix, `recurring_runner.py`, HF Jobs): [`../../scripts/hf/README.md`](../../scripts/hf/README.md).

## HouseKeeper flags

- `--area <path>`
  - Prioritized scope for housekeeping (default: `.`)
- `--instruction "<text>"`
  - Extra instruction appended to the base housekeeping prompt

Example:

```bash
CURSOR_API_KEY=... npm run run:housekeeper -- --area docs --instruction "focus on outdated operation docs"
```

## Scheduling

Use cron (or any scheduler) from this directory:

```bash
0 2 * * * cd /path/to/sft-wagmi/automation/cursor-sdk && CURSOR_API_KEY=... npm run run:housekeeper -- --area .
```

For unattended runs, direct output to a log file and review diffs before commit.

GitHub Actions schedule is provided in `.github/workflows/housekeeper-cursor-sdk.yml`.
