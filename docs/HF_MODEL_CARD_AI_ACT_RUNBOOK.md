# HF model cards -- AI Act prep runbook

This runbook prepares model card updates for Hugging Face repos so the public
documentation aligns with current transparency and governance expectations
(Article 50 transparency posture for limited-risk chatbot usage).

It does not auto-publish anything.

**Note:** [`scripts/prepare_hf_model_cards_ai_act.py`](../scripts/prepare_hf_model_cards_ai_act.py) also writes `output/hf-model-card-prep/README.md` with a manifest of generated drafts plus a minimal `hf upload` snippet—this file is the full pre-publish checklist. Optional CLI filters: `--family`, `--profile`, `--out-dir` (see `python3 scripts/prepare_hf_model_cards_ai_act.py --help`).

## 1) Generate draft cards

From `sft-wagmi/`:

```bash
python3 scripts/prepare_hf_model_cards_ai_act.py
```

Optional filtering:

```bash
python3 scripts/prepare_hf_model_cards_ai_act.py --family qwen --profile small
```

Outputs:

- `output/hf-model-card-prep/manifest.json`
- `output/hf-model-card-prep/README.md`
- `output/hf-model-card-prep/cards/<repo-slug>/README.md` (one draft per HF repo)

## 2) Human review before publish

For each generated draft:

- Confirm license field is correct for the base model family.
- Refresh dataset and evaluation values for the exact release.
- Validate that "intended use" and "out-of-scope" match production behavior.
- Verify links to red-team evidence and release docs are valid.
- Keep statements factual and avoid unverifiable legal claims.

## 3) Publish to HF

Authenticate once:

```bash
hf auth login
```

Upload per repository:

```bash
hf upload <repo-id> <local-readme-path> README.md
```

Example:

```bash
hf upload jeanbaptdzd/wagmi-qwen2.5-1.5b-sft output/hf-model-card-prep/cards/jeanbaptdzd__wagmi-qwen2.5-1.5b-sft/README.md README.md
```

## 4) Post-publish verification

- Open each model page and confirm markdown rendering + YAML metadata.
- Check model tags, pipeline tag, and license fields in the HF UI.
- Keep local draft snapshots in the release artifact trail.
