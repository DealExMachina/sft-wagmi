# data/next

Staging area for dataset entries destined for the next training version.

Drop `.jsonl` files here during development. Each entry **must** include
these fields:

```json
{
  "id": "unique-entry-id",
  "source": "handcrafted",
  "locale": "fr",
  "tags": ["guardrail", "category-name"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Merging into the training set

Run `scripts/merge_next.py` (or use the pipeline / Gradio UI):

```bash
python3 scripts/merge_next.py --bump minor
# or via pipeline:
python3 scripts/pipeline.py --merge-next --bump minor
```

This will:

1. Validate schema (id, source, locale, tags, messages) for every entry.
2. Check for duplicate IDs against existing train/eval data.
3. Shuffle and split new entries (85% train / 15% eval).
4. Append to `data/train.jsonl` and `data/eval.jsonl`.
5. Update `data/metadata.json` with new counts and distribution.
6. Bump `VERSION` (minor by default).
7. Clear this directory.
