# data/next

Staging area for dataset entries destined for the next training version.

Drop `.jsonl` files here during development. Each file should follow the same
chat-format schema used in `data/train.jsonl`:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

When preparing the next release:

1. Review and QA entries in this directory.
2. Merge them into the main dataset via the dexm generator or manually.
3. Bump `VERSION`, update `CHANGELOG.md`, retrain.
4. Clear this directory for the next cycle.

Files here are **not** consumed by `train.py` automatically -- they must be
merged into `data/train.jsonl` (and optionally `data/eval.jsonl`) first.
