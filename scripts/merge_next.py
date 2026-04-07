"""Merge data/next/*.jsonl into the main training dataset.

Reads all .jsonl files from data/next/, validates schema, appends entries
to data/train.jsonl (with a configurable eval split), updates metadata.json,
and optionally bumps VERSION.

Usage:
    python3 scripts/merge_next.py                # merge + bump patch
    python3 scripts/merge_next.py --bump minor   # merge + bump minor
    python3 scripts/merge_next.py --dry-run      # preview without writing
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
NEXT_DIR = DATA_DIR / "next"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
META_FILE = DATA_DIR / "metadata.json"
VERSION_FILE = ROOT / "VERSION"

EVAL_FRACTION = 0.15
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text().strip().split("\n"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ERROR: {path.name}:{i}: {e}")
            sys.exit(1)
    return rows


def validate(entries: list[dict], source_file: str) -> None:
    ids = set()
    for i, d in enumerate(entries, 1):
        for key in ("id", "source", "locale", "tags", "messages"):
            if key not in d:
                print(f"  ERROR: {source_file} entry {i}: missing '{key}'")
                sys.exit(1)
        if d["id"] in ids:
            print(f"  ERROR: {source_file} entry {i}: duplicate id '{d['id']}'")
            sys.exit(1)
        ids.add(d["id"])
        msgs = d.get("messages", [])
        if len(msgs) < 3 or msgs[0]["role"] != "system":
            print(f"  ERROR: {source_file} entry {i}: invalid message structure")
            sys.exit(1)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bump_version(bump_type: str) -> str:
    current = VERSION_FILE.read_text().strip()
    parts = current.split(".")
    if len(parts) != 3:
        print(f"  ERROR: invalid VERSION format: {current}")
        sys.exit(1)
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge data/next/ into training dataset")
    parser.add_argument("--bump", choices=["patch", "minor", "major"], default="minor",
                        help="Version bump type (default: minor)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--no-bump", action="store_true", help="Skip version bump")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  merge_next.py — merge data/next/ into training set")
    print(f"{'=' * 60}")

    jsonl_files = sorted(NEXT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("\n  No .jsonl files in data/next/. Nothing to merge.")
        return 0

    all_new: list[dict] = []
    for f in jsonl_files:
        entries = read_jsonl(f)
        validate(entries, f.name)
        print(f"  {f.name}: {len(entries)} entries")
        all_new.extend(entries)

    existing_train = read_jsonl(TRAIN_FILE)
    existing_eval = read_jsonl(EVAL_FILE)
    existing_ids = {d["id"] for d in existing_train + existing_eval}

    collisions = [d["id"] for d in all_new if d["id"] in existing_ids]
    if collisions:
        print(f"\n  ERROR: {len(collisions)} ID collisions with existing data:")
        for cid in collisions[:5]:
            print(f"    - {cid}")
        return 1

    random.seed(SEED)
    random.shuffle(all_new)

    n_eval = max(1, int(len(all_new) * EVAL_FRACTION))
    new_eval = all_new[:n_eval]
    new_train = all_new[n_eval:]

    print(f"\n  New entries: {len(all_new)} total")
    print(f"    -> train: +{len(new_train)}")
    print(f"    -> eval:  +{n_eval}")

    merged_train = existing_train + new_train
    merged_eval = existing_eval + new_eval

    locales = Counter(d["locale"] for d in merged_train + merged_eval)
    tags = Counter()
    source_types = Counter()
    for d in merged_train + merged_eval:
        for t in d.get("tags", []):
            tags[t] += 1
        source_types[d.get("source", "unknown")] += 1

    new_version = None
    if not args.no_bump:
        new_version = bump_version(args.bump)

    current_version = VERSION_FILE.read_text().strip()
    print(f"\n  Before: {len(existing_train)} train + {len(existing_eval)} eval = {len(existing_train) + len(existing_eval)} total (v{current_version})")
    print(f"  After:  {len(merged_train)} train + {len(merged_eval)} eval = {len(merged_train) + len(merged_eval)} total", end="")
    if new_version:
        print(f" (v{new_version})")
    else:
        print()

    if args.dry_run:
        print("\n  [DRY RUN] No files written.")
        return 0

    write_jsonl(TRAIN_FILE, merged_train)
    write_jsonl(EVAL_FILE, merged_eval)
    print(f"\n  Written: {TRAIN_FILE.name} ({len(merged_train)} rows)")
    print(f"  Written: {EVAL_FILE.name} ({len(merged_eval)} rows)")

    meta = {
        "version": new_version or current_version,
        "generatedAt": json.loads(META_FILE.read_text()).get("generatedAt", ""),
        "mergedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "output": {"train": "datasets/wagmi-sft/train.jsonl", "eval": "datasets/wagmi-sft/eval.jsonl"},
        "counts": {
            "total": len(merged_train) + len(merged_eval),
            "train": len(merged_train),
            "eval": len(merged_eval),
        },
        "distribution": {
            "byLocale": dict(locales.most_common()),
            "byTag": dict(tags.most_common()),
            "bySourceType": dict(source_types.most_common()),
        },
    }
    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"  Written: {META_FILE.name}")

    if new_version:
        VERSION_FILE.write_text(new_version + "\n")
        print(f"  Version bumped: {current_version} -> {new_version}")

    for f in jsonl_files:
        f.unlink()
    print(f"\n  Cleared {len(jsonl_files)} file(s) from data/next/")

    print(f"\n{'=' * 60}")
    print(f"  DONE — dataset ready for training")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
