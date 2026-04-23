#!/usr/bin/env python3
"""Generate AI Act-oriented model card drafts for HF repos.

This script prepares (but does not publish) model card README drafts for all
known Hub repos in `config.py` (adapter, merged, gguf). It is designed to make
the final Hugging Face update pass quick and consistent across repos.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import _REGISTRY  # project-internal source of truth

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "hf-model-card-prep"
VERSION_FILE = REPO_ROOT / "VERSION"
METADATA_FILE = REPO_ROOT / "data" / "metadata.json"
REDTEAM_ROOT = REPO_ROOT / "reports" / "redteam"


LICENSE_BY_FAMILY = {
    "qwen": "apache-2.0",
    "qwen3": "apache-2.0",
    "lfm2": "other",
}

LICENSE_NOTES_BY_FAMILY = {
    "qwen": "Apache 2.0 (Qwen base model)",
    "qwen3": "Apache 2.0 (Qwen base model)",
    "lfm2": "LFM Open License v1.0 (Liquid AI)",
}


@dataclass(frozen=True)
class RepoCard:
    repo_id: str
    family: str
    profile: str
    artifact_kind: str
    base_model: str
    license_spdx: str
    license_note: str
    redteam_md_relpath: str | None


def slugify_repo(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def read_version() -> str:
    if not VERSION_FILE.is_file():
        return "unknown"
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def read_dataset_summary() -> dict[str, Any]:
    if not METADATA_FILE.is_file():
        return {}
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


def latest_redteam_report(profile: str) -> str | None:
    if not REDTEAM_ROOT.exists():
        return None
    candidates = sorted(REDTEAM_ROOT.glob(f"v*/{profile}_redteam_*.md"))
    if not candidates:
        return None
    return str(candidates[-1].relative_to(REPO_ROOT))


def collect_repo_cards() -> list[RepoCard]:
    cards: list[RepoCard] = []
    seen: set[tuple[str, str, str]] = set()
    repo_fields = [
        ("adapter", "hub_adapter"),
        ("merged", "hub_merged"),
        ("gguf", "hub_gguf"),
    ]

    for family, profiles in _REGISTRY.items():
        for profile, cfg in profiles.items():
            base_model = str(cfg["model_id"])
            license_spdx = LICENSE_BY_FAMILY.get(family, "other")
            license_note = LICENSE_NOTES_BY_FAMILY.get(family, "See upstream base model license")
            redteam_md = latest_redteam_report(profile)

            for artifact_kind, field in repo_fields:
                repo_id = str(cfg[field])
                dedupe_key = (repo_id, family, profile)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                cards.append(
                    RepoCard(
                        repo_id=repo_id,
                        family=family,
                        profile=profile,
                        artifact_kind=artifact_kind,
                        base_model=base_model,
                        license_spdx=license_spdx,
                        license_note=license_note,
                        redteam_md_relpath=redteam_md,
                    )
                )
    return cards


def render_card(card: RepoCard, version: str, dataset_summary: dict[str, Any]) -> str:
    dataset_train = dataset_summary.get("counts", {}).get("train")
    dataset_eval = dataset_summary.get("counts", {}).get("eval")
    dataset_total = (
        (dataset_train if isinstance(dataset_train, int) else 0)
        + (dataset_eval if isinstance(dataset_eval, int) else 0)
    )
    if dataset_total <= 0:
        dataset_total_text = "TODO: fill exact count"
    else:
        dataset_total_text = str(dataset_total)

    report_hint = card.redteam_md_relpath or "TODO: attach latest red-team report path"
    title = f"Wagmi ({card.family}/{card.profile}) - {card.artifact_kind}"
    tags = [
        "wagmi",
        "deal-ex-machina",
        "sft",
        card.family,
        card.profile,
        card.artifact_kind,
    ]
    tags_block = "\n".join(f"- {tag}" for tag in tags)

    return f"""---
language:
- en
- fr
license: {card.license_spdx}
base_model: {card.base_model}
library_name: peft
pipeline_tag: text-generation
tags:
{tags_block}
---

# {title}

**Version:** {version}  
**Repo ID:** `{card.repo_id}`

## Model Summary

This model is part of the Wagmi assistant stack for Deal ex Machina. It is a `{card.artifact_kind}` artifact in the `{card.family}` family (`{card.profile}` profile).

## Intended Purpose

- Intended domain: questions about Deal ex Machina services, content, and related company context.
- Intended users: website visitors and authenticated users, depending on profile routing in production.
- Intended geographies/languages: French and English.

## Out-of-Scope Use

- General-purpose assistant usage unrelated to Deal ex Machina.
- Legal, medical, financial, hiring, credit, insurance, law-enforcement, or other high-impact decisions.
- Any use requiring guaranteed factual completeness.

## AI Act Transparency (Article 50) Notes

- This model powers a chatbot experience where users are informed they interact with AI.
- System scope is limited-risk as deployed (not categorized as high-risk use under current deployment assumptions).
- Human oversight remains with product operators; model output should not be used as sole basis for consequential decisions.

## Data and Training Provenance

- Base model: `{card.base_model}`
- Fine-tuning method: LoRA SFT (see project pipeline)
- Approximate SFT dataset size: {dataset_total_text} examples
- Data policy: no direct end-user chat logs are used for SFT

## Evaluation, Robustness, and Safety

- Red-team evidence: `{report_hint}`
- Include latest release metrics (quality + guardrail) before publishing.
- Add observed failure classes and mitigations in plain language.

## Known Limitations

- Domain-bounded assistant; degraded quality outside scope.
- Non-zero hallucination risk for edge prompts.
- Safety/robustness tests are finite and release-based.

## Risk Management and Incident Process

- Document escalation path for harmful/incorrect outputs.
- Link internal release gate evidence and retention policy.
- TODO: add public contact route for reporting model issues.

## License and Redistribution

- SPDX field: `{card.license_spdx}`
- License note: {card.license_note}
- Derivative distribution must comply with upstream model terms and Hugging Face terms.

## Maintainer Update Checklist

- [ ] Version/changelog links updated
- [ ] Dataset counts refreshed from `data/metadata.json`
- [ ] Latest red-team report attached or linked
- [ ] Limitations and out-of-scope section reviewed
- [ ] AI Act transparency language reviewed against current product behavior
- [ ] License section validated for this base model family
"""


def write_files(out_dir: Path, cards: list[RepoCard], version: str, dataset_summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for card in cards:
        repo_slug = slugify_repo(card.repo_id)
        repo_dir = cards_dir / repo_slug
        repo_dir.mkdir(parents=True, exist_ok=True)
        readme_path = repo_dir / "README.md"
        readme_path.write_text(render_card(card, version, dataset_summary), encoding="utf-8")
        manifest.append(
            {
                "repo_id": card.repo_id,
                "family": card.family,
                "profile": card.profile,
                "artifact_kind": card.artifact_kind,
                "base_model": card.base_model,
                "license_spdx": card.license_spdx,
                "draft_path": str(readme_path.relative_to(REPO_ROOT)),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": version,
        "count": len(manifest),
        "items": manifest,
    }
    manifest_path.write_text(f"{json.dumps(manifest_payload, indent=2)}\n", encoding="utf-8")

    runbook_lines = [
        "# HF Model Card Prep Output",
        "",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        "",
        "Use this folder to apply AI Act-oriented updates to Hugging Face model cards.",
        "",
        "## Generated Drafts",
    ]
    for item in manifest:
        runbook_lines.append(
            f"- `{item['repo_id']}` -> `{item['draft_path']}`"
        )
    runbook_lines.extend(
        [
            "",
            "## Publish Command (per repo)",
            "",
            "```bash",
            "hf auth login  # if needed",
            "hf upload <repo-id> <local-readme-path> README.md",
            "```",
            "",
            "Example:",
            "```bash",
            "hf upload jeanbaptdzd/wagmi-qwen2.5-1.5b-sft output/hf-model-card-prep/cards/jeanbaptdzd__wagmi-qwen2.5-1.5b-sft/README.md README.md",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(runbook_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AI Act-oriented Hugging Face model card drafts."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated card drafts and manifest.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="",
        help="Optional family filter (qwen, qwen3, lfm2).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="Optional profile filter (small, auth).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = read_version()
    dataset_summary = read_dataset_summary()
    cards = collect_repo_cards()

    if args.family:
        cards = [c for c in cards if c.family == args.family]
    if args.profile:
        cards = [c for c in cards if c.profile == args.profile]
    if not cards:
        raise SystemExit("No matching model repos found for given filters.")

    write_files(args.out_dir, cards, version, dataset_summary)
    rel = args.out_dir.relative_to(REPO_ROOT) if args.out_dir.is_relative_to(REPO_ROOT) else args.out_dir
    print(f"Generated {len(cards)} model card draft(s) in {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
