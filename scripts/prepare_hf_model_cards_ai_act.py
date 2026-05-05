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
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import _REGISTRY  # project-internal source of truth

DEFAULT_OUT_DIR = REPO_ROOT / "output" / "hf-model-card-prep"
VERSION_FILE = REPO_ROOT / "VERSION"
METADATA_FILE = REPO_ROOT / "data" / "metadata.json"
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"
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
    training_track: str


@dataclass(frozen=True)
class RedteamSummary:
    relpath: str
    profile: str
    family: str
    evaluated_at_utc: str
    release_gate: str
    pass_rate: str
    failures: str


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


def read_latest_release_notes(version: str) -> list[str]:
    if not CHANGELOG_FILE.is_file():
        return []
    content = CHANGELOG_FILE.read_text(encoding="utf-8")
    heading = f"## {version} --"
    start = content.find(heading)
    if start < 0:
        return []
    next_heading = content.find("\n## ", start + len(heading))
    section = content[start: next_heading if next_heading >= 0 else len(content)]
    bullets: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets[:4]


def infer_family_from_report_text(text: str) -> str:
    t = text.lower()
    if "qwen3" in t:
        return "qwen3"
    if "qwen" in t:
        return "qwen"
    if "lfm" in t or "liquidai" in t:
        return "lfm2"
    return "unknown"


def load_redteam_summaries(version: str) -> list[RedteamSummary]:
    if not REDTEAM_ROOT.exists():
        return []
    folder = REDTEAM_ROOT / f"v{version}"
    if not folder.exists():
        return []

    summaries: list[RedteamSummary] = []
    for path in sorted(folder.glob("*_redteam_*.md")):
        text = path.read_text(encoding="utf-8")
        profile_match = re.search(r"\*\*Profile\*\*:\s*`([^`]+)`", text)
        evaluated_match = re.search(r"\*\*Evaluated At \(UTC\)\*\*:\s*`([^`]+)`", text)
        gate_match = re.search(r"\*\*Release Gate\*\*:\s*\*\*([A-Z]+)\*\*", text)
        pass_rate_match = re.search(r"\*\*Pass Rate\*\*:\s*`([^`]+)`", text)
        failures_match = re.search(r"\*\*Failures\*\*:\s*`([^`]+)`", text)
        if not (profile_match and evaluated_match and gate_match and pass_rate_match and failures_match):
            continue
        summaries.append(
            RedteamSummary(
                relpath=str(path.relative_to(REPO_ROOT)),
                profile=profile_match.group(1),
                family=infer_family_from_report_text(text),
                evaluated_at_utc=evaluated_match.group(1),
                release_gate=gate_match.group(1),
                pass_rate=pass_rate_match.group(1),
                failures=failures_match.group(1),
            )
        )
    return summaries


def select_redteam_summary(
    summaries: list[RedteamSummary], family: str, profile: str
) -> RedteamSummary | None:
    candidates = [
        s for s in summaries
        if s.profile == profile and (s.family == family or s.family == "unknown")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda s: s.evaluated_at_utc)
    return candidates[-1]


def infer_training_track(repo_id: str) -> str:
    rid = repo_id.lower()
    if "-dpo-grpo" in rid:
        return "grpo"
    if "-dpo" in rid:
        return "dpo"
    return "sft"


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
                        training_track=infer_training_track(repo_id),
                    )
                )

            if family == "qwen" and profile == "auth":
                extra = [
                    ("adapter", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo"),
                    ("merged", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-merged"),
                    ("gguf", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-gguf"),
                    ("adapter", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo"),
                    ("merged", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo-merged"),
                    ("gguf", "jeanbaptdzd/wagmi-qwen2.5-14b-sft-dpo-grpo-gguf"),
                ]
                for artifact_kind, repo_id in extra:
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
                            training_track=infer_training_track(repo_id),
                        )
                    )
    return cards


def render_card(
    card: RepoCard,
    version: str,
    dataset_summary: dict[str, Any],
    release_notes: list[str],
    redteam_summary: RedteamSummary | None,
) -> str:
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
    dataset_snapshot_version = str(dataset_summary.get("version", "unknown"))

    title = f"Wagmi ({card.family}/{card.profile}/{card.training_track}) - {card.artifact_kind}"
    tags = [
        "wagmi",
        "deal-ex-machina",
        card.training_track,
        card.family,
        card.profile,
        card.artifact_kind,
    ]
    tags_block = "\n".join(f"- {tag}" for tag in tags)
    release_notes_block = "\n".join(f"- {note}" for note in release_notes) if release_notes else "- TODO: summarize latest release changes."

    if redteam_summary is None:
        redteam_block = (
            "- Latest release red-team report for this family/profile is not available.\n"
            "- Add a linked report before publishing a production-facing card."
        )
    else:
        redteam_block = (
            f"- Report: `{redteam_summary.relpath}`\n"
            f"- Evaluated at (UTC): `{redteam_summary.evaluated_at_utc}`\n"
            f"- Release gate: **{redteam_summary.release_gate}** (pass rate `{redteam_summary.pass_rate}`, failures `{redteam_summary.failures}`)"
        )

    if card.training_track == "dpo":
        training_method = "DPO safety alignment on top of the auth SFT adapter"
    elif card.training_track == "grpo":
        training_method = "GRPO refinement on top of auth DPO/SFT path"
    else:
        training_method = "LoRA SFT"

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

## Recent Training Updates

{release_notes_block}

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
- Training track: `{card.training_track}`
- Fine-tuning method: {training_method} (see project pipeline)
- Approximate SFT dataset size: {dataset_total_text} examples
- Dataset metadata snapshot version: `{dataset_snapshot_version}`
- Data policy: no direct end-user chat logs are used for SFT

## Evaluation, Robustness, and Safety

{redteam_block}

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

    release_notes = read_latest_release_notes(version)
    redteam_summaries = load_redteam_summaries(version)

    manifest: list[dict[str, Any]] = []
    for card in cards:
        repo_slug = slugify_repo(card.repo_id)
        repo_dir = cards_dir / repo_slug
        repo_dir.mkdir(parents=True, exist_ok=True)
        readme_path = repo_dir / "README.md"
        redteam_summary = select_redteam_summary(redteam_summaries, card.family, card.profile)
        readme_path.write_text(
            render_card(
                card=card,
                version=version,
                dataset_summary=dataset_summary,
                release_notes=release_notes,
                redteam_summary=redteam_summary,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "repo_id": card.repo_id,
                "family": card.family,
                "profile": card.profile,
                "artifact_kind": card.artifact_kind,
                "training_track": card.training_track,
                "base_model": card.base_model,
                "license_spdx": card.license_spdx,
                "redteam_report": redteam_summary.relpath if redteam_summary else None,
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
