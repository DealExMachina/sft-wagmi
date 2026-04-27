#!/usr/bin/env python3
"""Verify auth adapter resolution for Qwen 14B and LFM2 (no torch, no model load)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def probe(family: str) -> None:
    code = f"""
import os, sys
sys.path.insert(0, {str(ROOT)!r})
os.environ["LLM_FAMILY"] = {family!r}
os.environ["MODEL_PROFILE"] = "auth"
from config import resolve_profile_config
from pathlib import Path
c = resolve_profile_config()
local = Path(c.adapter_dir)
print("\\n[{family}/auth] base_model_id=" + c.model_id)
print("  adapter_dir=" + c.adapter_dir + " -> " + ("EXISTS" if local.is_dir() else "MISSING (Unsloth loads hub id)"))
print("  hub_adapter=" + c.hub_adapter)
"""
    subprocess.run([sys.executable, "-c", code], check=False)


def main() -> int:
    probe("qwen")
    probe("lfm2")
    print("\nPreflight done. Run on HF L40: ./scripts/hf/run_redteam_auth_l40.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
