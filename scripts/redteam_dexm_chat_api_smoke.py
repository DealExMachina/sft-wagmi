#!/usr/bin/env python3
"""Smoke red-team prompts against dexm-one-page POST /api/chat (garde-fous serveur).

Reuses the same JSON cases as eval_redteam and `guardrail_checks.check_case` (no torch).

For a quick deterministic check, use prompts that trigger action_refused (no LLM needed).

Usage:
  export CHAT_API_BASE_URL=https://staging.example.com   # no trailing slash on path
  python3 scripts/redteam_dexm_chat_api_smoke.py --max-cases 8

Optional:
  --cookie 'sb-...'   Supabase session cookie for authenticated paths (if needed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = ROOT / "data" / "redteam_guardrail_cases.json"

# Ensure imports resolve
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def post_chat(base: str, session_id: str, locale: str, user_text: str, cookie: str | None) -> tuple[str, str]:
    """Return (kind, body_or_snippet) where kind is json|stream|error."""
    url = f"{base.rstrip('/')}/api/chat"
    payload = {
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"type": "text", "text": user_text}],
            }
        ],
        "locale": locale,
        "sessionId": session_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "X-Session-ID": session_id,
    }
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        return "error", f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}"
    except urllib.error.URLError as exc:
        return "error", str(exc)

    if "application/json" in ctype:
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return "error", "invalid JSON"
        return "json", json.dumps(data, ensure_ascii=False)

    text = raw.decode("utf-8", errors="replace")
    return "stream", text[:24000]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("CHAT_API_BASE_URL", "").strip(), help="Origin + site, e.g. https://foo.pages.dev")
    parser.add_argument("--cases", type=Path, default=CASES_FILE)
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    parser.add_argument("--cookie", default=os.environ.get("DEXM_CHAT_COOKIE"))
    args = parser.parse_args()
    if not args.base_url:
        print("ERROR: set CHAT_API_BASE_URL or pass --base-url", file=sys.stderr)
        return 2

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    os.chdir(ROOT)
    from guardrail_checks import check_case

    session_id = str(uuid.uuid4())
    passed = 0
    for i, case in enumerate(cases, 1):
        prompt = case["prompt"]
        locale = case.get("locale") or "en"
        kind, payload = post_chat(args.base_url, session_id, locale, prompt, args.cookie)
        if kind == "error":
            print(f"[{i:02d}/{len(cases)}] {case['id']} ERROR {payload[:500]}")
            continue
        if kind == "json":
            data = json.loads(payload)
            if data.get("type") == "action_refused":
                response = data.get("content") or ""
            elif data.get("type") == "blocked":
                response = data.get("content") or ""
            elif data.get("type") == "email_required":
                response = data.get("content") or ""
            else:
                response = payload
        else:
            response = payload

        ev = check_case(case, response if isinstance(response, str) else str(response))
        ok = ev["passed"]
        passed += int(ok)
        print(f"[{i:02d}/{len(cases)}] {case['id']} {'PASS' if ok else 'FAIL'} ({kind})")
        if not ok:
            print("   failures:", ", ".join(ev["failures"]))

    print(f"\nSummary: {passed}/{len(cases)} passed (dexm /api/chat path)")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
