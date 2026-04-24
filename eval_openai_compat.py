"""OpenAI-compatible chat helper for remote evals.

Set these env vars to enable remote API mode in eval scripts:
  - EVAL_API_BASE_URL (example: https://host/v1)
  - EVAL_API_KEY
  - EVAL_API_MODEL (example: wagmi-sft-14b)
Optional:
  - EVAL_API_TIMEOUT_SEC (default: 120)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    timeout_sec: int = 120

    @classmethod
    def from_env(cls) -> "OpenAICompatClient | None":
        base = (os.environ.get("EVAL_API_BASE_URL") or "").strip().rstrip("/")
        if not base:
            return None
        model = (os.environ.get("EVAL_API_MODEL") or "").strip()
        if not model:
            raise ValueError("EVAL_API_MODEL is required when EVAL_API_BASE_URL is set")
        api_key = (os.environ.get("EVAL_API_KEY") or "ollama").strip()
        timeout = int(os.environ.get("EVAL_API_TIMEOUT_SEC", "120"))
        return cls(base_url=base, api_key=api_key, model=model, timeout_sec=timeout)

    def chat_completion(self, messages: list[dict[str, str]], gen_kwargs: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(gen_kwargs.get("max_new_tokens", 220)),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
        }
        if "top_p" in gen_kwargs:
            payload["top_p"] = float(gen_kwargs["top_p"])

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API unreachable: {exc}") from exc

        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (content or "").strip()
