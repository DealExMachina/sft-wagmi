"""Evaluate tool-calling behavior on email/calendar assistant scenarios."""

import datetime
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset

MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "small").strip().lower()
PROFILE_CONFIG = {
    "small": {
        "base_model_id": os.environ.get("SMALL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
        "adapter_dir": os.environ.get("SMALL_OUTPUT_DIR", "output/wagmi-qwen2.5-1.5b-sft"),
        "hub_adapter": os.environ.get("SMALL_HUB_MODEL_ID", "jeanbaptdzd/wagmi-qwen2.5-1.5b-sft"),
    },
    "auth": {
        "base_model_id": os.environ.get("AUTH_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct"),
        "adapter_dir": os.environ.get("AUTH_OUTPUT_DIR", "output/wagmi-qwen2.5-14b-sft"),
        "hub_adapter": os.environ.get("AUTH_HUB_MODEL_ID", "jeanbaptdzd/wagmi-qwen2.5-14b-sft"),
    },
}
if MODEL_PROFILE not in PROFILE_CONFIG:
    raise ValueError(f"Unsupported MODEL_PROFILE={MODEL_PROFILE}. Expected one of: {', '.join(PROFILE_CONFIG.keys())}")

BASE_MODEL_ID = PROFILE_CONFIG[MODEL_PROFILE]["base_model_id"]
ADAPTER_DIR = PROFILE_CONFIG[MODEL_PROFILE]["adapter_dir"]
HUB_ADAPTER = PROFILE_CONFIG[MODEL_PROFILE]["hub_adapter"]
OUTPUT_FILE = Path(f"output/{MODEL_PROFILE}_tool_eval_results.json")
TOOLING_FILE = os.environ.get("AUTH_TOOLING_FILE", "data/tooling_email_calendar.jsonl")
TOOLING_LIMIT = int(os.environ.get("TOOLING_EVAL_LIMIT", "20"))
DTYPE = torch.bfloat16

PROFILE_GEN_KWARGS = {
    "small": dict(
        max_new_tokens=int(os.environ.get("SMALL_MAX_NEW_TOKENS", "220")),
        temperature=float(os.environ.get("SMALL_TEMPERATURE", "0.0")),
        do_sample=False,
        repetition_penalty=float(os.environ.get("SMALL_REPETITION_PENALTY", "1.05")),
    ),
    "auth": dict(
        max_new_tokens=int(os.environ.get("AUTH_MAX_NEW_TOKENS", "220")),
        temperature=float(os.environ.get("AUTH_TEMPERATURE", "0.0")),
        do_sample=False,
        repetition_penalty=float(os.environ.get("AUTH_REPETITION_PENALTY", "1.05")),
    ),
}
GEN_KWARGS = PROFILE_GEN_KWARGS[MODEL_PROFILE]


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        maybe = json.loads(text)
        if isinstance(maybe, dict):
            return maybe
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            maybe = json.loads(text[start : end + 1])
            if isinstance(maybe, dict):
                return maybe
        except json.JSONDecodeError:
            return None
    return None


def expected_is_tool_call(expected_assistant: str) -> bool:
    parsed = extract_json(expected_assistant)
    return isinstance(parsed, dict) and "tool_name" in parsed


def run():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    adapter_path = ADAPTER_DIR if Path(ADAPTER_DIR).exists() else HUB_ADAPTER
    print(f"\nLoading base model: {BASE_MODEL_ID}")
    print(f"Loading adapter: {adapter_path} (profile={MODEL_PROFILE})")

    if MODEL_PROFILE == "auth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            dtype=DTYPE,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=DTYPE, device_map="auto")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    ds = load_dataset("json", data_files={"eval": TOOLING_FILE})["eval"]
    if TOOLING_LIMIT > 0:
        ds = ds.select(range(min(TOOLING_LIMIT, len(ds))))
    print(f"Tool eval cases: {len(ds)} from {TOOLING_FILE}\n")

    results = []
    counts = {
        "total": len(ds),
        "valid_json": 0,
        "tool_name_match": 0,
        "required_args_match": 0,
        "non_tool_ok": 0,
    }

    for i, row in enumerate(ds, 1):
        messages = row["messages"]
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        prompt_messages = [m for m in messages if m["role"] in ("system", "user")]

        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

        expected_tool = extract_json(expected)
        pred_tool = extract_json(pred)
        expects_tool = expected_is_tool_call(expected)

        row_metrics = {
            "valid_json": False,
            "tool_name_match": False,
            "required_args_match": False,
            "non_tool_ok": False,
        }

        if expects_tool:
            row_metrics["valid_json"] = isinstance(pred_tool, dict)
            if row_metrics["valid_json"]:
                counts["valid_json"] += 1
                row_metrics["tool_name_match"] = pred_tool.get("tool_name") == expected_tool.get("tool_name")
                if row_metrics["tool_name_match"]:
                    counts["tool_name_match"] += 1
                expected_args = expected_tool.get("arguments", {})
                pred_args = pred_tool.get("arguments", {})
                if isinstance(expected_args, dict) and isinstance(pred_args, dict):
                    required_keys = [k for k in expected_args.keys() if expected_args.get(k) not in (None, "", [])]
                    row_metrics["required_args_match"] = all(k in pred_args and pred_args[k] not in (None, "", []) for k in required_keys)
                    if row_metrics["required_args_match"]:
                        counts["required_args_match"] += 1
        else:
            row_metrics["non_tool_ok"] = pred_tool is None
            if row_metrics["non_tool_ok"]:
                counts["non_tool_ok"] += 1

        results.append(
            {
                "id": row.get("id", f"case-{i}"),
                "locale": row.get("locale"),
                "expects_tool_call": expects_tool,
                "user": user_text,
                "expected_assistant": expected,
                "prediction": pred,
                "metrics": row_metrics,
            }
        )
        print(f"[{i:02d}/{len(ds)}] {row.get('id', f'case-{i}')} done")

    expected_tool_total = sum(1 for r in results if r["expects_tool_call"])
    expected_non_tool_total = len(results) - expected_tool_total
    summary = {
        "expected_tool_total": expected_tool_total,
        "expected_non_tool_total": expected_non_tool_total,
        "valid_json_rate": counts["valid_json"] / expected_tool_total if expected_tool_total else 0.0,
        "tool_name_match_rate": counts["tool_name_match"] / expected_tool_total if expected_tool_total else 0.0,
        "required_args_match_rate": counts["required_args_match"] / expected_tool_total if expected_tool_total else 0.0,
        "non_tool_ok_rate": counts["non_tool_ok"] / expected_non_tool_total if expected_non_tool_total else 0.0,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": BASE_MODEL_ID,
        "adapter": adapter_path,
        "stage": "tool-call-eval",
        "evaluatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "genKwargs": GEN_KWARGS,
        "counts": counts,
        "summary": summary,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\nTool-calling eval summary")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to {OUTPUT_FILE}")
    return payload


if __name__ == "__main__":
    run()
