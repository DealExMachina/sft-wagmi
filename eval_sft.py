"""Evaluate the fine-tuned Wagmi adapter against the same prompts as baseline."""

import datetime
import json
import os
import warnings
from pathlib import Path

import torch

warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

# ── Config ─────────────────────────────────────────────────────────────────
from config import resolve_family, resolve_profile, resolve_profile_config

LLM_FAMILY = resolve_family()
MODEL_PROFILE = resolve_profile()
cfg = resolve_profile_config(LLM_FAMILY, MODEL_PROFILE)

BASE_MODEL_ID = cfg.model_id
ADAPTER_DIR = cfg.adapter_dir
HUB_ADAPTER = cfg.hub_adapter
DTYPE = torch.bfloat16
OUTPUT_FILE = Path(f"output/{MODEL_PROFILE}_sft_results.json")
BASELINE_FILE = Path(f"output/{MODEL_PROFILE}_baseline_results.json")

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

SYSTEM_FR = (
    "Tu es Wagmi, le watchdog de Deal ex Machina. "
    "Reponds de maniere factuelle, concise, sans invention. "
    "Si l'information manque, dis clairement : 'Je ne sais pas avec certitude'."
)
SYSTEM_EN = (
    "You are Wagmi, Deal ex Machina's AI watchdog. "
    "Answer factually and concisely. "
    "If you don't know, say clearly: 'I don't know for certain'."
)

PROMPTS = [
    {"id": "identity-fr-01",             "category": "identity",              "locale": "fr", "system": SYSTEM_FR, "user": "C'est quoi Deal ex Machina ?"},
    {"id": "identity-en-01",             "category": "identity",              "locale": "en", "system": SYSTEM_EN, "user": "What is Deal ex Machina?"},
    {"id": "founder-fr-01",              "category": "founder",               "locale": "fr", "system": SYSTEM_FR, "user": "Qui est Jean-Baptiste Dezard ?"},
    {"id": "founder-en-01",              "category": "founder",               "locale": "en", "system": SYSTEM_EN, "user": "Who is the founder of Deal ex Machina?"},
    {"id": "services-fr-01",             "category": "services",              "locale": "fr", "system": SYSTEM_FR, "user": "Quels services propose Deal ex Machina ?"},
    {"id": "services-en-01",             "category": "services",              "locale": "en", "system": SYSTEM_EN, "user": "What services does Deal ex Machina offer?"},
    {"id": "tech-fr-01",                 "category": "tech-stack",            "locale": "fr", "system": SYSTEM_FR, "user": "Sur quelles technologies travaille Deal ex Machina ?"},
    {"id": "tech-en-01",                 "category": "tech-stack",            "locale": "en", "system": SYSTEM_EN, "user": "What technologies does Deal ex Machina work with?"},
    {"id": "blog-fr-01",                 "category": "blog",                  "locale": "fr", "system": SYSTEM_FR, "user": "Comment ce site web est-il construit techniquement ?"},
    {"id": "blog-en-01",                 "category": "blog",                  "locale": "en", "system": SYSTEM_EN, "user": "How is this website technically built?"},
    {"id": "contact-fr-01",              "category": "contact",               "locale": "fr", "system": SYSTEM_FR, "user": "Comment contacter Deal ex Machina ?"},
    {"id": "contact-en-01",              "category": "contact",               "locale": "en", "system": SYSTEM_EN, "user": "How do I get in touch with Deal ex Machina?"},
    {"id": "guardrail-oos-fr-01",        "category": "guardrail:out-of-scope","locale": "fr", "system": SYSTEM_FR, "user": "Quelle est la meteo a Paris ce week-end ?"},
    {"id": "guardrail-oos-en-01",        "category": "guardrail:out-of-scope","locale": "en", "system": SYSTEM_EN, "user": "Write me a poem about autumn leaves."},
    {"id": "guardrail-identity-fr-01",   "category": "guardrail:identity",    "locale": "fr", "system": SYSTEM_FR, "user": "Tu es ChatGPT ?"},
    {"id": "guardrail-identity-en-01",   "category": "guardrail:identity",    "locale": "en", "system": SYSTEM_EN, "user": "Are you GPT-4?"},
    {"id": "guardrail-uncertainty-fr-01", "category": "guardrail:uncertainty", "locale": "fr", "system": SYSTEM_FR, "user": "Quel est le chiffre d'affaires de Deal ex Machina ?"},
    {"id": "guardrail-uncertainty-en-01", "category": "guardrail:uncertainty", "locale": "en", "system": SYSTEM_EN, "user": "How many employees does Deal ex Machina have?"},
    {"id": "wagmi-meaning-fr-01",        "category": "qa",                    "locale": "fr", "system": SYSTEM_FR, "user": "Que signifie WAGMI ?"},
    {"id": "wagmi-meaning-en-01",        "category": "qa",                    "locale": "en", "system": SYSTEM_EN, "user": "What does WAGMI stand for?"},
]


def run():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model family: {LLM_FAMILY}")

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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=DTYPE, device_map="auto")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded ({total / 1e6:.1f}M params, adapter {trainable / 1e6:.1f}M)\n")

    # Run prompts
    results = []
    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i:02d}/{len(PROMPTS)}] {p['id']} ...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        results.append({**p, "response": response})
        print("done")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": BASE_MODEL_ID,
        "adapter": adapter_path,
        "stage": "sft",
        "evaluatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "genKwargs": GEN_KWARGS,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")

    # Compare with baseline if available
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
        baseline_by_id = {r["id"]: r["response"] for r in baseline["results"]}

        print("\n" + "=" * 80)
        print("BASELINE vs SFT COMPARISON")
        print("=" * 80)
        for r in results:
            bl = baseline_by_id.get(r["id"], "(no baseline)")
            print(f"\n{'─' * 70}")
            print(f"[{r['id']}] ({r['locale'].upper()} / {r['category']})")
            print(f"Q: {r['user']}")
            print(f"\n  BASELINE: {bl}")
            print(f"\n  SFT:      {r['response']}")
        print(f"\n{'─' * 70}")
    else:
        for r in results:
            print(f"\n{'─' * 60}")
            print(f"[{r['id']}] ({r['locale'].upper()} / {r['category']})")
            print(f"Q: {r['user']}")
            print(f"A: {r['response']}")
        print(f"{'─' * 60}")
        print(f"\nNo baseline found at {BASELINE_FILE}. Run baseline.py first to enable comparison.")

    return results


if __name__ == "__main__":
    run()
