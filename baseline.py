"""Wagmi baseline evaluation — runs Qwen2.5-1.5B-Instruct (no fine-tuning)."""

import datetime
import json
import os
from pathlib import Path

import torch

from prompt_encode import build_generate_inputs

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PROFILE = os.environ.get("MODEL_PROFILE", "small").strip().lower()
PROFILE_MODEL_IDS = {
    "small": os.environ.get("SMALL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
    "auth": os.environ.get("AUTH_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct"),
}
if MODEL_PROFILE not in PROFILE_MODEL_IDS:
    raise ValueError(f"Unsupported MODEL_PROFILE={MODEL_PROFILE}. Expected one of: {', '.join(PROFILE_MODEL_IDS.keys())}")

MODEL_ID = PROFILE_MODEL_IDS[MODEL_PROFILE]
MAX_SEQ_LEN = 2048
DTYPE = torch.bfloat16
OUTPUT_FILE = Path(f"output/{MODEL_PROFILE}_baseline_results.json")

GEN_KWARGS = dict(
    max_new_tokens=300,
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
)

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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {MODEL_ID} (profile={MODEL_PROFILE}) ...")

    if MODEL_PROFILE == "auth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="auto")
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)\n")
    device = next(model.parameters()).device

    results = []
    for i, p in enumerate(PROMPTS, 1):
        print(f"[{i:02d}/{len(PROMPTS)}] {p['id']} ...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        inputs = build_generate_inputs(tokenizer, messages, device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        results.append({**p, "response": response})
        print("done")

    for r in results:
        print(f"\n{'─' * 60}")
        print(f"[{r['id']}] ({r['locale'].upper()} / {r['category']})")
        print(f"Q: {r['user']}")
        print(f"A: {r['response']}")
    print(f"{'─' * 60}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_ID,
        "stage": "baseline",
        "evaluatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "genKwargs": GEN_KWARGS,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run()
