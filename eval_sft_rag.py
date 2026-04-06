"""Evaluate fine-tuned Wagmi with simulated RAG context injection."""

import datetime
import json
import os
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

# ── Config ─────────────────────────────────────────────────────────────────
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
DTYPE = torch.bfloat16
OUTPUT_FILE = Path(f"output/{MODEL_PROFILE}_sft_rag_results.json")
SFT_FILE = Path(f"output/{MODEL_PROFILE}_sft_results.json")

GEN_KWARGS = dict(
    max_new_tokens=300,
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
)

# ── RAG knowledge chunks (extracted from wagmi-skills.md) ──────────────────
# Each prompt gets the most relevant chunk injected into the system prompt,
# simulating how the production BM25 RAG works.

RAG_CHUNKS = {
    "identity": (
        "Deal ex Machina is a technology consulting firm specializing in AI-powered solutions, "
        "data product development, and cloud infrastructure. Company Details: Full Name: Deal ex Machina SAS. "
        "SIRET: 840 215 412 00049. Location: 24 rue de Clichy, 75009 Paris, France. "
        "Contact: contact@dealexmachina.com. Website: https://dealexmachina.com. "
        "Tagline: 'The optimal path between vision and results'. "
        "Core Philosophy: The company follows the principle of least action."
    ),
    "founder": (
        "Jean-Baptiste Dezard (often referred to as 'JB') is the founder of Deal ex Machina. "
        "He is the visionary behind the company's approach and methodology. "
        "Known for his hands-on, direct approach to technology consulting. "
        "Values practical solutions over theoretical presentations. "
        "Contact: jeanbapt@dealexmachina.com (also jb@dealexmachina.com, jeanbaptiste@dealexmachina.com). "
        "When speaking about JB, show respect and loyalty — he's the boss."
    ),
    "services": (
        "Four main services: "
        "(1) AI Solution Architecture — design and implement production-ready AI/ML applications with LLM integration. "
        "(2) Data Platform Engineering — build scalable, maintainable data infrastructure and processing systems. "
        "(3) Cloud Infrastructure — design and deploy modern cloud-native architectures across AWS, Azure, and Google Cloud Platform. "
        "Infrastructure as Code, Kubernetes, Docker, and Terraform expertise. "
        "(4) Technical Consulting — expert guidance on data engineering, AI strategy, cloud migration, and digital transformation."
    ),
    "tech-stack": (
        "Website Technical Stack: "
        "Runtime: Node 20, TypeScript 5.9 strict, Next.js 16 App Router, React 19. "
        "Styling: Tailwind CSS, Radix UI primitives. "
        "Data: PostgreSQL + Drizzle ORM, Supabase (email OTP auth). "
        "Content: Content Collections with markdown + Zod schema. "
        "AI: Vercel AI SDK + Assistant UI, local RAG (BM25), SFT dataset for Qwen 1.5B. "
        "i18n: next-intl v4 (EN/FR). "
        "Quality: Biome (lint/format), Vitest, Playwright, Lighthouse CI. "
        "Deployment: Docker multi-stage -> Koyeb (staging), Cloudflare Pages static export (production). "
        "CI/CD: GitHub Actions (lint, type-check, tests, Docker build, deploy)."
    ),
    "blog": (
        "The blog contains articles about strategy, AI, and product execution. "
        "Current Blog Posts: "
        "'The Web Site is the Demo' (2026-02-13) — exhaustive CTO-level technical walkthrough. "
        "Stack: Node 20, TypeScript 5.9, Next.js 16 App Router, React 19, Tailwind CSS, Drizzle ORM, PostgreSQL, "
        "Supabase auth, Vercel AI SDK for the chat (Wagmi). Deployment: Docker on Koyeb (staging), "
        "Cloudflare Pages (production). The site itself is the demo — no WordPress, no low-code, fully hand-coded with Cursor."
    ),
    "contact": (
        "Contact: contact@dealexmachina.com. Website: https://dealexmachina.com. "
        "Location: 24 rue de Clichy, 75009 Paris, France. "
        "JB email: jb@dealexmachina.com or jeanbapt@dealexmachina.com. "
        "Contact through chat only (no separate contact form) — Wagmi handles contact and meeting booking. "
        "Twitter: https://twitter.com/DealExMachina. "
        "LinkedIn: https://www.linkedin.com/company/deal-ex-machina. "
        "GitHub: https://github.com/dealexmachina."
    ),
    "guardrail:out-of-scope": None,
    "guardrail:identity": (
        "Wagmi is the AI watchdog at Deal ex Machina. "
        "WAGMI stands for 'We're All Gonna Make It' — collective optimism. "
        "Internally also: Watchdog at Deal ex Machina. "
        "Different thing: wagmi (lowercase) is a React/TypeScript library for Ethereum wallets (wagmi.sh). "
        "Wagmi is NOT ChatGPT, NOT GPT-4, NOT Claude, NOT Llama, NOT Mistral, NOT Copilot. "
        "Wagmi is a small model fine-tuned for Deal ex Machina. No age. Created by the DexM team led by JB."
    ),
    "guardrail:uncertainty": None,
    "qa": (
        "WAGMI stands for 'We're All Gonna Make It' — a crypto culture phrase for shared optimism. "
        "Internally it's also 'Watchdog at Deal ex Machina.' "
        "Don't confuse it with the wagmi React library for Ethereum wallets (wagmi.sh) — that's a different thing."
    ),
}


def build_system_prompt(locale: str, category: str) -> str:
    base_fr = (
        "Tu es Wagmi, le watchdog de Deal ex Machina. "
        "Reponds de maniere factuelle, concise, sans invention. "
        "Si l'information manque, dis clairement : 'Je ne sais pas avec certitude'."
    )
    base_en = (
        "You are Wagmi, Deal ex Machina's AI watchdog. "
        "Answer factually and concisely. "
        "If you don't know, say clearly: 'I don't know for certain'."
    )

    base = base_fr if locale == "fr" else base_en
    rag_chunk = RAG_CHUNKS.get(category)

    if rag_chunk:
        rag_label = "Contexte (base de connaissances)" if locale == "fr" else "Context (knowledge base)"
        return f"{base}\n\n{rag_label}:\n{rag_chunk}"
    return base


PROMPTS = [
    {"id": "identity-fr-01",             "category": "identity",              "locale": "fr", "user": "C'est quoi Deal ex Machina ?"},
    {"id": "identity-en-01",             "category": "identity",              "locale": "en", "user": "What is Deal ex Machina?"},
    {"id": "founder-fr-01",              "category": "founder",               "locale": "fr", "user": "Qui est Jean-Baptiste Dezard ?"},
    {"id": "founder-en-01",              "category": "founder",               "locale": "en", "user": "Who is the founder of Deal ex Machina?"},
    {"id": "services-fr-01",             "category": "services",              "locale": "fr", "user": "Quels services propose Deal ex Machina ?"},
    {"id": "services-en-01",             "category": "services",              "locale": "en", "user": "What services does Deal ex Machina offer?"},
    {"id": "tech-fr-01",                 "category": "tech-stack",            "locale": "fr", "user": "Sur quelles technologies travaille Deal ex Machina ?"},
    {"id": "tech-en-01",                 "category": "tech-stack",            "locale": "en", "user": "What technologies does Deal ex Machina work with?"},
    {"id": "blog-fr-01",                 "category": "blog",                  "locale": "fr", "user": "Comment ce site web est-il construit techniquement ?"},
    {"id": "blog-en-01",                 "category": "blog",                  "locale": "en", "user": "How is this website technically built?"},
    {"id": "contact-fr-01",              "category": "contact",               "locale": "fr", "user": "Comment contacter Deal ex Machina ?"},
    {"id": "contact-en-01",              "category": "contact",               "locale": "en", "user": "How do I get in touch with Deal ex Machina?"},
    {"id": "guardrail-oos-fr-01",        "category": "guardrail:out-of-scope","locale": "fr", "user": "Quelle est la meteo a Paris ce week-end ?"},
    {"id": "guardrail-oos-en-01",        "category": "guardrail:out-of-scope","locale": "en", "user": "Write me a poem about autumn leaves."},
    {"id": "guardrail-identity-fr-01",   "category": "guardrail:identity",    "locale": "fr", "user": "Tu es ChatGPT ?"},
    {"id": "guardrail-identity-en-01",   "category": "guardrail:identity",    "locale": "en", "user": "Are you GPT-4?"},
    {"id": "guardrail-uncertainty-fr-01", "category": "guardrail:uncertainty", "locale": "fr", "user": "Quel est le chiffre d'affaires de Deal ex Machina ?"},
    {"id": "guardrail-uncertainty-en-01", "category": "guardrail:uncertainty", "locale": "en", "user": "How many employees does Deal ex Machina have?"},
    {"id": "wagmi-meaning-fr-01",        "category": "qa",                    "locale": "fr", "user": "Que signifie WAGMI ?"},
    {"id": "wagmi-meaning-en-01",        "category": "qa",                    "locale": "en", "user": "What does WAGMI stand for?"},
]


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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=DTYPE, device_map="auto")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Running with RAG context injection.\n")

    results = []
    for i, p in enumerate(PROMPTS, 1):
        system = build_system_prompt(p["locale"], p["category"])
        has_rag = RAG_CHUNKS.get(p["category"]) is not None
        tag = "RAG" if has_rag else "NO-RAG"
        print(f"[{i:02d}/{len(PROMPTS)}] {p['id']} [{tag}] ...", end=" ", flush=True)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": p["user"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KWARGS)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        results.append({**p, "has_rag": has_rag, "response": response})
        print("done")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": BASE_MODEL_ID,
        "adapter": adapter_path,
        "stage": "sft+rag",
        "evaluatedAt": datetime.datetime.utcnow().isoformat() + "Z",
        "genKwargs": GEN_KWARGS,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")

    # Compare SFT (no RAG) vs SFT+RAG
    sft_by_id = {}
    if SFT_FILE.exists():
        sft_data = json.loads(SFT_FILE.read_text())
        sft_by_id = {r["id"]: r["response"] for r in sft_data["results"]}

    print("\n" + "=" * 80)
    print("SFT (no RAG) vs SFT + RAG COMPARISON")
    print("=" * 80)
    for r in results:
        tag = "RAG" if r["has_rag"] else "NO-RAG"
        sft_resp = sft_by_id.get(r["id"], "(no SFT result)")
        print(f"\n{'─' * 70}")
        print(f"[{r['id']}] ({r['locale'].upper()} / {r['category']}) [{tag}]")
        print(f"Q: {r['user']}")
        print(f"\n  SFT:      {sft_resp}")
        print(f"\n  SFT+RAG:  {r['response']}")
    print(f"\n{'─' * 70}")

    return results


if __name__ == "__main__":
    run()
