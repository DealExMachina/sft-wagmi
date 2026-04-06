"""Build text-only generation inputs across tokenizers and processors.

Mistral Small 3.x via Unsloth can expose a multimodal processor. To avoid processor
chat-template/image edge cases in text-only flows, we always route through the
underlying text tokenizer when available.
"""


def build_generate_inputs(tokenizer, messages: list, device):
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    prompt = text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = text_tokenizer(prompt, return_tensors="pt")
    out = {}
    for key, val in dict(batch).items():
        out[key] = val.to(device) if hasattr(val, "to") else val
    return out
