"""Build model inputs from chat messages without calling PixtralProcessor(images=...).

Unsloth loads Mistral Small 3.1 with a Pixtral-style processor. A positional call like
``tokenizer(prompt_string, ...)`` is interpreted as *images*, not text, which raises
PIL errors. Encoding via ``apply_chat_template(..., tokenize=True)`` uses the text
tokenizer path and is safe for text-only SFT eval and inference.
"""


def build_generate_inputs(tokenizer, messages: list, device):
    batch = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    out = {}
    for key, val in dict(batch).items():
        out[key] = val.to(device) if hasattr(val, "to") else val
    return out
