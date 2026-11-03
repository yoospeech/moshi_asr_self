"""Verify that a local SmolLM2 checkpoint can provide TTS text features.

This mirrors the project's LFM feature-extraction convention: a system prompt
asks the model to repeat a user sentence and the same sentence is teacher
forced as the assistant turn.  The script prints the last-layer assistant
hidden-state shape, which is the tensor a TTS conditioner would consume.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ECHO_SYSTEM_PROMPT = (
    "Repeat the user's message exactly. Output only the original message, "
    "without explanation or quotation marks."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/SmolLM2-135M")
    parser.add_argument("--text", default="Hello, this is a TTS smoke test adfjasdf asdfkjal;sd dasdfas asdf asdf asdf asfd asfd")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    messages = [
        {"role": "system", "content": ECHO_SYSTEM_PROMPT},
        {"role": "user", "content": args.text},
    ]
    if tokenizer.chat_template:
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": args.text}],
            tokenize=True, add_generation_prompt=False,
            return_tensors="pt", return_dict=True,
        ).to(device)
        prompt_format = "chat template"
    else:
        # The base checkpoint has no chat template. Keep this plain prompt
        # explicit and identical for training and inference.
        prefix_text = (
            f"System: {ECHO_SYSTEM_PROMPT}\n"
            f"User: {args.text}\n"
            "Assistant:"
        )
        prefix = tokenizer(prefix_text, return_tensors="pt").to(device)
        full = tokenizer(prefix_text + args.text, return_tensors="pt").to(device)
        prompt_format = "plain fallback"

    with torch.inference_mode():
        output = model(**full, output_hidden_states=True, return_dict=True, use_cache=False)
        generated = model.generate(**prefix, do_sample=False, max_new_tokens=32)

    start = prefix["input_ids"].shape[1]
    token_ids = full["input_ids"][0, start:]
    hidden = output.hidden_states[-1][0, start:]
    special_ids = set(tokenizer.all_special_ids)
    keep = torch.tensor([item.item() not in special_ids for item in token_ids], device=device)
    generated_text = tokenizer.decode(generated[0, start:], skip_special_tokens=True)

    print(f"model: {args.model}")
    print(f"prompt format: {prompt_format}")
    print(f"device: {device}, dtype: {dtype}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"assistant feature shape: {tuple(hidden[keep].shape)}")
    print(f"input text: {args.text!r}")
    print(f"greedy reply: {generated_text!r}")
    print(f"exact match: {generated_text.strip() == args.text.strip()}")


if __name__ == "__main__":
    main()
