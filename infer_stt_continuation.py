"""Reference inference with forced user-audio codes and sampled shared text."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import sphn
import torch
from transformers import AutoTokenizer

from moshi.models import loaders as moshi_loaders
from smollm_moshi_stt_continuation import STTContinuationConfig, SmolLM2MoshiSTTContinuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="defaults to config.json beside checkpoint")
    parser.add_argument("--audio", required=True, help="stereo or mono user-audio input")
    parser.add_argument("--user-channel", type=int, default=1)
    parser.add_argument("--model", help="override temporal_model in config")
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--mimi-repo", default=moshi_loaders.DEFAULT_REPO)
    parser.add_argument("--mimi-device")
    parser.add_argument("--tail-seconds", type=float, default=10.0)
    parser.add_argument("--text-temperature", type=float, default=0.7)
    parser.add_argument("--text-top-k", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", default="inference_text.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> STTContinuationConfig:
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    names = {field.name for field in fields(STTContinuationConfig)}
    values = {key: value for key, value in raw.items() if key in names}
    if args.model:
        values["temporal_model"] = args.model
    return STTContinuationConfig(**values)


def sample_text(logits: torch.Tensor, *, temperature: float, top_k: int, greedy: bool) -> int:
    if greedy or temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scores = logits.float() / temperature
    if 0 < top_k < scores.shape[-1]:
        values, indices = torch.topk(scores, top_k, dim=-1)
        sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return int(indices.gather(-1, sampled).item())
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())


def load_user_waveform(path: Path, channel: int, sample_rate: int) -> torch.Tensor:
    wav, _ = sphn.read(path, sample_rate=sample_rate)
    if wav.ndim == 1:
        if channel != 0:
            print(f"warning: mono input; using channel 0 instead of requested {channel}")
        selected = wav
    else:
        if channel >= wav.shape[0]:
            raise ValueError(f"{path}: missing channel {channel}; shape={wav.shape}")
        selected = wav[channel]
    return torch.from_numpy(selected).float().clamp(-1.0, 1.0)


@torch.inference_mode()
def force_audio_sample_text(
    model: SmolLM2MoshiSTTContinuation,
    audio_codes: torch.Tensor,
    config: STTContinuationConfig,
    *,
    temperature: float,
    top_k: int,
    greedy: bool,
) -> list[int]:
    """Force every delayed audio input and sample only the delayed text stream."""
    if audio_codes.shape[0] != 1:
        raise ValueError("reference inference currently supports batch size 1")
    if audio_codes.shape[1] != config.num_codebooks:
        raise ValueError(
            f"Mimi returned {audio_codes.shape[1]} codebooks; expected {config.num_codebooks}"
        )

    device = model.device
    audio_codes = audio_codes.to(device)
    initial = model._get_initial_token()
    pattern = initial
    generated: list[int] = []
    total_pattern_steps = audio_codes.shape[-1] + max(config.delays)

    for step in range(total_pattern_steps):
        _, text_logits = model.forward_text(pattern)
        frame = torch.empty(
            (1, model.num_codebooks, 1), dtype=torch.long, device=device
        )

        if step < config.text_delay_steps:
            frame[:, 0, 0] = model.text_initial_token_id
        else:
            token = sample_text(
                text_logits[0, 0, -1],
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
            )
            frame[:, 0, 0] = token
            generated.append(token)

        # Audio is never sampled. Each codebook is copied from the user Mimi
        # codes at the source frame selected by Moshi's delay pattern.
        for codebook in range(config.num_codebooks):
            delay = config.delays[codebook + model.audio_offset]
            source = step - delay
            if 0 <= source < audio_codes.shape[-1]:
                frame[:, codebook + model.audio_offset, 0] = audio_codes[:, codebook, source]
            else:
                frame[:, codebook + model.audio_offset, 0] = model.initial_token_id

        pattern = torch.cat([pattern, frame], dim=-1)
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == total_pattern_steps:
            print(f"inference frame {step + 1}/{total_pattern_steps}", flush=True)
    return generated


def render_tokens(
    token_ids: list[int], tokenizer: AutoTokenizer, config: STTContinuationConfig
) -> tuple[str, list[dict[str, object]]]:
    special = {
        config.text_padding_id: "PAD",
        config.text_epad_id: "EPAD",
        config.user_begin_id: "USER_BEGIN",
        config.user_end_id: "USER_END",
        config.assistant_begin_id: "ASSISTANT_BEGIN",
        config.assistant_end_id: "ASSISTANT_END",
    }
    events: list[dict[str, object]] = []
    lexical: list[int] = []
    for frame, token_id in enumerate(token_ids):
        if token_id in special:
            label = special[token_id]
        elif 0 <= token_id < len(tokenizer):
            label = tokenizer.decode([token_id], skip_special_tokens=False)
            lexical.append(token_id)
        else:
            label = f"UNKNOWN_{token_id}"
        events.append({"frame": frame, "time": frame / config.frame_rate, "id": token_id, "token": label})
    return tokenizer.decode(lexical, skip_special_tokens=False), events


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.temporal_model)
    model = SmolLM2MoshiSTTContinuation(config, dtype=dtype, device=device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(package["model"], strict=True)
    model.eval()

    mimi_checkpoint = args.mimi_checkpoint
    if mimi_checkpoint is None:
        mimi_checkpoint = str(moshi_loaders.hf_get(moshi_loaders.MIMI_NAME, args.mimi_repo))
    mimi_device = torch.device(args.mimi_device) if args.mimi_device else device
    mimi = moshi_loaders.get_mimi(
        mimi_checkpoint, device=mimi_device, num_codebooks=config.num_codebooks
    )
    mimi.eval()

    waveform = load_user_waveform(Path(args.audio), args.user_channel, mimi.sample_rate)
    tail_samples = max(0, round(args.tail_seconds * mimi.sample_rate))
    if tail_samples:
        waveform = torch.cat([waveform, torch.zeros(tail_samples)])
    audio_codes = mimi.encode(waveform[None, None].to(mimi_device))
    generated = force_audio_sample_text(
        model,
        audio_codes,
        config,
        temperature=args.text_temperature,
        top_k=args.text_top_k,
        greedy=args.greedy,
    )
    decoded, events = render_tokens(generated, tokenizer, config)
    payload = {
        "audio": str(Path(args.audio)),
        "user_channel": args.user_channel,
        "checkpoint_step": package.get("step"),
        "decoded_text": decoded,
        "events": events,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(decoded)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
