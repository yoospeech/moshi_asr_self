"""Offline batched synthesis for SmolLM2 + Moshi Depformer TTS checkpoints.

This intentionally favors a simple, verifiable inference path over speed: the
full temporal prefix is recomputed for every 12.5 Hz frame.  It does not depend
on Moshi's streaming cache implementation.
"""

from __future__ import annotations

import argparse
import json
import wave
from array import array
from dataclasses import fields
from pathlib import Path

import torch
from transformers import AutoTokenizer

from smollm_moshi_tts import SmolLM2MoshiTTS, TTSModelConfig
from moshi.models import loaders as moshi_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="defaults to config.json beside the checkpoint")
    parser.add_argument("--model", help="override the temporal model path stored in config.json")
    parser.add_argument("--text", action="append", help="text to synthesize; may be repeated")
    parser.add_argument("--text-file", help="UTF-8 file containing one utterance per line")
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--mimi-repo", default=moshi_loaders.DEFAULT_REPO)
    parser.add_argument("--mimi-device", help="e.g. cuda:1; defaults to the model device")
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=250)
    parser.add_argument("--frames-per-word", type=int, default=5)
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.text or [])
    if args.text_file:
        texts.extend(
            line.strip()
            for line in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts:
        raise SystemExit("provide at least one --text or --text-file")
    return texts


def load_config(args: argparse.Namespace) -> TTSModelConfig:
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    valid_names = {item.name for item in fields(TTSModelConfig)}
    values = {name: value for name, value in raw.items() if name in valid_names}
    if args.model:
        values["temporal_model"] = args.model
    return TTSModelConfig(**values)


def make_text_timeline(
    text: str,
    tokenizer: AutoTokenizer,
    config: TTSModelConfig,
    frames_per_word: int,
    tail_frames: int,
    max_frames: int,
) -> torch.Tensor:
    """Create an alignment-free approximation of the training text timeline."""
    timeline = torch.full((max_frames,), config.text_padding_id, dtype=torch.long)
    occupied = torch.zeros(max_frames, dtype=torch.bool)
    cursor = 1
    for word in text.split():
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if not pieces:
            continue
        if cursor >= max_frames - tail_frames:
            break
        epad = cursor - 1
        if not occupied[epad]:
            timeline[epad] = config.text_epad_id
            occupied[epad] = True
        for offset, token in enumerate(pieces):
            position = cursor + offset
            if position >= max_frames - tail_frames:
                break
            timeline[position] = token
            occupied[position] = True
        cursor += max(frames_per_word, len(pieces))
    used_frames = min(max_frames, cursor + tail_frames)
    return timeline[:used_frames]


def sample_top_k(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if 0 < top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, top_k, dim=-1)
        sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return indices.gather(-1, sampled).squeeze(-1)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)


@torch.inference_mode()
def generate_codes(
    model: SmolLM2MoshiTTS,
    timelines: list[torch.Tensor],
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, list[int]]:
    device = model.device
    batch_size = len(timelines)
    lengths = [timeline.numel() for timeline in timelines]
    audio_frames = max(lengths)
    text = torch.full(
        (batch_size, audio_frames), model.tts_config.text_padding_id,
        dtype=torch.long, device=device,
    )
    for index, timeline in enumerate(timelines):
        text[index, : timeline.numel()] = timeline.to(device)

    initial = model._get_initial_token().expand(batch_size, -1, -1)
    pattern = initial
    pattern_frames = audio_frames + max(model.delays)
    for frame in range(pattern_frames):
        transformer_out, _ = model.forward_text(pattern)
        temporal = transformer_out[:, -1:]
        target = torch.full(
            (batch_size, model.num_codebooks, 1), model.initial_token_id,
            dtype=torch.long, device=device,
        )
        target[:, 0, 0] = text[:, frame] if frame < audio_frames else model.text_padding_token_id

        # Recompute the tiny depth sequence after each sampled RQ.  Position q
        # is causal and consumes the already sampled q-1 token.
        for codebook in range(model.dep_q):
            if frame < model.delays[codebook + model.audio_offset]:
                continue
            depth_logits = model.forward_depformer_training(target, temporal)
            target[:, codebook + model.audio_offset, 0] = sample_top_k(
                depth_logits[:, codebook, 0], temperature, top_k
            )
        pattern = torch.cat((pattern, target), dim=-1)
        if frame == 0 or (frame + 1) % 25 == 0 or frame + 1 == pattern_frames:
            print(f"generating frame {frame + 1}/{pattern_frames}", flush=True)

    codes = torch.empty(
        batch_size, model.dep_q, audio_frames, dtype=torch.long, device=device
    )
    # pattern[:, :, 0] is the initial input; generated pattern frame p is p+1.
    for codebook in range(model.dep_q):
        delay = model.delays[codebook + model.audio_offset]
        codes[:, codebook] = pattern[:, codebook + model.audio_offset, 1 + delay : 1 + delay + audio_frames]
    return codes, lengths


def write_wav(path: Path, waveform: torch.Tensor, sample_rate: int = 24_000) -> None:
    samples = waveform.detach().float().cpu().clamp(-1, 1)
    pcm = array("h", (samples * 32767.0).to(torch.int16).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    texts = load_texts(args)
    config = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.temporal_model)
    model = SmolLM2MoshiTTS(config, dtype=dtype, device=device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(package["model"], strict=True)
    model.eval()
    print(f"loaded checkpoint step={package.get('step', 'unknown')} batch={len(texts)}")

    tail_frames = max(1, round(args.tail_seconds * config.frame_rate))
    max_frames = max(1, round(args.max_seconds * config.frame_rate))
    timelines = [
        make_text_timeline(
            text, tokenizer, config, args.frames_per_word, tail_frames, max_frames
        )
        for text in texts
    ]
    codes, lengths = generate_codes(
        model, timelines, args.audio_temperature, args.audio_top_k
    )

    mimi_checkpoint = args.mimi_checkpoint
    if mimi_checkpoint is None:
        mimi_checkpoint = str(moshi_loaders.hf_get(moshi_loaders.MIMI_NAME, args.mimi_repo))
    mimi_device = torch.device(args.mimi_device) if args.mimi_device else device
    mimi = moshi_loaders.get_mimi(
        mimi_checkpoint, device=mimi_device, num_codebooks=config.num_codebooks
    )
    mimi.eval()
    waveforms = mimi.decode(codes.to(mimi_device)).cpu()

    output_dir = Path(args.output_dir)
    samples_per_frame = round(24_000 / config.frame_rate)
    for index, (text, frame_length) in enumerate(zip(texts, lengths)):
        output = output_dir / f"sample_{index:03d}.wav"
        write_wav(output, waveforms[index, 0, : frame_length * samples_per_frame])
        print(f"saved {output}: {text}")


if __name__ == "__main__":
    main()
