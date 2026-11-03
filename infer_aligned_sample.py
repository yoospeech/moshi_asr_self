"""Synthesize manifest samples from their exact training text timelines."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import sphn
import torch
from transformers import AutoTokenizer

from infer_tts import generate_codes, write_wav
from smollm_moshi_tts import SmolLM2MoshiTTS, TTSModelConfig
from tts_dataset import DailyTalkDataset
from moshi.models import loaders as moshi_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--sample-path",
        action="append",
        required=True,
        help="manifest path field to synthesize; may be repeated",
    )
    parser.add_argument("--config", help="defaults to config.json beside checkpoint")
    parser.add_argument("--model", help="override temporal model path from config")
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--mimi-checkpoint")
    parser.add_argument("--mimi-repo", default=moshi_loaders.DEFAULT_REPO)
    parser.add_argument("--mimi-device")
    parser.add_argument("--audio-temperature", type=float, default=0.6)
    parser.add_argument("--audio-top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TTSModelConfig:
    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint.parent / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(TTSModelConfig)}
    values = {name: value for name, value in raw.items() if name in valid}
    if args.model:
        values["temporal_model"] = args.model
    return TTSModelConfig(**values)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    config = load_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.temporal_model)
    dataset = DailyTalkDataset(
        args.manifest,
        tokenizer,
        frame_rate=config.frame_rate,
        text_padding_id=config.text_padding_id,
        text_epad_id=config.text_epad_id,
    )
    entries = {entry["path"]: index for index, entry in enumerate(dataset.entries)}
    missing = [path for path in args.sample_path if path not in entries]
    if missing:
        raise SystemExit(f"samples are absent or rejected by dataset filters: {missing}")

    items = [dataset[entries[path]] for path in args.sample_path]
    timelines = [item["text_tokens"] for item in items]
    model = SmolLM2MoshiTTS(config, dtype=dtype, device=device)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(package["model"], strict=True)
    model.eval()
    codes, lengths = generate_codes(
        model,
        timelines,
        temperature=args.audio_temperature,
        top_k=args.audio_top_k,
    )

    mimi_checkpoint = args.mimi_checkpoint
    if mimi_checkpoint is None:
        mimi_checkpoint = str(
            moshi_loaders.hf_get(moshi_loaders.MIMI_NAME, args.mimi_repo)
        )
    mimi_device = torch.device(args.mimi_device) if args.mimi_device else device
    mimi = moshi_loaders.get_mimi(
        mimi_checkpoint,
        device=mimi_device,
        num_codebooks=config.num_codebooks,
    ).eval()
    generated = mimi.decode(codes.to(mimi_device)).cpu()

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_per_frame = round(24_000 / config.frame_rate)
    rows = []
    for index, (path_text, frame_length) in enumerate(
        zip(args.sample_path, lengths), 1
    ):
        audio_path = Path(path_text)
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        metadata = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))
        transcript = " ".join(
            word for word, _timestamps, _speaker in metadata["alignments"]
        )
        stem = f"sample_{index:02d}_{audio_path.stem}"
        write_wav(
            output_dir / f"{stem}_generated.wav",
            generated[index - 1, 0, : frame_length * samples_per_frame],
        )
        waveform, _ = sphn.read(audio_path, sample_rate=24_000)
        reference = torch.from_numpy(
            waveform if waveform.ndim == 1 else waveform[0]
        ).float()
        write_wav(output_dir / f"{stem}_reference.wav", reference)
        rows.append((stem, frame_length / config.frame_rate, transcript))

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": package.get("step"),
        "delay_pattern": config.delays,
        "audio_temperature": args.audio_temperature,
        "audio_top_k": args.audio_top_k,
        "seed": args.seed,
        "samples": [
            {"name": name, "duration": duration, "transcript": transcript}
            for name, duration, transcript in rows
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(rows)} reference/generated pairs to {output_dir}")


if __name__ == "__main__":
    main()
