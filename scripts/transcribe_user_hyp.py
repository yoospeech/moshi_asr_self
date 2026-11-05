#!/usr/bin/env python3
"""Attach whisper-timestamped transcripts and word timestamps to a manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import sphn
import torch
import whisper_timestamped as whisper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read stereo audio paths from a manifest, transcribe only the user "
            "channel, and add user_hyp to each row."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--user-channel", type=int, default=1,
        help="zero-based channel index in each stereo WAV (default: 1/right)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="output JSONL; defaults to MANIFEST.with_suffix('.user_hyp.jsonl')",
    )
    parser.add_argument("--model", default="medium")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="inference device; auto uses CUDA when available (default: auto)",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--user-json-dir", type=Path,
        help="directory for user_N.json sidecars; defaults to each WAV's directory",
    )
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--missing", choices=("keep", "error", "drop"), default="keep")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def transcribe(
    model: Any,
    audio: np.ndarray,
    *,
    language: str | None,
    beam_size: int,
    vad: bool,
) -> dict[str, Any]:
    duration = audio.shape[-1] / 16_000
    result = whisper.transcribe(
        model,
        audio, task="transcribe", language=language,
        vad="auditok" if vad and duration > 10 else False,
        best_of=5, beam_size=beam_size,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_previous_text=False,
        fp16=model.device.type == "cuda", verbose=None,
    )
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    text_parts = []
    for segment in result.get("segments", []):
        segment_words = []
        for word in segment.get("words", []):
            item = {
                "word": word["text"].strip(),
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "probability": round(float(word.get("confidence", 0.0)), 6),
            }
            if item["word"]:
                segment_words.append(item)
                words.append(item)
        segment_text = segment.get("text", "").strip()
        if segment_text:
            text_parts.append(segment_text)
        segments.append({
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "text": segment_text,
            "words": segment_words,
        })
    language_probability = result.get("language_probs", {}).get(
        result.get("language", language)
    )
    return {
        "text": " ".join(text_parts).strip(),
        "language": result.get("language", language),
        "language_probability": (
            round(float(language_probability), 6)
            if language_probability is not None else None
        ),
        "words": words,
        "segments": segments,
    }


def user_sidecar(hypothesis: dict[str, Any]) -> dict[str, Any]:
    return {
        "alignments": [
            [
                word["word"],
                [word["start"], word["end"]],
                "SPEAKER_USER",
            ]
            for word in hypothesis["words"]
        ]
    }


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve() if args.output else manifest.with_suffix(".user_hyp.jsonl")
    if output == manifest:
        raise SystemExit("refusing to overwrite the input manifest")
    cache_dir = (
        args.cache_dir.resolve() if args.cache_dir
        else output.parent / f".{output.stem}_cache"
    )
    rows = read_jsonl(manifest)
    if not 1 <= args.start_row <= len(rows):
        raise SystemExit(f"--start-row must be between 1 and {len(rows)}")

    candidates: list[tuple[int, Path]] = []
    missing_rows = []
    for row_number in range(args.start_row, len(rows) + 1):
        audio_path = Path(rows[row_number - 1]["path"])
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        if audio_path.is_file():
            candidates.append((row_number, audio_path))
        else:
            missing_rows.append(row_number)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if args.missing == "error" and missing_rows:
        preview = ", ".join(map(str, missing_rows[:10]))
        raise SystemExit(f"missing user WAV for {len(missing_rows)} rows; first: {preview}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading whisper-timestamped model={args.model} device={device}", flush=True)
    model = whisper.load_model(args.model, device=device)
    for position, (row_number, audio_path) in enumerate(candidates, start=1):
        cache_path = cache_dir / f"row_{row_number:06d}.json"
        if cache_path.exists() and not args.overwrite_cache:
            print(f"[{position}/{len(candidates)}] cached {audio_path.name}", flush=True)
            continue
        waveform, _ = sphn.read(audio_path, sample_rate=16_000)
        if waveform.ndim != 2:
            raise RuntimeError(f"{audio_path}: expected stereo audio, got shape {waveform.shape}")
        if not 0 <= args.user_channel < waveform.shape[0]:
            raise RuntimeError(
                f"{audio_path}: user channel {args.user_channel} is unavailable; "
                f"audio has {waveform.shape[0]} channels"
            )
        user_audio = np.ascontiguousarray(waveform[args.user_channel], dtype=np.float32)
        print(
            f"[{position}/{len(candidates)}] transcribing {audio_path.name} "
            f"channel={args.user_channel}",
            flush=True,
        )
        hypothesis = transcribe(
            model, user_audio, language=args.language or None,
            beam_size=args.beam_size, vad=not args.no_vad,
        )
        atomic_json(cache_path, {
            "manifest_row": row_number,
            "user_channel": args.user_channel,
            "user_hyp": hypothesis,
        })

    output_rows = []
    transcribed = 0
    dropped = 0
    for row_number, row in enumerate(rows, start=1):
        cache_path = cache_dir / f"row_{row_number:06d}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            row = dict(row)
            row["user_channel"] = cached["user_channel"]
            row["user_hyp"] = cached["user_hyp"]
            audio_path = Path(row["path"])
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            sidecar_dir = (
                args.user_json_dir.resolve()
                if args.user_json_dir is not None
                else audio_path.parent
            )
            sidecar_path = sidecar_dir / f"user_{audio_path.stem}.json"
            atomic_json(
                sidecar_path,
                user_sidecar(cached["user_hyp"]),
            )
            transcribed += 1
            output_rows.append(row)
        elif args.missing == "drop":
            dropped += 1
        else:
            output_rows.append(row)
    atomic_jsonl(output, output_rows)
    print(
        f"wrote {output}: rows={len(output_rows)} transcribed={transcribed} "
        f"missing_or_unprocessed={len(rows) - transcribed} dropped={dropped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
