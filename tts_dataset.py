"""Raw DailyTalk input and SmolLM2-aligned text timelines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import sphn
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class DailyTalkDataset(Dataset[dict[str, Any]]):
    """Read target speech and word-level alignments from DailyTalkContiguous.

    Audio tokenization intentionally happens after collation so frozen Mimi can
    run online on the training device.
    """

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        sample_rate: int = 24000,
        frame_rate: float = 12.5,
        text_padding_id: int,
        text_epad_id: int,
        main_channel: int = 0,
        min_duration: float = 5.0,
        max_duration: float = 45.0,
        min_words: int = 5,
        min_words_per_second: float = 0.5,
        max_words_per_second: float = 2.5,
        min_speech_coverage: float = 0.05,
        max_speech_coverage: float = 0.8,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.text_padding_id = text_padding_id
        self.text_epad_id = text_epad_id
        self.main_channel = main_channel
        with self.manifest.open(encoding="utf-8") as handle:
            raw_entries = [json.loads(line) for line in handle if line.strip()]
        self.original_size = len(raw_entries)
        self.entries = []
        self.rejected: dict[str, int] = {
            "duration": 0,
            "words": 0,
            "words_per_second": 0,
            "alignment": 0,
            "speech_coverage": 0,
        }
        for entry in raw_entries:
            duration = float(entry["duration"])
            if not min_duration <= duration <= max_duration:
                self.rejected["duration"] += 1
                continue
            audio_path = Path(entry["path"])
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            alignments = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))["alignments"]
            word_count = len(alignments)
            if word_count < min_words:
                self.rejected["words"] += 1
                continue
            words_per_second = word_count / duration
            if not min_words_per_second <= words_per_second <= max_words_per_second:
                self.rejected["words_per_second"] += 1
                continue
            intervals = []
            invalid = False
            for _word, timestamps, _speaker in alignments:
                start, end = map(float, timestamps)
                if start < 0 or end < start or end > duration + 0.25:
                    invalid = True
                    break
                intervals.append((max(0.0, start), min(duration, end)))
            if invalid:
                self.rejected["alignment"] += 1
                continue
            intervals.sort()
            merged: list[list[float]] = []
            for start, end in intervals:
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            speech_coverage = sum(end - start for start, end in merged) / duration
            if not min_speech_coverage <= speech_coverage <= max_speech_coverage:
                self.rejected["speech_coverage"] += 1
                continue
            self.entries.append(entry)
        self.filtered_hours = sum(float(entry["duration"]) for entry in self.entries) / 3600
        if not self.entries:
            raise ValueError(f"empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _word_tokens(self, word: str) -> list[int]:
        # Preserve SmolLM2's word-boundary BPE form for every aligned word.
        return self.tokenizer.encode(" " + word, add_special_tokens=False)

    def _aligned_text(self, alignments: list[Any], frame_length: int) -> torch.Tensor:
        text = torch.full((frame_length,), self.text_padding_id, dtype=torch.long)
        occupied = torch.zeros(frame_length, dtype=torch.bool)
        for word, timestamps, _speaker in alignments:
            start = min(frame_length - 1, max(0, round(float(timestamps[0]) * self.frame_rate)))
            epad = max(0, start - 1)
            if not occupied[epad]:
                text[epad] = self.text_epad_id
                occupied[epad] = True
            for offset, token in enumerate(self._word_tokens(str(word))):
                position = start + offset
                if position >= frame_length:
                    break
                # Equation 5 forbids overwriting pieces from a preceding word.
                if occupied[position]:
                    continue
                text[position] = token
                occupied[position] = True
        return text

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        metadata = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))
        wav, _ = sphn.read(audio_path, sample_rate=self.sample_rate)
        if wav.ndim == 1:
            main = wav
        else:
            if self.main_channel >= wav.shape[0]:
                raise ValueError(f"{audio_path}: missing channel {self.main_channel}")
            main = wav[self.main_channel]
        waveform = torch.from_numpy(main).float().clamp(-1.0, 1.0)
        samples_per_frame = self.sample_rate / self.frame_rate
        frame_length = max(1, math.ceil(waveform.numel() / samples_per_frame))
        return {
            "id": audio_path.stem,
            "waveform": waveform,
            "text_tokens": self._aligned_text(metadata["alignments"], frame_length),
            "frame_length": frame_length,
        }


def collate_dailytalk(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_samples = max(item["waveform"].numel() for item in batch)
    max_frames = max(item["frame_length"] for item in batch)
    waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    text_tokens = torch.full((len(batch), max_frames), -1, dtype=torch.long)
    for index, item in enumerate(batch):
        sample_length = item["waveform"].numel()
        frame_length = item["frame_length"]
        waveforms[index, 0, :sample_length] = item["waveform"]
        text_tokens[index, :frame_length] = item["text_tokens"]
    return {
        "waveforms": waveforms,
        "text_tokens": text_tokens,
        "frame_lengths": torch.tensor([item["frame_length"] for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
    }
