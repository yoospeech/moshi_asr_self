"""DailyTalk user audio and aligned text for STT continuation."""

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
        user_begin_id: int,
        user_end_id: int,
        assistant_begin_id: int,
        assistant_end_id: int,
        user_channel: int = 1,
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
        self.boundary_ids = {
            "user": (user_begin_id, user_end_id),
            "assistant": (assistant_begin_id, assistant_end_id),
        }
        self.user_channel = user_channel
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
            "missing_user_alignment": 0,
        }
        for entry in raw_entries:
            duration = float(entry["duration"])
            if not min_duration <= duration <= max_duration:
                self.rejected["duration"] += 1
                continue
            audio_path = Path(entry["path"])
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            user_alignment_path = audio_path.with_name(f"user_{audio_path.stem}.json")
            if not user_alignment_path.exists():
                self.rejected["missing_user_alignment"] += 1
                continue
            user_alignments = json.loads(user_alignment_path.read_text(encoding="utf-8"))["alignments"]
            assistant_alignments = json.loads(
                audio_path.with_suffix(".json").read_text(encoding="utf-8")
            )["alignments"]
            word_count = len(user_alignments)
            if word_count < min_words:
                self.rejected["words"] += 1
                continue
            words_per_second = word_count / duration
            if not min_words_per_second <= words_per_second <= max_words_per_second:
                self.rejected["words_per_second"] += 1
                continue
            intervals = []
            invalid = False
            for _word, timestamps, _speaker in user_alignments + assistant_alignments:
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
            self.entries.append(
                {
                    **entry,
                    "user_alignments": user_alignments,
                    "assistant_alignments": assistant_alignments,
                }
            )
        self.filtered_hours = sum(float(entry["duration"]) for entry in self.entries) / 3600
        if not self.entries:
            raise ValueError(f"empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _word_tokens(self, word: str) -> list[int]:
        # Preserve SmolLM2's word-boundary BPE form for every aligned word.
        return self.tokenizer.encode(" " + word, add_special_tokens=False)

    def _aligned_text(
        self,
        user_alignments: list[Any],
        assistant_alignments: list[Any],
        frame_length: int,
    ) -> torch.Tensor:
        """Build one shared text stream from user and assistant turns.

        Only user audio is modeled. Both timestamped text sides are targets.
        PAD and EPAD keep their Moshi meanings, while BEGIN/END delimit every
        chronological speaker run. Compact collisions shift later text tokens
        instead of dropping a target.
        """
        words = [
            (float(timestamps[0]), "user", str(word))
            for word, timestamps, _speaker in user_alignments
        ] + [
            (float(timestamps[0]), "assistant", str(word))
            for word, timestamps, _speaker in assistant_alignments
        ]
        words.sort(key=lambda item: item[0])

        tokens: list[int] = [self.text_padding_id] * frame_length
        occupied = torch.zeros(frame_length, dtype=torch.bool)
        if not words:
            return torch.tensor(tokens, dtype=torch.long)

        def ensure(position: int) -> None:
            if position >= len(tokens):
                extension = position + 1 - len(tokens)
                tokens.extend([self.text_padding_id] * extension)
                occupied.resize_(len(tokens))
                occupied[-extension:] = False

        current_speaker: str | None = None
        last_piece_end = 0
        for start_seconds, speaker, word in words:
            desired_start = max(0, round(start_seconds * self.frame_rate))
            pieces = self._word_tokens(word)
            if not pieces:
                continue

            first_word_in_region = speaker != current_speaker
            if first_word_in_region:
                if current_speaker is not None:
                    end_position = last_piece_end
                    ensure(end_position)
                    while occupied[end_position]:
                        end_position += 1
                        ensure(end_position)
                    tokens[end_position] = self.boundary_ids[current_speaker][1]
                    occupied[end_position] = True

                # Region prefixes require two slots. Only this first word may
                # move when the timestamp leaves no room for EPAD + BEGIN.
                word_start = max(2, desired_start)
                ensure(word_start + len(pieces) - 1)
                while occupied[word_start - 2] or occupied[word_start - 1]:
                    word_start += 1
                    ensure(word_start + len(pieces) - 1)
                tokens[word_start - 2] = self.text_epad_id
                tokens[word_start - 1] = self.boundary_ids[speaker][0]
                occupied[word_start - 2 : word_start] = True
                current_speaker = speaker
            else:
                # Paper-faithful word alignment: do not shift a normal word to
                # make room for EPAD. Omit EPAD if it would overwrite text.
                word_start = desired_start
                ensure(word_start + len(pieces) - 1)
                epad_position = max(0, word_start - 1)
                if not occupied[epad_position]:
                    tokens[epad_position] = self.text_epad_id
                    occupied[epad_position] = True

            for offset, token in enumerate(pieces):
                position = word_start + offset
                ensure(position)
                # Preserve an earlier word piece on rare BPE overlap, matching
                # the same no-overwrite principle used for EPAD.
                if occupied[position]:
                    continue
                tokens[position] = token
                occupied[position] = True
            last_piece_end = max(last_piece_end, word_start + len(pieces))

        if current_speaker is not None:
            end_position = last_piece_end
            ensure(end_position)
            while occupied[end_position]:
                end_position += 1
                ensure(end_position)
            tokens[end_position] = self.boundary_ids[current_speaker][1]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        audio_path = Path(entry["path"])
        if not audio_path.is_absolute():
            audio_path = self.root / audio_path
        wav, _ = sphn.read(audio_path, sample_rate=self.sample_rate)
        if wav.ndim == 1:
            main = wav
        else:
            if self.user_channel >= wav.shape[0]:
                raise ValueError(f"{audio_path}: missing channel {self.user_channel}")
            main = wav[self.user_channel]
        waveform = torch.from_numpy(main).float().clamp(-1.0, 1.0)
        samples_per_frame = self.sample_rate / self.frame_rate
        frame_length = max(1, math.ceil(waveform.numel() / samples_per_frame))
        text_tokens = self._aligned_text(
            entry["user_alignments"], entry["assistant_alignments"], frame_length
        )
        return {
            "id": audio_path.stem,
            "waveform": waveform,
            "text_tokens": text_tokens,
            "audio_frame_length": frame_length,
            "text_frame_length": text_tokens.numel(),
        }


def collate_dailytalk(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_samples = max(item["waveform"].numel() for item in batch)
    max_frames = max(item["text_frame_length"] for item in batch)
    waveforms = torch.zeros(len(batch), 1, max_samples, dtype=torch.float32)
    text_tokens = torch.full((len(batch), max_frames), -1, dtype=torch.long)
    for index, item in enumerate(batch):
        sample_length = item["waveform"].numel()
        frame_length = item["text_frame_length"]
        waveforms[index, 0, :sample_length] = item["waveform"]
        text_tokens[index, :frame_length] = item["text_tokens"]
    return {
        "waveforms": waveforms,
        "text_tokens": text_tokens,
        "audio_frame_lengths": torch.tensor(
            [item["audio_frame_length"] for item in batch], dtype=torch.long
        ),
        "text_frame_lengths": torch.tensor(
            [item["text_frame_length"] for item in batch], dtype=torch.long
        ),
        "ids": [item["id"] for item in batch],
    }
