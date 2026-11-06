"""Offline regression checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel

from stt_continuation.dataset import DailyTalkDataset, TextAlignmentCollision
from stt_continuation.model import (
    STTContinuationConfig, SmolLM2MoshiSTTContinuation, masked_text_continuation_loss,
)
from stt_continuation.inference import force_audio_sample_text


class AlignmentTests(unittest.TestCase):
    def test_manifest_targets_and_audio_channel(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" user": [1], " assistant": [2]}[text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.json").write_text(json.dumps({"alignments": [
                ["assistant", [.8, 1.0], "SPEAKER_MAIN"]
            ]}))
            # A conflicting sidecar must not override the embedded hypothesis.
            (root / "user_0.json").write_text('{"alignments": []}')
            manifest = root / "hyp.jsonl"
            manifest.write_text(json.dumps({
                "path": "0.wav", "duration": 2, "user_channel": 0,
                "user_hyp": {"words": [{"word": "user", "start": .8, "end": 1.0}]},
            }) + "\n")
            ds = DailyTalkDataset(
                manifest, Tokenizer(), text_padding_id=10, text_epad_id=11,
                min_duration=0, min_words=1, min_words_per_second=0,
                min_speech_coverage=0, max_speech_coverage=1,
            )
            self.assertEqual(ds.entries[0]["user_alignment_source"], "manifest:user_hyp.words")
            audio = torch.stack([torch.ones(48000), torch.zeros(48000)]).numpy()
            with patch("stt_continuation.dataset.sphn.read", return_value=(audio, 24000)):
                item = ds[0]
            self.assertTrue((item["waveform"] == 1).all())
            self.assertEqual(item["asr_tokens"][10].item(), 1)
            self.assertEqual(item["continuation_tokens"][10].item(), 2)

    def test_padding_and_separate_channels(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" multi": [1, 2], " next": [3], " reply": [4]}[text]

        ds = object.__new__(DailyTalkDataset)
        ds.tokenizer = Tokenizer()
        ds.frame_rate = 12.5
        ds.text_padding_id, ds.text_epad_id = 10, 11
        user = [["multi", [0, .2], "SPEAKER_USER"],
                ["next", [.24, .4], "SPEAKER_USER"]]
        assistant = [["reply", [.24, .4], "SPEAKER_MAIN"]]
        a, c = ds._aligned_text_streams(user, assistant, 8)
        self.assertEqual(a.tolist(), [11, 1, 2, 3, 10, 10, 10, 10])
        self.assertEqual(c.tolist(), [10, 10, 11, 4, 10, 10, 10, 10])
        user[1][1][0] = .16
        with self.assertRaises(TextAlignmentCollision):
            ds._aligned_text_streams(user, assistant, 8)


class ModelTests(unittest.TestCase):
    def test_inference_training_parity_and_gradients(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            LlamaModel(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2,
            )).save_pretrained(directory)
            cfg = STTContinuationConfig(
                temporal_model=directory, text_card=34,
                text_padding_id=32, text_epad_id=33, audio_card=16,
            )
            model = SmolLM2MoshiSTTContinuation(cfg, device="cpu", dtype=torch.float32)
            model.eval()
            audio = torch.randint(0, 16, (1, 4, 8))
            observed = []
            forward = model.forward_text

            def record(prefix):
                result = forward(prefix)
                observed.append(result[1][:, :, -1].clone())
                return result

            model.forward_text = record
            a, c = force_audio_sample_text(
                model, audio, cfg, temperature=0, top_k=0, greedy=True,
            )
            model.forward_text = forward
            self.assertEqual((len(a), len(c)), (8, 8))
            streams = torch.full((1, 6, 8 + max(cfg.delays)), -1, dtype=torch.long)
            streams[:, :2] = cfg.text_padding_id
            streams[0, 0, :8] = torch.tensor(a)
            streams[0, 1, :8] = torch.tensor(c)
            streams[:, 2:, :8] = audio
            logits, mask = model.forward_text_training(streams)
            for channel in range(2):
                for t in range(8):
                    self.assertTrue(mask[0, channel, t])
                    torch.testing.assert_close(
                        logits[0, channel, t],
                        observed[t + cfg.delays[channel]][0, channel],
                        atol=1e-5, rtol=1e-5,
                    )
            loss, before = masked_text_continuation_loss(logits, mask, streams)
            changed = streams.clone()
            changed[:, 1] = (changed[:, 1] + 1) % cfg.text_card
            _, after = masked_text_continuation_loss(logits, mask, changed)
            torch.testing.assert_close(before["loss/asr"], after["loss/asr"])
            loss.backward()
            for head in (model.text_linear, model.continuation_text_linear):
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            checkpoint = Path(directory) / "roundtrip.pt"
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)


if __name__ == "__main__":
    unittest.main()
