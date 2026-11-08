"""Offline regression checks: python -m unittest discover -s tests -v."""

import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel

from stt_continuation.dataset import DailyTalkDataset, TextAlignmentCollision
from stt_continuation.model import (
    STTContinuationConfig, SmolLM2MoshiSTTContinuation, masked_multistream_loss,
)
from stt_continuation.inference import force_audio_sample_text, load_config
from stt_continuation.train import acoustic_prompt_mask, resolve_model_source


class AlignmentTests(unittest.TestCase):
    def test_manifest_target_and_audio_channel(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" assistant": [2]}[text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0.json").write_text(json.dumps({"alignments": [
                ["assistant", [.8, 1.0], "SPEAKER_MAIN"]
            ]}))
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "path": "0.wav", "duration": 2, "user_channel": 0,
            }) + "\n")
            ds = DailyTalkDataset(
                manifest, Tokenizer(), text_padding_id=10, text_epad_id=11,
                min_duration=0, min_words=1, min_words_per_second=0,
                min_speech_coverage=0, max_speech_coverage=1,
            )
            audio = torch.stack([torch.ones(48000), torch.zeros(48000)]).numpy()
            with patch("stt_continuation.dataset.sphn.read", return_value=(audio, 24000)):
                item = ds[0]
            self.assertTrue((item["waveform"] == 1).all())
            self.assertEqual(item["continuation_tokens"][10].item(), 2)

    def test_padding_and_collision(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {" multi": [1, 2], " next": [3]}[text]

        ds = object.__new__(DailyTalkDataset)
        ds.tokenizer = Tokenizer()
        ds.frame_rate = 12.5
        ds.text_padding_id, ds.text_epad_id = 10, 11
        user = [["multi", [0, .2], "SPEAKER_USER"],
                ["next", [.24, .4], "SPEAKER_USER"]]
        a = ds._aligned_text_stream(user, 8)
        self.assertEqual(a.tolist(), [11, 1, 2, 3, 10, 10, 10, 10])
        user[1][1][0] = .16
        with self.assertRaises(TextAlignmentCollision):
            ds._aligned_text_stream(user, 8)


class ModelTests(unittest.TestCase):
    def test_half_batch_is_text_only(self):
        torch.manual_seed(7)
        mask = acoustic_prompt_mask(4, 0.5, device="cpu")
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual((~mask).sum().item(), 2)

    def test_models_directory_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            path, repository = resolve_model_source(
                str(root / "SmolLM2-135M"), str(root)
            )
            self.assertEqual(path, root / "SmolLM2-135M")
            self.assertEqual(repository, "HuggingFaceTB/SmolLM2-135M")
            path.mkdir(parents=True)
            existing, repository = resolve_model_source(str(path), str(root))
            self.assertEqual(existing, path)
            self.assertIsNone(repository)

    def test_inference_rejects_incompatible_delay_pattern(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            base = {
                "text_context_mode": "moshi",
                "text_delay_seconds": 0.08,
                "acoustic_delay_steps": 1,
            }
            config_path.write_text(json.dumps(base))
            args = SimpleNamespace(
                checkpoint=str(Path(directory) / "checkpoint.pt"),
                config=str(config_path),
                model=None,
            )
            self.assertEqual(load_config(args).delays, [0, 0, 1, 1, 1])
            base["acoustic_delay_steps"] = 2
            config_path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "expected"):
                load_config(args)

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
                depformer_dim=32, depformer_heads=4, depformer_layers=2,
                depformer_ffn_dim=64,
            )
            model = SmolLM2MoshiSTTContinuation(cfg, device="cpu", dtype=torch.float32)
            model.eval()
            self.assertEqual(cfg.delays, [0, 0, 1, 1, 1])
            causal_probe = torch.randint(0, 16, (1, 5, 12))
            with torch.no_grad():
                reference = model.forward_training(causal_probe)[0]
                changed = causal_probe.clone()
                changed[:, 1, 4] = (changed[:, 1, 4] + 1) % cfg.audio_card
                current = model.forward_training(changed)[0]
                self.assertTrue(torch.isfinite(current[:, 0, 4]).all())
                changed = causal_probe.clone()
                changed[:, 1:, 5:] = (changed[:, 1:, 5:] + 1) % cfg.audio_card
                future = model.forward_training(changed)[0]
            audio = torch.randint(0, 16, (1, 4, 8))
            observed = []
            forward = model.forward_text

            def record(prefix):
                result = forward(prefix)
                observed.append(result[1][:, :, -1].clone())
                return result

            model.forward_text = record
            a = force_audio_sample_text(
                model, audio, cfg, temperature=0, top_k=0, greedy=True,
            )
            model.forward_text = forward
            self.assertEqual(len(a), 8)
            streams = torch.full((1, 5, 8 + max(cfg.delays)), -1, dtype=torch.long)
            streams[:, 0] = cfg.text_padding_id
            streams[0, 0, :8] = torch.tensor(a)
            streams[:, 1:, :8] = audio
            logits, mask, audio_logits, audio_mask = model.forward_training(streams)
            for t in range(8):
                self.assertTrue(mask[0, 0, t])
                torch.testing.assert_close(logits[0, 0, t], observed[t][0, 0], atol=1e-5, rtol=1e-5)
            self.assertEqual(audio_logits.shape, (1, 4, streams.shape[-1], 16))
            self.assertTrue(audio_mask[:, :, :8].all())
            text_logits, text_mask, no_audio_logits, no_audio_mask = (
                model.forward_training(streams, predict_audio=False)
            )
            torch.testing.assert_close(text_logits, logits)
            torch.testing.assert_close(text_mask, mask)
            self.assertIsNone(no_audio_logits)
            self.assertIsNone(no_audio_mask)
            loss, before = masked_multistream_loss(
                logits, mask, streams, audio_logits=audio_logits, audio_mask=audio_mask,
            )
            changed = streams.clone()
            changed[:, 0] = (changed[:, 0] + 1) % cfg.text_card
            _, after = masked_multistream_loss(logits, mask, changed)
            self.assertNotEqual(before["loss/assistant"].item(), after["loss/assistant"].item())
            loss.backward()
            for head in (model.text_linear, model.linears[0]):
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            checkpoint = Path(directory) / "roundtrip.pt"
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)


if __name__ == "__main__":
    unittest.main()
