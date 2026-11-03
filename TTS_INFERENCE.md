# Inference

> **Development status:** Inference is experimental and under active
> development. Generated speech is not yet consistently intelligible or
> production-ready.

## Raw-text batch inference

`infer_tts.py` synthesizes one or more utterances without the streaming cache.
It recomputes the full temporal prefix at every frame, samples RQ0 through RQ3,
undoes the training delay pattern, and decodes the result with Mimi:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 infer_tts.py \
  --checkpoint runs/assistant_tts_batch3/checkpoint_00002500.pt \
  --config runs/assistant_tts_batch3/config.json \
  --model models/SmolLM2-135M \
  --mimi-device cuda:1 \
  --text "Hello, it is nice to meet you." \
  --text "The weather is especially pleasant today." \
  --output-dir inference_outputs
```

Use `--text-file prompts.txt` to synthesize one non-empty line per utterance.
Raw text has no word timestamps, so this path approximates the training
PAD/EPAD timeline. `--frames-per-word` controls the assumed speaking rate and
`--tail-seconds` controls silence after the final word.

## Where time alignment comes from

The manifest itself contains only a waveform path and duration:

```json
{"path": "data_stereo/914.wav", "duration": 10.92}
```

The word alignment is loaded from the adjacent JSON sidecar with the same
stem, for example `data_stereo/914.json`:

```json
{
  "alignments": [
    ["Is", [0.00, 0.16], "SPEAKER_MAIN"],
    ["that", [0.16, 0.32], "SPEAKER_MAIN"]
  ]
}
```

`tts_dataset.py` converts these entries into the 12.5 Hz text stream used for
training:

1. Create one text frame every 80 ms and initialize the stream with PAD.
2. Keep only the configured main-speaker words.
3. Convert each word start time with `round(start_seconds * 12.5)`.
4. Tokenize `" " + word` with the model tokenizer.
5. Put EPAD one frame before the word start, then place its token pieces from
   the start frame onward.
6. Pair the resulting text stream with Mimi audio codes. The model applies
   delay pattern `[0, 25, 27, 27, 27]`: text has no delay, semantic RQ0 is
   delayed by 25 frames (2 seconds), and acoustic RQ1-RQ3 by 27 frames.

The current dataset implementation should be treated carefully when a first
word starts exactly at `0.00` seconds: EPAD and the first token target the same
frame, so the first token can be skipped. This is a known development issue.

## Exact time-aligned training-sample inference

`infer_aligned_sample.py` reconstructs the exact PAD/EPAD text timeline used
during training from the sidecar word timestamps. It generates all Mimi audio
codebooks autoregressively. The reference waveform and its audio tokens are
not supplied to the model; the left/main reference channel is only saved next
to the generated file for listening comparison.

```bash
CUDA_VISIBLE_DEVICES=0 python3 infer_aligned_sample.py \
  --checkpoint runs/assistant_tts_batch3/checkpoint_00100000.pt \
  --manifest /home/ysw/Documents/eee/moshi-finetune/daily-talk-contiguous/dailytalk.jsonl \
  --sample-path data_stereo/914.wav \
  --sample-path data_stereo/197.wav \
  --sample-path data_stereo/1231.wav \
  --output-dir samples
```

This experiment separates text-to-audio learning from the speaking-rate
heuristic in `infer_tts.py`: it supplies ground-truth word timing but does not
teacher-force any ground-truth audio. The checked-in examples in
[`samples/`](samples/) use audio temperature `0.6`, top-k `100`, seed `42`,
and the delay pattern above.
