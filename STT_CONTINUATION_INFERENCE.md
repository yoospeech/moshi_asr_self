# STT Continuation Inference

This guide runs inference with a trained STT-continuation checkpoint. Mimi
encodes the selected user waveform into four audio codebooks. Those audio
tokens are force-fed at every frame, and only the shared text channel is
sampled.

The expected text sequence contains the existing PAD/EPAD alignment tokens and
the four region markers:

```text
PAD ... EPAD USER_BEGIN user text USER_END
PAD ... EPAD ASSISTANT_BEGIN assistant text ASSISTANT_END
```

There is no assistant audio input or output.

## Sampled inference

Run the following command from the repository root:

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/stt_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --device cuda \
  --mimi-device cuda:0 \
  --tail-seconds 10 \
  --text-temperature 0.7 \
  --text-top-k 25 \
  --output inference_outputs/0.json
```

Replace `--checkpoint` with the checkpoint produced by the new training run.
The adjacent `config.json` is loaded automatically. Available checkpoints can
be listed with:

```bash
ls -lh runs/stt_continuation/checkpoint_*.pt
```

## Greedy inference

For deterministic checkpoint comparisons, use `--greedy`. Temperature and
top-k are ignored in greedy mode.

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/stt_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --device cuda \
  --mimi-device cuda:0 \
  --tail-seconds 10 \
  --greedy \
  --output inference_outputs/0.greedy.json
```

## Audio channel selection

DailyTalk stereo files use channel 1 for the user waveform:

```text
--user-channel 1
```

For an already separated mono user WAV, use channel 0:

```text
--user-channel 0
```

If a mono file is supplied with the default channel 1, the script prints a
warning and automatically reads channel 0.

## Silence tail

`--tail-seconds 10` appends ten seconds of silence before Mimi encoding. This
keeps the audio clock running after the user waveform ends so the model has
time to emit the assistant-text continuation. Increase this value if the
generated assistant region is cut off; reduce it for shorter test runs.

The appended silence is also encoded by Mimi and force-fed. Audio codebooks are
never sampled by the language model.

## Output JSON

The output contains the decoded lexical text and every predicted text event:

```json
{
  "audio": "/data/db/daily-talk-contiguous/data_stereo/0.wav",
  "user_channel": 1,
  "checkpoint_step": 100000,
  "decoding": {
    "greedy": false,
    "temperature": 0.7,
    "top_k": 25,
    "seed": 42
  },
  "decoded_text": "...",
  "turns": [
    {
      "speaker": "user",
      "text": "Hello",
      "start_frame": 25,
      "end_frame": 40,
      "start_time": 2.0,
      "end_time": 3.2,
      "complete": true
    },
    {
      "speaker": "assistant",
      "text": "Hi",
      "start_frame": 45,
      "end_frame": 60,
      "start_time": 3.6,
      "end_time": 4.8,
      "complete": true
    }
  ],
  "events": [
    {
      "frame": 0,
      "time": 0.0,
      "id": 49153,
      "token": "EPAD"
    }
  ]
}
```

`turns` parses `USER_BEGIN/USER_END` and
`ASSISTANT_BEGIN/ASSISTANT_END` into clean chronological dialogue turns.
`complete` is false when generation ends or a new BEGIN arrives before the
matching END token. `events` preserves `PAD`, `EPAD`, and all four boundary
tokens for frame-level debugging. `decoded_text` contains all lexical token IDs
decoded with the SmolLM2 tokenizer.

## Device options

- `--device auto`: use CUDA when available, otherwise CPU.
- `--device cuda`: require CUDA for the language model.
- `--device cpu`: run the language model on CPU.
- `--mimi-device cuda:0`: place frozen Mimi on visible GPU 0.

When `CUDA_VISIBLE_DEVICES=0` is set, the selected physical GPU appears inside
the process as `cuda:0`.

## Current limitation

This is a correctness-oriented reference implementation. Because the temporal
backbone is Hugging Face SmolLM2 rather than Moshi's native streaming
Transformer, the script recomputes the full text/audio prefix for each frame.
It force-feeds audio with the correct delay pattern, but it is not yet an
optimized real-time inference loop.
