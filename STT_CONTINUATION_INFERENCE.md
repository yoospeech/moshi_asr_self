# STT Continuation Inference

This guide runs inference with a trained STT-continuation checkpoint. Mimi
encodes the selected user waveform into four audio codebooks. Those observed
audio tokens are force-fed at every frame. User and assistant text are sampled
from independent output channels. The training-only Depth Transformer audio
prediction is not sampled during inference.

Each channel conditions on user audio and its own generated text history only.
Changing ASR text does not change continuation logits, or vice versa, for fixed
weights and audio. Assistant text uses delay 0; user text uses delay 1 (80 ms)
to read the corresponding semantic audio before predicting text. Use checkpoints
whose config records `text_context_mode: independent`; older mixed-context
checkpoints require their original inference code and cannot be silently reused.

Both text channels retain Moshi's existing PAD/EPAD alignment tokens:

```text
physical channel 0 (assistant): PAD ... EPAD assistant inner-monologue text
physical channel 1 (user ASR):  PAD ... EPAD user text
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
  "asr_text": "Hello",
  "continuation_text": "Hi",
  "transcript": {
    "user_asr": "Hello",
    "assistant_continuation": "Hi"
  },
  "turns": [
    {
      "speaker": "user",
      "channel": "asr",
      "text": "Hello",
      "start_time": 2.0,
      "end_time": 3.2
    },
    {
      "speaker": "assistant",
      "channel": "continuation",
      "text": "Hi",
      "start_time": 0.0,
      "end_time": 1.2
    }
  ],
  "channels": {
    "asr": {
      "text": "Hello",
      "events": []
    },
    "continuation": {
      "text": "Hi",
      "events": []
    }
  }
}
```

Use `transcript.user_asr` and `transcript.assistant_continuation` for the
readable result. `channels.asr.events` and `channels.continuation.events` preserve PAD, EPAD,
and lexical predictions for frame-level debugging. Speaker BEGIN/END markers
are absent because stream identity carries the role: ASR is user text and
continuation is assistant inner-monologue text.
The `turns` array is a readable two-channel view with first/last lexical event
times; it does not claim to recover multiple dialogue-turn boundaries.

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
