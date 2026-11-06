# Audio-to-Text Continuation Inference

Inference selects the user waveform, encodes it with frozen Mimi, force-feeds
all four observed audio codebooks, and generates one assistant text stream.
There is no ASR output and no generated assistant audio.

## Sampled decoding

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/lfm3_1_7b_moshi_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --device cuda \
  --mimi-device cuda:0 \
  --tail-seconds 10 \
  --text-temperature 0.7 \
  --text-top-k 25 \
  --output inference_outputs/0.json
```

The adjacent `config.json` is loaded automatically. Use `--model` only when
the checkpoint's stored temporal-model path must be overridden.

## Greedy decoding

Add `--greedy` for deterministic argmax decoding:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/lfm3_1_7b_moshi_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --device cuda \
  --mimi-device cuda:0 \
  --tail-seconds 10 \
  --greedy \
  --output inference_outputs/0.greedy.json
```

## Channel and silence tail

DailyTalk stereo user audio is channel 1. For a mono user WAV, select channel
0. If mono audio is passed with channel 1, the CLI warns and uses channel 0.

The silence tail is encoded by Mimi and keeps the 12.5 Hz clock running after
the original WAV ends. Increase it if the assistant response is truncated.

## Output

The readable result is `transcript.assistant_continuation`. For frame-level
inspection, `channels.continuation.events` records each generated PAD, EPAD, or
lexical token with frame index and time. `turns` contains one assistant entry
spanning its first and last lexical events.

## Compatibility and speed

Inference requires `text_context_mode: "moshi"` and delay pattern
`[0, 0, 1, 1, 1]`. Dual-text/ASR checkpoints are rejected.

The reference loop recomputes the complete prefix for every frame. It does not
yet use a streaming KV cache and can be very slow with LFM3-1.7B.
