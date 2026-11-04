# SmolLM2 Moshi STT Continuation

> **Development status:** experimental research prototype under active development.

User-only streaming STT-continuation experiment based on Moshi's delayed-stream
modeling architecture. SmolLM2-135M is the Temporal Transformer, while frozen
Mimi converts user speech into one semantic and three acoustic token streams.

## Architecture

- SmolLM2-135M Temporal Transformer
- Moshi Depth Transformer with per-RQ parameters
- four online-extracted Mimi codebooks
- 12.5 Hz PAD/EPAD text timeline from word timestamps
- stream order `[shared-text, user-semantic, user-acoustic-1, user-acoustic-2, user-acoustic-3]`
- delay pattern `[25, 0, 2, 2, 2]`
- semantic-token loss weight 100, acoustic-token weights 1

User semantic audio leads text by 25 frames, or 2 seconds at 12.5 Hz. The
acoustic codebooks trail the semantic codebook by two frames. This pattern is
the four-codebook form derived from Moshi Appendix C, Figure 8.

Only the user waveform is supplied to Mimi. The single shared text stream
contains both sides in chronological speaker runs:
`USER_BEGIN ... user text ... USER_END`, followed by
`ASSISTANT_BEGIN ... assistant text ... ASSISTANT_END` (and repeated for later
turns). Within each region, Moshi's original `EPAD -> word pieces -> PAD`
alignment is unchanged. There is no assistant audio stream.

At a region boundary, the last PAD position becomes EPAD and the first word
follows the role marker directly: `PAD ... EPAD -> USER_BEGIN -> first word`
or `PAD ... EPAD -> ASSISTANT_BEGIN -> first word`. A second EPAD is not added
between BEGIN and that first word.

Training retains Moshi's joint text/audio objective. At streaming inference,
all four Mimi audio streams must be force-fed from user audio and only the
delayed text stream is sampled.

## Setup

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_asr_self.git
cd moshi_asr_self
pip install -r requirements.txt
pip install -e thirdparty/moshi/moshi
```

Place SmolLM2 at `models/SmolLM2-135M`. If `--mimi-checkpoint` is omitted, the
Mimi checkpoint is resolved from `kyutai/moshiko-pytorch-bf16`.

## Data

The trainer reads the DailyTalkContiguous manifest and adjacent word-alignment
JSON files. The selected user waveform is resampled to 24 kHz, and frozen Mimi
extracts four audio codebooks online.

Generate `user_hyp` text and word timestamps with faster-whisper Turbo:

```bash
python3 scripts/transcribe_user_hyp.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --output /data/db/daily-talk-contiguous/dailytalk.user_hyp.jsonl \
  --user-channel 1 \
  --model turbo \
  --device cuda \
  --device-index 0 \
  --compute-type float16
```

The input manifest is never overwritten. Per-item cache files allow an
interrupted run to resume without transcribing completed audio again.
For every completed row, the script also writes `data_stereo/user_N.json` in
the exact existing sidecar format: a single `alignments` key containing
`[word, [start, end], "SPEAKER_USER"]` entries. It does not modify the existing
assistant-side `data_stereo/N.json`.

Default filters retain conversations with:

- duration between 5 and 45 seconds;
- at least 5 aligned words;
- 0.5 to 2.5 aligned words per second;
- valid word timestamps;
- target-speech coverage between 5% and 80%.

## Training

For two GPUs, keep the trainable model on visible GPU 0 and frozen Mimi on
visible GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_stt_continuation.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --model models/SmolLM2-135M \
  --mimi-device cuda:1 \
  --num-codebooks 4 \
  --text-delay 2.0 \
  --acoustic-delay 2 \
  --depformer-dim 512 \
  --depformer-heads 8 \
  --depformer-layers 4 \
  --depformer-ffn-dim 2048 \
  --batch-size 2 \
  --grad-accum 2 \
  --steps 100000 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 20 \
  --log-every 10 \
  --save-every 500 \
  --output runs/stt_continuation
```

Use `CUDA_VISIBLE_DEVICES=0` with `--mimi-device cuda:0` when running both
models on one GPU.

More details are in
[STT_CONTINUATION_TRAINING.md](STT_CONTINUATION_TRAINING.md).

## Reference inference

The reference inference path force-feeds every Mimi codebook from the selected
user waveform and samples only the shared text stream. Extra silence keeps the
audio clock running while the assistant-text tail is generated.

```bash
python3 infer_stt_continuation.py \
  --checkpoint runs/stt_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --tail-seconds 10 \
  --output inference_outputs/0.json
```

This implementation recomputes the SmolLM2 prefix at each frame. It is a
correctness-oriented reference for forced-audio generation, not yet a
latency-optimized live streaming loop.
