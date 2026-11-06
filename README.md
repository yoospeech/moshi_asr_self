# SmolLM2 Moshi STT Continuation

> **Development status:** experimental research prototype under active development.

User-only streaming STT-continuation experiment based on Moshi's delayed-stream
modeling architecture. SmolLM2-135M is the Temporal Transformer, while frozen
Mimi converts user speech into one semantic and three acoustic token streams.

## Repository layout

```text
stt_continuation/
  model.py          # temporal model, two text heads, losses
  dataset.py        # user_hyp and assistant alignments, PAD/EPAD
  train.py          # training and validation
  inference.py      # force-fed user audio, two text outputs
  distributed.py    # DDP process setup and training objective
scripts/
  transcribe_user_hyp.py
tests/
  test_continuation.py
train_stt_continuation.py  # CLI entry point
infer_stt_continuation.py  # CLI entry point
```

The root CLI commands below remain supported. Offline regression checks run
with `python -m unittest discover -s tests -v`.

## Model architecture

- SmolLM2-135M Temporal Transformer
- four online-extracted Mimi codebooks
- 12.5 Hz PAD/EPAD text timeline from word timestamps
- stream order `[asr-text, continuation-text, user-semantic, user-acoustic-1, user-acoustic-2, user-acoustic-3]`
- delay pattern `[25, 26, 0, 2, 2, 2]`
- independent ASR and continuation text losses conditioned on Mimi audio tokens

User semantic audio leads text by 25 frames, or 2 seconds at 12.5 Hz. The
acoustic codebooks trail the semantic codebook by two frames. This pattern is
the four-codebook form derived from Moshi Appendix C, Figure 8.

Only the user waveform is supplied to Mimi. Text uses two independent streams
on the same 12.5 Hz clock. Stream 0 contains aligned user ASR targets and
stream 1 contains aligned assistant continuation targets. Overlapping text no
longer competes for one frame. There is no assistant audio stream.

Each text channel independently follows Moshi's original PAD/EPAD rule. Every
word starts at its timestamped frame, with EPAD in the preceding frame unless
that would overwrite a lexical token. A word beginning at frame zero instead
uses EPAD at frame zero and shifts its lexical pieces by one frame. No speaker
BEGIN/END tokens are used: channel 0 is user ASR and channel 1 is assistant
continuation (the inner-monologue target).

Training uses all four Mimi audio streams as conditioning inputs and optimizes
the two delayed text losses. There is no audio-token prediction objective or
Depformer. At inference, Mimi streams are force-fed while ASR and continuation
text are sampled autoregressively.

Checkpoints produced by the earlier joint audio/text objective contain
Depformer parameters and are not compatible with this text-only architecture;
start a new training run and output directory after this change.

Single-text checkpoints and checkpoints with speaker BEGIN/END tokens are
also incompatible with the two-head, PAD/EPAD-only vocabulary.

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

The trainer takes `dailytalk.user_hyp.jsonl` produced by the transcription
script. User ASR targets come directly from each row's `user_hyp.words`, while
continuation/inner-monologue targets come from the timestamped `alignments` in
`data_stereo/N.json`. The generated `data_stereo/user_N.json` files are retained
as a compatibility fallback. The selected user waveform is resampled to 24 kHz,
and frozen Mimi extracts four audio codebooks online.

Generate `user_hyp` text and word timestamps with whisper-timestamped Medium:

```bash
python3 scripts/transcribe_user_hyp.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --output /data/db/daily-talk-contiguous/dailytalk.user_hyp.jsonl \
  --user-channel 1 \
  --model medium \
  --device auto
```

Audio longer than 10 seconds uses Auditok VAD by default. Pass `--no-vad` to
disable it. Word alignment is provided by whisper-timestamped and does not use
OpenAI Whisper's Triton word-timestamp kernel.

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
- no same-channel lexical BPE collision at 12.5 Hz.

## Training

For DDP on two available GPUs, launch one training process per GPU. Each rank
loads its own trainable model and frozen Mimi on its local GPU. Omit
`--mimi-device` in DDP mode:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_stt_continuation.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.user_hyp.jsonl \
  --model models/SmolLM2-135M \
  --num-codebooks 4 \
  --text-delay 2.0 \
  --continuation-delay-steps 1 \
  --acoustic-delay 2 \
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

`--batch-size` is per rank: 2 GPUs × batch 2 × accumulation 2 gives an
effective batch of 8. `--steps` retains its existing meaning of microbatches
per rank; 100000 steps with accumulation 2 makes 50000 optimizer updates.
Only rank 0 writes checkpoints/logs and evaluates the validation set; other
ranks wait during validation. Training shards are shuffled each epoch, with
at most world_size-1 samples dropped to avoid duplicate examples across ranks.

Use only GPUs allocated to your job in `CUDA_VISIBLE_DEVICES`. For one GPU,
use `python3 train_stt_continuation.py` without torchrun. The earlier
single-process model/Mimi split still supports `--mimi-device cuda:1`.

More details are in
[STT_CONTINUATION_TRAINING.md](STT_CONTINUATION_TRAINING.md).

## Reference inference

The reference inference path force-feeds every Mimi codebook from the selected
user waveform and samples separate ASR and continuation text streams. Extra
silence keeps the audio clock running while the continuation tail is generated.

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/stt_continuation/checkpoint_00100000.pt \
  --audio /data/db/daily-talk-contiguous/data_stereo/0.wav \
  --user-channel 1 \
  --tail-seconds 10 \
  --text-temperature 0.7 \
  --text-top-k 25 \
  --device cuda \
  --mimi-device cuda:0 \
  --output inference_outputs/0.json
```

Without `--greedy`, text is sampled with `--text-temperature 0.7` and
`--text-top-k 25` by default. The output JSON contains the decoded text,
checkpoint step, decoding settings, and frame/time/token events. Mono input
automatically uses channel 0.

See [STT_CONTINUATION_INFERENCE.md](STT_CONTINUATION_INFERENCE.md) for greedy
decoding, channel selection, output format, device options, and current
streaming limitations.
