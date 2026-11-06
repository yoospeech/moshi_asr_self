# Moshi STT Continuation

> **Development status:** experimental research prototype under active development.

User-only streaming STT-continuation experiment based on Moshi's delayed-stream
modeling architecture. LFM3-1.7B or SmolLM2 can be used as the Temporal
Transformer, while frozen Mimi converts user speech into one semantic and three
acoustic token streams.

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

- LFM3-1.7B Temporal Transformer (recommended; SmolLM2 remains supported)
- four online-extracted Mimi codebooks
- 12.5 Hz PAD/EPAD text timeline from word timestamps
- stream order `[assistant-text, user-text, user-semantic, user-acoustic-1, user-acoustic-2, user-acoustic-3]`
- Moshi-derived ASR delay pattern `[0, 1, 0, 1, 1, 1]`
- assistant text loss plus joint user text/audio losses

Assistant inner-monologue and user semantic audio use delay 0. User text is
delayed by one frame (80 ms), so text[t] can see semantic audio[t]. It cannot
yet see acoustic audio[t], which is also delayed by one frame. The acoustic codebooks
trail the semantic codebook by one frame, matching Moshi Equation 6 and the
released Moshi checkpoint configuration for acoustic delay `tau = 1`.

At Temporal step `s`, the interleaved inputs are:

| stream | token at step `s` |
|---|---|
| assistant text | `assistant_text[s]` |
| user text | `user_text[s-1]` |
| user semantic | `semantic[s]` |
| user acoustic 1 | `acoustic_1[s-1]` |
| user acoustic 2 | `acoustic_2[s-1]` |
| user acoustic 3 | `acoustic_3[s-1]` |

Consequently, `user_text[t]` is predicted at Temporal step `t+1`, after
`semantic[t]` has entered the causal history.

Only the user waveform is supplied to Mimi. Text uses two independent streams
on the same 12.5 Hz clock. Stream 0 contains aligned assistant text targets and
stream 1 contains aligned user ASR targets. Overlapping text no
longer competes for one frame. There is no assistant audio stream.

Each text branch sees only user audio and its own text history. Branches are
stacked along the batch dimension and processed with shared Transformer weights;
they never attend to each other's text. Gradients still update shared weights.
Assistant text keeps delay 0; user text uses delay 1 for audio-conditioned ASR.
This sets stream timing without imposing a speaker order.
Temporal compute and activation memory grow roughly twofold versus mixed context.
New checkpoints record `text_context_mode: independent`; the inference CLI rejects
older mixed-context configs to avoid silently changing their behavior. Train into
a new output directory for this architecture.

Each text channel independently follows Moshi's original PAD/EPAD rule. Every
word starts at its timestamped frame, with EPAD in the preceding frame unless
that would overwrite a lexical token. A word beginning at frame zero instead
uses EPAD at frame zero and shifts its lexical pieces by one frame. No speaker
BEGIN/END tokens are used: channel 0 is assistant text and channel 1 is user ASR.

Training optimizes assistant text, user text, and user Mimi audio jointly. The
native Moshi Depth Transformer predicts the four user codebooks autoregressively
within each frame from the user Temporal hidden: semantic first, followed by
three acoustic codebooks. At inference, the observed user Mimi streams are
force-fed while user and assistant text are sampled independently; generated
audio logits do not replace the observed audio.

This is a joint text/audio training architecture, not the earlier text-only
variant. Start a new training run and output directory after this change.

Single-text checkpoints and checkpoints with speaker BEGIN/END tokens are
also incompatible with the two-head, PAD/EPAD-only vocabulary.

## Setup

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_asr_self.git
cd moshi_asr_self
pip install -r requirements.txt
pip install -e thirdparty/moshi/moshi
```

Passing `--model LiquidAI/LFM3-1.7B` downloads the model and tokenizer into
`models/LFM3-1.7B` when they are not already available. Change the parent
directory with `--model-download-root`. In DDP, rank 0 downloads the files and
the other ranks wait before loading them. LFM3 custom model code is trusted
automatically. A local model directory can be passed instead. If
`--mimi-checkpoint` is omitted, the Mimi checkpoint is resolved from
`kyutai/moshiko-pytorch-bf16`.

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

- duration between 5 and 200 seconds;
- at least 5 aligned words;
- 0.5 to 2.5 aligned words per second;
- valid word timestamps;
- target-speech coverage between 5% and 80%.
- no same-channel lexical BPE collision at 12.5 Hz.

## Training

Verify the complete Temporal, two-text-head, Depth Transformer, loss, backward,
and optimizer path before a full run:

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
train_stt_continuation.py \
  --model LiquidAI/LFM3-1.7B \
  --model-download-root models \
  --trust-remote-code \
  --smoke-test \
  --smoke-steps 1 \
  --text-delay 0.08 \
  --acoustic-delay 1 \
  --batch-size 1 \
  --output /tmp/moshi_stt_smoke
```

For DDP on two available GPUs, launch one training process per GPU. Each rank
loads its own trainable model and frozen Mimi on its local GPU. Omit
`--mimi-device` in DDP mode:

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
-m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_stt_continuation.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.user_hyp.jsonl \
  --model LiquidAI/LFM3-1.7B \
  --model-download-root models \
  --trust-remote-code \
  --num-codebooks 4 \
  --text-delay 0.08 \
  --continuation-delay-steps 0 \
  --acoustic-delay 1 \
  --depformer-dim 1024 \
  --depformer-heads 16 \
  --depformer-layers 6 \
  --depformer-ffn-dim 4096 \
  --audio-loss-weight 1.0 \
  --semantic-loss-weight 100 \
  --batch-size 2 \
  --grad-accum 2 \
  --steps 100000 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --min-duration 5 \
  --max-duration 200 \
  --min-words 5 \
  --min-words-per-second 0.5 \
  --max-words-per-second 2.5 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 20 \
  --log-every 10 \
  --save-every 500 \
  --print-model \
  --output runs/lfm3_1_7b_stt_continuation_delay1
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
Both streams are generated from user audio and their own previously generated
text only; neither stream receives the other stream's text. Use a checkpoint
trained with the independent-context configuration described above.

```bash
cd /home/speech/Documents/workspace/moshi_stt_self

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=thirdparty/moshi/moshi \
/data/util/anaconda3/envs/duplexomni/bin/python \
infer_stt_continuation.py \
  --checkpoint runs/lfm3_1_7b_stt_continuation_delay1/checkpoint_00100000.pt \
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
checkpoint step, decoding settings, readable `transcript` and `turns` fields,
and frame/time/token events. `transcript.user_asr` and
`transcript.assistant_continuation` are the concise results; `channels.*.events`
is the detailed PAD/EPAD/token timeline. Mono input automatically uses channel 0.

See [STT_CONTINUATION_INFERENCE.md](STT_CONTINUATION_INFERENCE.md) for greedy
decoding, channel selection, output format, device options, and current
streaming limitations.
