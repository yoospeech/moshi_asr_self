# Moshi Audio-to-Text Continuation

> Experimental research prototype.

This project takes the user audio channel as input and predicts only the
assistant continuation text. User ASR/user text is not an input, target, loss,
or inference output.

```text
stereo WAV --channel 1--> frozen Mimi --> user audio tokens
                                             |
assistant PAD/EPAD timeline -----------------+--> Temporal Transformer
                                                    |-- assistant text head
                                                    `-- Moshi Depth Transformer (training only)
```

## Streams and timing

```text
streams: [assistant_text, user_semantic, user_acoustic_1, user_acoustic_2, user_acoustic_3]
delays:  [0,              0,             1,               1,               1]
```

Assistant text and user semantic audio are placed at the current 12.5 Hz
frame. The three acoustic codebooks lag by one frame (80 ms), following
Moshi's semantic/acoustic interleaving. There is no user text stream and no
assistant audio stream.

Assistant word timestamps are converted to the original Moshi PAD/EPAD
timeline. EPAD is placed immediately before a word unless that would overwrite
a lexical token. No speaker BEGIN/END tokens are used.

Training jointly optimizes assistant continuation text cross-entropy and user
semantic/acoustic codebook losses through Moshi's Depth Transformer. Inference
force-feeds observed user Mimi tokens and samples only assistant text; Depth
audio predictions are not fed back.

## Repository layout

```text
stt_continuation/model.py       model, interleaving, Depth Transformer, losses
stt_continuation/dataset.py     user audio and assistant PAD/EPAD target
stt_continuation/train.py       single-GPU/DDP training and validation
stt_continuation/inference.py   force-fed user-audio inference
train_stt_continuation.py       training CLI
infer_stt_continuation.py       inference CLI
tests/test_continuation.py      offline regression tests
```

## Setup

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_asr_self.git
cd moshi_asr_self
pip install -r requirements.txt
pip install -e thirdparty/moshi/moshi
```

`--model LiquidAI/LFM3-1.7B` downloads the model to `models/LFM3-1.7B`
when absent. In DDP, rank 0 downloads it and the other ranks wait. LFM3 remote
model code is trusted automatically; the explicit flag below documents that
requirement. If `--mimi-checkpoint` is omitted, Mimi is resolved from the
configured Kyutai repository.

## Data

Use the original manifest:

```text
/data/db/daily-talk-contiguous/dailytalk.jsonl
```

For a row whose audio path is `data_stereo/406.wav`, user audio is read from
WAV channel 1 by default and assistant continuation timestamps are read from
`data_stereo/406.json`. `user_hyp`, `user_406.json`, and Whisper transcripts
are not used. Relative audio paths are resolved from the manifest directory.

Default filtering keeps samples with duration 5–200 seconds, at least 5
assistant words, 0.5–2.5 assistant words/second, valid timestamps, 5–80%
assistant timestamp coverage, and no lexical BPE collision at 12.5 Hz.

## Smoke test

This checks model construction, Temporal and Depth forward passes, losses,
backward, and optimizer updates without loading a dataset or Mimi:

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
  --acoustic-delay 1 \
  --batch-size 1 \
  --output /tmp/moshi_continuation_smoke
```

## Two-GPU DDP training

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
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --model LiquidAI/LFM3-1.7B \
  --model-download-root models \
  --trust-remote-code \
  --num-codebooks 4 \
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
  --output runs/lfm3_1_7b_moshi_continuation
```

Do not pass `--mimi-device` in DDP; every rank places frozen Mimi and its model
on the local GPU. Batch size is per rank, so the effective batch above is
`2 GPUs × batch 2 × accumulation 2 = 8`. `--steps` counts microbatches. Only
rank 0 validates and writes checkpoints.

For one GPU, run the script directly without `torch.distributed.run`. A separate
Mimi device can then be selected with `--mimi-device cuda:1`.

### CLI metrics

```text
loss                    total optimized loss
assistant_text          assistant continuation cross-entropy
audio                   weighted mean of user Mimi codebook losses
assistant_np_accuracy   assistant lexical/EPAD accuracy excluding PAD
assistant_all_accuracy  assistant accuracy including PAD frames
audio_acc               mean Mimi codebook accuracy (validation)
```

PAD occupies many frames, so `assistant_all_accuracy` is usually higher. Use
`assistant_np_accuracy` to judge whether actual continuation tokens improve.

```bash
tensorboard --logdir runs/lfm3_1_7b_moshi_continuation/tensorboard
```

## Inference

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

Add `--greedy` for deterministic argmax decoding; temperature and top-k are
then ignored. For mono user audio use `--user-channel 0`. `--tail-seconds 10`
appends silence before Mimi encoding so the audio clock has time to emit a
response.

The output JSON contains only assistant continuation data:

```json
{
  "stream_order": ["assistant_text", "user_semantic", "user_acoustic_1", "user_acoustic_2", "user_acoustic_3"],
  "delay_pattern": [0, 0, 1, 1, 1],
  "continuation_text": "Assistant response",
  "transcript": {"assistant_continuation": "Assistant response"},
  "channels": {"continuation": {"text": "Assistant response", "events": []}}
}
```

## Compatibility and limitations

New checkpoints record `text_context_mode: "moshi"`. Older dual-text/ASR
checkpoints are rejected because their stream count, heads, and delay pattern
are incompatible.

Reference inference recomputes the full prefix at every 12.5 Hz frame. It is
correct for this interleaving but slow; KV-cache streaming is not implemented.

Run regression tests with:

```bash
/data/util/anaconda3/envs/duplexomni/bin/python \
  -m unittest discover -s tests -v
```

See [training details](STT_CONTINUATION_TRAINING.md) and
[inference details](STT_CONTINUATION_INFERENCE.md).
