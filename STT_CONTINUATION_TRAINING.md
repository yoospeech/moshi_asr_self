# User-only STT continuation training

The training path is:

`two aligned text streams + user Mimi streams -> SmolLM2 -> two text heads`

Moshi's delayed-stream transformation and masks are retained, while the audio
Depformer and audio-token prediction objective are disabled. The Temporal
Transformer is replaced with the local SmolLM2 checkpoint.

## Delay pattern

The physical stream order is:

```text
[asr-text, continuation-text, semantic, acoustic-1, acoustic-2, acoustic-3]
```

The delay pattern is:

```text
[25, 26, 0, 2, 2, 2]
```

At 12.5 Hz, ASR text is delayed by 25 frames, or 2 seconds, relative to user
semantic audio. Continuation text is delayed one additional pattern step, so
continuation frame `t` can condition on ASR frame `t`. Acoustic streams are
delayed by two frames relative to semantic audio. The online stream builder
appends enough PAD frames to preserve both delayed text targets.

The objective is the mean of ASR text CE and continuation text CE. Mimi audio
tokens are conditioning inputs, not prediction targets. At inference, user
Mimi tokens are forced and both text streams are sampled.

The training manifest is `dailytalk.user_hyp.jsonl`, produced by
`scripts/transcribe_user_hyp.py`. ASR targets come directly from each row's
`user_hyp.words`; `data_stereo/user_N.json` is only a compatibility fallback.
Continuation targets use the timestamped assistant `alignments` in
`data_stereo/N.json` as the assistant inner-monologue stream. Both channels
independently use the original Moshi PAD/EPAD construction; there are no
speaker BEGIN/END tokens.

## Validation and training

Validate the forward/backward path first:

```bash
python3 train_stt_continuation.py --smoke-test
```

Then train on aligned data:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_stt_continuation.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.user_hyp.jsonl \
  --num-codebooks 4 \
  --text-delay 2.0 \
  --continuation-delay-steps 1 \
  --acoustic-delay 2 \
  --batch-size 2 \
  --grad-accum 2 \
  --steps 100000 \
  --output runs/stt_continuation
```

TensorBoard logs include separate ASR/continuation CE and non-PAD accuracy,
validation metrics, and both optimizer learning rates:

```bash
tensorboard --logdir runs/stt_continuation/tensorboard
```

The manifest is deterministically split into train and validation sets. The
exact indices and seed are saved in `OUTPUT/split.json`.

DDP uses one model and one frozen Mimi per GPU. Do not pass `--mimi-device`;
the local rank selects the device. Batch size is per rank, so the example has
an effective batch of 8. Steps count microbatches per rank, as in the previous
trainer; accumulation 2 means one optimizer update every two steps. A final
partial accumulation group is also applied. Only rank 0 saves unwrapped model
weights (compatible with the existing inference CLI) and runs validation.
Training metrics are averaged across ranks; validation uses rank 0's full
validation loader, limited by `--valid-batches`.

Test DDP before a full run (uses synthetic tokens, without loading Mimi):

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc_per_node=2 train_stt_continuation.py \
  --model models/SmolLM2-135M --smoke-test --smoke-steps 5 --grad-accum 2 \
  --output runs/ddp_smoke
```

The smoke test checks that all ranks finish with identical model parameters.
