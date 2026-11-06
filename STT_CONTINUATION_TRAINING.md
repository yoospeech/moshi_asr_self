# User-only STT continuation training

The training path is:

`two aligned text streams + user Mimi streams -> SmolLM2 -> two text heads + Moshi Depth Transformer`

Moshi's delayed-stream transformation, masks, and audio Depth Transformer are
retained. The Temporal Transformer uses LFM3-1.7B (or another compatible
Hugging Face base model), and user text is added as one extra Temporal channel.

## Delay pattern

The physical stream order is:

```text
[assistant-text, user-text, user-semantic, user-acoustic-1, user-acoustic-2, user-acoustic-3]
```

The delay pattern is:

```text
[0, 1, 0, 1, 1, 1]
```

Inner-monologue and semantic audio use delay 0. User text uses delay 1 (80 ms)
to condition on its corresponding semantic audio frame. Each branch sees
only user audio and its own past text; Transformer weights are shared but
attention histories are separate batch rows. Acoustic streams are
delayed by one frame relative to semantic audio. The online stream builder
appends enough PAD frames to preserve both delayed text targets.

The objective combines the mean of assistant/user text CE with the weighted
user-audio CE. Moshi's native Depth Transformer predicts semantic then acoustic
codebooks from the user branch hidden and preceding within-frame codebooks.
At inference, user Mimi tokens are force-fed and only both text streams are
sampled.

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
  --model LiquidAI/LFM3-1.7B \
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
  --output runs/lfm3_1_7b_stt_continuation_delay1
```

TensorBoard logs include separate user/assistant CE and non-PAD accuracy,
user-audio Depth losses, validation metrics, and both optimizer learning rates:

```bash
tensorboard --logdir runs/lfm3_1_7b_stt_continuation_delay1/tensorboard
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
  --model LiquidAI/LFM3-1.7B --trust-remote-code \
  --smoke-test --smoke-steps 5 --grad-accum 2 \
  --output runs/lfm3_1_7b_ddp_smoke
```

The smoke test checks that all ranks finish with identical model parameters.
