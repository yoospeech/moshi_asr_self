# User-only STT continuation training

The training path is:

`aligned text + user Mimi streams -> SmolLM2 Temporal Transformer -> Moshi Depformer`

The Moshi Depth Transformer, delayed-stream transformation, masks, and
teacher-forcing path come from `thirdparty/moshi/moshi/moshi/models/lm.py`.
Only the Temporal Transformer is replaced with the local SmolLM2 checkpoint.

## Delay pattern

The physical stream order is:

```text
[text, semantic, acoustic-1, acoustic-2, acoustic-3]
```

The delay pattern is:

```text
[25, 0, 2, 2, 2]
```

At 12.5 Hz, text is delayed by 25 frames, or 2 seconds, relative to user
semantic audio. Acoustic streams are delayed by two frames relative to the
semantic stream. The online stream builder appends enough text PAD frames to
preserve the delayed text targets at the end of each item.

The joint objective is retained: text CE plus semantic-weighted audio CE. At
inference, user Mimi tokens are forced and only text is sampled.

## Validation and training

Validate the forward/backward path first:

```bash
python3 train_stt_continuation.py --smoke-test
```

Then train on aligned data:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_stt_continuation.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --mimi-device cuda:1 \
  --num-codebooks 4 \
  --text-delay 2.0 \
  --acoustic-delay 2 \
  --batch-size 2 \
  --grad-accum 2 \
  --steps 100000 \
  --output runs/stt_continuation
```

TensorBoard logs include masked accuracy and CE loss for each RQ level, text
accuracy, validation metrics, and both optimizer learning rates:

```bash
tensorboard --logdir runs/stt_continuation/tensorboard
```

The manifest is deterministically split into train and validation sets. The
exact indices and seed are saved in `OUTPUT/split.json`.
