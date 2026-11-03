# Assistant TTS training

This is the first (assistant-only) stage of the Moshi-style model:

`aligned text + assistant Mimi streams -> SmolLM2 temporal -> Moshi Depformer`

The Depth Transformer, delayed-stream transformation, masks, and teacher-forcing
path come directly from `thirdparty/moshi/moshi/moshi/models/lm.py`. Only the
temporal transformer is replaced with the local SmolLM2 checkpoint.

## Data contract

The trainer consumes the original DailyTalkContiguous manifest directly:

```json
{"path": "data_stereo/0.wav", "duration": 24.521950113378686}
```

The left/main channel is decoded and resampled to 24 kHz. Frozen Mimi extracts
four codebooks online in every training and validation step. Word timestamps
from the adjacent JSON sidecar are tokenized word-by-word with the local
SmolLM2 tokenizer and converted into a 12.5 Hz PAD/EPAD timeline. PAD and EPAD
are allocated immediately after the tokenizer's native vocabulary.

The paper TTS layout delays semantic audio by 2 seconds (25 steps at 12.5 Hz)
relative to text. Acoustic codebooks receive another two-step delay, so the
four-codebook training pattern is `[0, 25, 27, 27, 27]` for
`[text, semantic, acoustic-1, acoustic-2, acoustic-3]`.
The online stream builder appends 27 text-PAD frames per item so the final
delayed audio targets are not dropped from training.

## Validation and training

Validate the full forward/backward path first:

```bash
python3 train_tts.py --smoke-test
```

Then train on aligned/precomputed data:

```bash
python3 train_tts.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --num-codebooks 4 \
  --tts-audio-delay 2.0 \
  --acoustic-delay 2 \
  --batch-size 2 \
  --grad-accum 8 \
  --steps 100000 \
  --output runs/assistant_tts
```

For a lower-memory two-GPU run, keep the trainable model on GPU 0, place
frozen online Mimi on GPU 1, and use a compact Moshi-structured Depformer:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_tts.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --mimi-device cuda:1 \
  --depformer-dim 512 \
  --depformer-heads 8 \
  --depformer-layers 4 \
  --depformer-ffn-dim 2048 \
  --batch-size 1 \
  --grad-accum 8
```

The pretrained temporal backbone uses a lower learning rate than the new
stream embeddings, output heads, and Depformer. Checkpoints contain both model
and optimizer state. The next data-preparation stage must encode waveforms with
the chosen Mimi checkpoint and construct the aligned DSM text timeline.

TensorBoard logs are written to `OUTPUT/tensorboard` and include masked top-1
accuracy and CE loss for every RQ level, text accuracy, aggregate losses, and
both optimizer learning rates:

```bash
tensorboard --logdir runs/assistant_tts/tensorboard
```

The token manifest is deterministically split into train and validation sets.
The default validation ratio is 5%; the exact indices and seed are saved to
`OUTPUT/split.json`. Validation metrics use the `valid/` TensorBoard prefix.

```bash
python3 train_tts.py \
  --manifest data/train.jsonl \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --valid-batches 20
```

DailyTalk is filtered before splitting. Defaults retain conversations between
5 and 45 seconds with at least 5 aligned words, 0.5–2.5 words/second, valid
timestamp ranges, and 5–80% target-speech coverage. On the current manifest
this keeps 2,148 conversations (15.13 hours, 68,675 aligned words).
