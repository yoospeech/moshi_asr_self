# SmolLM2 Moshi TTS

Assistant-only streaming TTS experiment based on the Moshi delayed-stream
modeling architecture. SmolLM2-135M is used as the Temporal Transformer and
Moshi's depthwise Depformer predicts Mimi RVQ tokens.

## Architecture

- SmolLM2-135M Temporal Transformer
- Moshi Depth Transformer with per-RQ parameters
- four online-extracted Mimi codebooks
- 12.5 Hz PAD/EPAD text timeline from word timestamps
- TTS delay pattern `[0, 25, 27, 27, 27]`
- semantic-token loss weight 100, acoustic-token weights 1

The first stream is text, the second is Mimi's semantic codebook, and the last
three are acoustic codebooks. Audio is delayed two seconds behind text; acoustic
codebooks receive an additional two-frame delay.

## Setup

Clone with the Moshi submodule:

```bash
git clone --recurse-submodules https://github.com/yoospeech/moshi_tts_self.git
cd moshi_tts_self
pip install -r requirements.txt
pip install -e thirdparty/moshi/moshi
```

Place or download the SmolLM2 checkpoint at `models/SmolLM2-135M`. Model
weights, datasets, and training runs are intentionally excluded from Git.

## Data

The trainer reads the original DailyTalkContiguous manifest and adjacent word
alignment files directly. The main/left stereo channel is resampled to 24 kHz,
and frozen Mimi extracts audio tokens online.

Default filters retain conversations that satisfy:

- duration between 5 and 45 seconds;
- at least 5 aligned words;
- 0.5 to 2.5 aligned words per second;
- valid word timestamps;
- target-speech coverage between 5% and 80%.

On the current 2,541-item manifest this retains 2,148 conversations, 15.13
hours, and 68,675 aligned words.

## Training

For two 8 GB GPUs, keep the trainable model on GPU 0 and frozen Mimi on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_tts.py \
  --manifest /data/db/daily-talk-contiguous/dailytalk.jsonl \
  --model models/SmolLM2-135M \
  --mimi-device cuda:1 \
  --num-codebooks 4 \
  --depformer-dim 512 \
  --depformer-heads 8 \
  --depformer-layers 4 \
  --depformer-ffn-dim 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --steps 100000 \
  --learning-rate 2e-4 \
  --temporal-learning-rate 2e-5 \
  --valid-ratio 0.05 \
  --valid-every 500 \
  --save-every 500 \
  --output runs/assistant_tts
```

If `--mimi-checkpoint` is omitted, the Mimi checkpoint is resolved from
`kyutai/moshiko-pytorch-bf16`.

Monitor training with:

```bash
tensorboard --logdir runs/assistant_tts/tensorboard --port 6006
```

RQ-specific loss and masked top-1 accuracy, text metrics, validation metrics,
and learning rates are logged to TensorBoard. More implementation details are
in [TTS_TRAINING.md](TTS_TRAINING.md).

## Offline batch inference

`infer_tts.py` synthesizes one or more utterances without the streaming cache.
It recomputes the full temporal prefix at every frame, samples RQ0 through RQ3,
undoes the training delay pattern, and decodes the result with Mimi:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 infer_tts.py \
  --checkpoint runs/assistant_tts_batch3/checkpoint_00002500.pt \
  --config runs/assistant_tts_batch3/config.json \
  --model models/SmolLM2-135M \
  --mimi-device cuda:1 \
  --text "오늘은 날씨가 정말 좋네요." \
  --text "안녕하세요. 만나서 반갑습니다." \
  --output-dir inference_outputs
```

Use `--text-file prompts.txt` to synthesize one non-empty line per utterance.
Because raw text has no word timestamps, the offline path approximates the
training PAD/EPAD timeline. `--frames-per-word` controls the assumed speaking
rate, while `--tail-seconds` controls silence after the final word.
