# Time-aligned training-set samples

These examples were synthesized from the exact text/PAD/EPAD timelines built
from each training sample's word timestamps. The reference waveform and Mimi
audio tokens were not supplied during generation.

- Checkpoint: `runs/assistant_tts_batch3/checkpoint_00100000.pt`
- Delay pattern: `[0, 25, 27, 27, 27]`
- Sampling: temperature `0.6`, top-k `100`, seed `42`
- Codec: frozen Mimi, four codebooks, 24 kHz

| Sample | Expected transcript | Whisper-small on generated audio |
|---|---|---|
| `sample_01_914` | Is that Sue? This is George. | Is this Sue? This is George. |
| `sample_02_197` | Wrong number. No, it is not. | Wrong number. No, it is not. |
| `sample_03_1231` | Sophie, I hear your team won the match! Congratulations! | Sophie, I hear you're team-walling. Really? |

Each sample has a `*_reference.wav` and `*_generated.wav` file. The third pair
is intentionally retained as a representative imperfect result rather than
selecting only successful generations.
