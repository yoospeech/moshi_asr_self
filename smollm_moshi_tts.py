"""SmolLM2 temporal backbone with Moshi's original Depth Transformer.

The temporal stack is replaced by a pretrained Hugging Face causal model.  All
audio delay handling and the teacher-forced Depformer path stay in Moshi's
``LMModel`` implementation.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


_MOSHI_SOURCE = Path(__file__).resolve().parent / "thirdparty" / "moshi" / "moshi"
if str(_MOSHI_SOURCE) not in sys.path:
    sys.path.insert(0, str(_MOSHI_SOURCE))

from moshi.models.lm import LMModel, LMOutput  # noqa: E402


@dataclass
class TTSModelConfig:
    temporal_model: str = "models/SmolLM2-135M"
    text_card: int = 49154
    audio_card: int = 2048
    num_codebooks: int = 4
    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 6
    depformer_dim_feedforward: int = 4096
    context: int = 3000
    frame_rate: float = 12.5
    tts_audio_delay_seconds: float = 2.0
    acoustic_delay_steps: int = 2
    text_padding_id: int = 49152
    text_epad_id: int = 49153

    @property
    def tts_audio_delay_steps(self) -> int:
        return int(round(self.tts_audio_delay_seconds * self.frame_rate))

    @property
    def delays(self) -> list[int]:
        # Paper TTS setup: audio is 2 seconds behind text, then acoustic
        # codebooks are delayed by tau relative to the semantic codebook.
        audio = self.tts_audio_delay_steps
        return [0, audio] + [audio + self.acoustic_delay_steps] * (self.num_codebooks - 1)


class SmolLM2MoshiTTS(LMModel):
    """Assistant-only DSM model: SmolLM2 temporal model + Moshi Depformer."""

    def __init__(
        self,
        config: TTSModelConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        temporal_config = AutoConfig.from_pretrained(config.temporal_model)
        dim = int(temporal_config.hidden_size)
        self.tts_config = config

        # LMModel creates the exact Moshi embeddings, delay logic and
        # Depformer. Its temporal transformer is replaced immediately below.
        super().__init__(
            delays=config.delays,
            n_q=config.num_codebooks,
            dep_q=config.num_codebooks,
            card=config.audio_card,
            text_card=config.text_card,
            dim=dim,
            num_heads=int(temporal_config.num_attention_heads),
            num_layers=1,
            hidden_scale=4,
            norm="rms_norm_f32",
            causal=True,
            context=config.context,
            existing_text_padding_id=config.text_padding_id,
            depformer_dim=config.depformer_dim,
            depformer_dim_feedforward=config.depformer_dim_feedforward,
            depformer_num_heads=config.depformer_num_heads,
            depformer_num_layers=config.depformer_num_layers,
            depformer_context=config.num_codebooks,
            depformer_gating="silu",
            depformer_pos_emb="none",
            depformer_multi_linear=True,
            depformer_weights_per_step=True,
            device=device,
            dtype=dtype,
        )

        self.transformer = AutoModel.from_pretrained(
            config.temporal_model,
            torch_dtype=dtype,
        ).to(device=device)
        self.out_norm = nn.Identity()
        self._seed_text_embeddings_from_temporal()

    def _seed_text_embeddings_from_temporal(self) -> None:
        """Copy pretrained text embeddings while retaining DSM special rows."""
        source = self.transformer.get_input_embeddings().weight
        count = min(source.shape[0], self.text_emb.weight.shape[0])
        with torch.no_grad():
            self.text_emb.weight[:count].copy_(source[:count].to(self.text_emb.weight))

    def forward_text(
        self,
        sequence: torch.Tensor,
        sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cross_attention_src is not None:
            raise ValueError("SmolLM2 temporal mode does not use Moshi cross-attention conditions")

        _, codebooks, _ = sequence.shape
        if codebooks != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} streams, got {codebooks}")

        hidden = self.text_emb(sequence[:, 0])
        for index, embedding in enumerate(self.emb):
            hidden = hidden + embedding(sequence[:, index + self.audio_offset])
        if sum_condition is not None:
            hidden = hidden + sum_condition.to(hidden)

        attention_mask = (sequence != self.zero_token_id).any(dim=1)
        output = self.transformer(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        temporal_hidden = output.last_hidden_state
        return temporal_hidden, self.text_linear(temporal_hidden)[:, None]

    def checkpoint_metadata(self) -> dict[str, object]:
        return asdict(self.tts_config)


def masked_tts_loss(
    output: LMOutput,
    streams: torch.Tensor,
    *,
    text_weight: float = 1.0,
    audio_weight: float = 1.0,
    semantic_weight: float = 100.0,
    text_padding_id: int | None = None,
    text_padding_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equation 7: text CE plus normalized, semantic-weighted audio CE."""
    audio_targets = streams[:, 1 : 1 + output.logits.shape[1]]
    text_targets = streams[:, :1]

    per_codebook = []
    metrics: dict[str, torch.Tensor] = {}
    for codebook in range(output.logits.shape[1]):
        mask = output.mask[:, codebook]
        valid_logits = output.logits[:, codebook][mask].float()
        valid_targets = audio_targets[:, codebook][mask]
        codebook_loss = nn.functional.cross_entropy(valid_logits, valid_targets)
        per_codebook.append(codebook_loss)
        metrics[f"rq/{codebook}_loss"] = codebook_loss.detach()
        metrics[f"rq/{codebook}_accuracy"] = (
            valid_logits.argmax(dim=-1) == valid_targets
        ).float().mean().detach()
    audio_weights = output.logits.new_ones(len(per_codebook), dtype=torch.float32)
    audio_weights[0] = semantic_weight
    audio_losses = torch.stack(per_codebook)
    audio_loss = (audio_losses * audio_weights).sum() / audio_weights.sum()

    text_logits = output.text_logits[output.text_mask].float()
    valid_text_targets = text_targets[output.text_mask]
    text_losses = nn.functional.cross_entropy(text_logits, valid_text_targets, reduction="none")
    if text_padding_id is not None:
        weights = torch.where(
            valid_text_targets == text_padding_id,
            text_losses.new_tensor(text_padding_weight),
            text_losses.new_tensor(1.0),
        )
        text_loss = (text_losses * weights).sum() / weights.sum()
    else:
        text_loss = text_losses.mean()
    total = audio_weight * audio_loss + text_weight * text_loss
    metrics.update({
        "loss/total": total.detach(),
        "loss/audio": audio_loss.detach(),
        "loss/text": text_loss.detach(),
        "text/accuracy": (
            text_logits.argmax(dim=-1) == valid_text_targets
        ).float().mean().detach(),
    })
    return total, metrics
