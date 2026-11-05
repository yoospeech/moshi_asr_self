"""Audio-conditioned SmolLM2 text-continuation model."""

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

from moshi.models.lm import LMModel  # noqa: E402
from moshi.models.lm_utils import _delay_sequence, _undelay_sequence  # noqa: E402


@dataclass
class STTContinuationConfig:
    temporal_model: str = "models/SmolLM2-135M"
    text_card: int = 49154
    audio_card: int = 2048
    num_codebooks: int = 4
    context: int = 3000
    frame_rate: float = 12.5
    text_delay_seconds: float = 2.0
    acoustic_delay_steps: int = 2
    text_padding_id: int = 49152
    text_epad_id: int = 49153
    user_begin_id: int = 49154
    user_end_id: int = 49155
    assistant_begin_id: int = 49156
    assistant_end_id: int = 49157

    @property
    def text_delay_steps(self) -> int:
        return int(round(self.text_delay_seconds * self.frame_rate))

    @property
    def delays(self) -> list[int]:
        # User semantic audio leads text; acoustic codebooks trail semantic audio.
        return [self.text_delay_steps, 0] + [self.acoustic_delay_steps] * (
            self.num_codebooks - 1
        )


class SmolLM2MoshiSTTContinuation(LMModel):
    """User-only delayed-stream STT-continuation model."""

    def __init__(
        self,
        config: STTContinuationConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        temporal_config = AutoConfig.from_pretrained(config.temporal_model)
        dim = int(temporal_config.hidden_size)
        self.stt_continuation_config = config

        # Keep Moshi's stream embeddings and delay metadata, but disable the
        # audio Depformer: Mimi codes are conditioning inputs, not targets.
        super().__init__(
            delays=config.delays,
            n_q=config.num_codebooks,
            dep_q=0,
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
            device=device,
            dtype=dtype,
        )
        # LMModel allocates a projection even with dep_q=0; it is unreachable
        # in text-only training/inference, so do not retain it in the model or
        # checkpoints.
        del self.depformer_in

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
        return asdict(self.stt_continuation_config)

    def forward_text_training(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Moshi delays and predict text without audio-token logits."""
        batch, codebooks, _ = streams.shape
        if codebooks != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} streams, got {codebooks}")
        initial = self._get_initial_token().expand(batch, -1, -1)
        delayed = _delay_sequence(self.delays, streams, initial)
        delayed = torch.cat([initial, delayed], dim=2)
        _, text_logits = self.forward_text(delayed[:, :, :-1])
        text_logits, text_mask = _undelay_sequence(
            self.delays[:1], text_logits, fill_value=float("nan")
        )
        text_mask &= streams[:, :1] != self.zero_token_id
        return text_logits, text_mask


def masked_text_continuation_loss(
    text_logits: torch.Tensor,
    text_mask: torch.Tensor,
    streams: torch.Tensor,
    *,
    text_padding_id: int | None = None,
    text_padding_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute text-only CE conditioned on the input Mimi audio tokens."""
    text_targets = streams[:, :1]
    valid_text_logits = text_logits[text_mask].float()
    valid_text_targets = text_targets[text_mask]
    text_losses = nn.functional.cross_entropy(
        valid_text_logits, valid_text_targets, reduction="none"
    )
    if text_padding_id is not None:
        weights = torch.where(
            valid_text_targets == text_padding_id,
            text_losses.new_tensor(text_padding_weight),
            text_losses.new_tensor(1.0),
        )
        text_loss = (text_losses * weights).sum() / weights.sum()
    else:
        text_loss = text_losses.mean()
    text_predictions = valid_text_logits.argmax(dim=-1)
    non_pad_text_mask = (
        valid_text_targets != text_padding_id
        if text_padding_id is not None
        else torch.ones_like(valid_text_targets, dtype=torch.bool)
    )
    non_pad_text_accuracy = (
        (text_predictions[non_pad_text_mask] == valid_text_targets[non_pad_text_mask]).float().mean()
        if non_pad_text_mask.any()
        else valid_text_logits.new_zeros(())
    )
    metrics = {
        "loss/total": text_loss.detach(),
        "loss/text": text_loss.detach(),
        "text/accuracy": (text_predictions == valid_text_targets).float().mean().detach(),
        "text/non_pad_accuracy": non_pad_text_accuracy.detach(),
    }
    return text_loss, metrics
