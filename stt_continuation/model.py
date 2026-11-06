"""Audio-conditioned SmolLM2 text-continuation model."""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


_MOSHI_SOURCE = Path(__file__).resolve().parent.parent / "thirdparty" / "moshi" / "moshi"
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
    continuation_delay_steps: int = 1
    acoustic_delay_steps: int = 2
    text_padding_id: int = 49152
    text_epad_id: int = 49153

    @property
    def text_delay_steps(self) -> int:
        return int(round(self.text_delay_seconds * self.frame_rate))

    @property
    def continuation_text_delay_steps(self) -> int:
        return self.text_delay_steps + self.continuation_delay_steps

    @property
    def delays(self) -> list[int]:
        # ASR text follows user audio; continuation follows ASR by one or more
        # pattern steps so it can condition on the aligned ASR token.
        return [self.text_delay_steps, self.continuation_text_delay_steps, 0] + [
            self.acoustic_delay_steps
        ] * (
            self.num_codebooks - 1
        )


class SmolLM2MoshiSTTContinuation(LMModel):
    """Two text outputs conditioned on force-fed user Mimi audio."""

    @property
    def num_text_streams(self) -> int:
        return 2

    @property
    def num_codebooks(self) -> int:
        return self.num_text_streams + self.n_q

    @property
    def audio_offset(self) -> int:
        return self.num_text_streams

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

        # LMModel owns the ASR text embedding/head. Add a second independent
        # text stream for continuation while retaining the same vocabulary.
        self.continuation_text_emb = copy.deepcopy(self.text_emb)
        self.continuation_text_linear = nn.Linear(
            dim,
            config.text_card,
            bias=self.text_linear.bias is not None,
            device=device,
            dtype=dtype,
        )

        self.transformer = AutoModel.from_pretrained(
            config.temporal_model,
            torch_dtype=dtype,
        ).to(device=device)
        self.out_norm = nn.Identity()
        self._seed_text_embeddings_from_temporal()

    def _get_initial_token(self) -> torch.Tensor:
        device = self.device
        text = torch.full(
            (1, self.num_text_streams, 1),
            self.text_initial_token_id,
            device=device,
            dtype=torch.long,
        )
        audio = torch.full(
            (1, self.num_audio_codebooks, 1),
            self.initial_token_id,
            device=device,
            dtype=torch.long,
        )
        return torch.cat([text, audio], dim=1)

    def _seed_text_embeddings_from_temporal(self) -> None:
        """Copy pretrained text embeddings while retaining DSM special rows."""
        source = self.transformer.get_input_embeddings().weight
        count = min(source.shape[0], self.text_emb.weight.shape[0])
        with torch.no_grad():
            self.text_emb.weight[:count].copy_(source[:count].to(self.text_emb.weight))
            self.continuation_text_emb.weight[:count].copy_(
                source[:count].to(self.continuation_text_emb.weight)
            )

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
        hidden = hidden + self.continuation_text_emb(sequence[:, 1])
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
        asr_logits = self.text_linear(temporal_hidden)
        continuation_logits = self.continuation_text_linear(temporal_hidden)
        return temporal_hidden, torch.stack([asr_logits, continuation_logits], dim=1)

    def checkpoint_metadata(self) -> dict[str, object]:
        return asdict(self.stt_continuation_config)

    def forward_text_training(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stream delays and predict both text channels."""
        batch, codebooks, _ = streams.shape
        if codebooks != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} streams, got {codebooks}")
        initial = self._get_initial_token().expand(batch, -1, -1)
        delayed = _delay_sequence(self.delays, streams, initial)
        delayed = torch.cat([initial, delayed], dim=2)
        _, text_logits = self.forward_text(delayed[:, :, :-1])
        text_logits, text_mask = _undelay_sequence(
            self.delays[: self.num_text_streams], text_logits, fill_value=float("nan")
        )
        text_mask &= streams[:, : self.num_text_streams] != self.zero_token_id
        return text_logits, text_mask


def masked_text_continuation_loss(
    text_logits: torch.Tensor,
    text_mask: torch.Tensor,
    streams: torch.Tensor,
    *,
    text_padding_id: int | None = None,
    text_padding_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute independent ASR and continuation CE losses."""
    text_targets = streams[:, :2]
    channel_losses = []
    metrics: dict[str, torch.Tensor] = {}
    for channel, name in enumerate(("asr", "continuation")):
        mask = text_mask[:, channel]
        logits = text_logits[:, channel][mask].float()
        targets = text_targets[:, channel][mask]
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        if text_padding_id is not None:
            weights = torch.where(
                targets == text_padding_id,
                losses.new_tensor(text_padding_weight),
                losses.new_tensor(1.0),
            )
            channel_loss = (losses * weights).sum() / weights.sum()
        else:
            channel_loss = losses.mean()
        predictions = logits.argmax(dim=-1)
        non_pad = (
            targets != text_padding_id
            if text_padding_id is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        non_pad_accuracy = (
            (predictions[non_pad] == targets[non_pad]).float().mean()
            if non_pad.any()
            else logits.new_zeros(())
        )
        channel_losses.append(channel_loss)
        metrics[f"loss/{name}"] = channel_loss.detach()
        metrics[f"{name}/accuracy"] = (predictions == targets).float().mean().detach()
        metrics[f"{name}/non_pad_accuracy"] = non_pad_accuracy.detach()

    total = torch.stack(channel_losses).mean()
    metrics["loss/total"] = total.detach()
    return total, metrics
