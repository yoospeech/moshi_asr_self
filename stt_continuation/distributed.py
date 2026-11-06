"""Process-local device selection and the DDP training entry point."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from .model import masked_text_continuation_loss


def setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError("LOCAL_RANK exceeds the visible GPU count")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(minutes=60),
        )
    return rank, world_size, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


class TrainingObjective(nn.Module):
    """DDP must invoke forward, not a method on the unwrapped model."""

    def __init__(self, model, padding_id):
        super().__init__()
        self.model = model
        self.padding_id = padding_id

    def forward(self, streams):
        logits, mask = self.model.forward_text_training(streams)
        return masked_text_continuation_loss(
            logits, mask, streams, text_padding_id=self.padding_id,
        )
