"""Model construction (segmentation_models_pytorch) and size reporting."""
from __future__ import annotations

import io

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_model(arch: str = "unet", encoder: str = "resnet34", weights: str | None = "imagenet",
                in_channels: int = 1, classes: int = 1) -> nn.Module:
    """Single-channel input: smp adapts the pretrained first conv by summing its RGB filters, which
    is the right thing for a grayscale H-alpha image (no fake RGB replication needed)."""
    return smp.create_model(arch, encoder_name=encoder, encoder_weights=weights,
                            in_channels=in_channels, classes=classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module) -> float:
    """Serialized state_dict size (what ships in the repo / submission)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
