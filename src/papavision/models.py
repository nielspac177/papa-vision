"""Model zoo: a from-scratch CNN and three ImageNet-pretrained transfer backbones.

All models expose the same interface — ``forward(x) -> logits`` of shape
``(B, num_classes)`` — so the training/evaluation code is architecture-agnostic.
The Grad-CAM module relies on ``target_layer_name`` to know which feature map to
hook for each architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

NUM_CLASSES_DEFAULT = 3


# --------------------------------------------------------------------------- #
# From-scratch CNN
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """Two 3x3 convolutions (each Conv-BN-ReLU) followed by 2x2 max pooling.

    BatchNorm stabilizes training of the from-scratch network on a small dataset,
    and stacking two convolutions per block grows the receptive field cheaply.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """A compact 4-block CNN trained from scratch (~0.4M params at width=32).

    Architecture: stem -> 4 ConvBlocks (channels double each block) -> global
    average pooling -> dropout -> linear classifier. Global average pooling (vs. a
    large flattened FC layer) keeps the parameter count tiny and is what the
    Grad-CAM hook targets (``features`` is the last conv block).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES_DEFAULT,
        width: int = 32,
        dropout: float = 0.3,
        in_ch: int = 3,
    ):
        super().__init__()
        # Channels grow then plateau (.., 4x) rather than doubling to 8x, which
        # keeps the network genuinely compact (~0.6M params at width=32) — the
        # last 8x block would otherwise dominate the parameter budget.
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 4
        self.features = nn.Sequential(
            ConvBlock(in_ch, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
            ConvBlock(c3, c4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )
        # The conv feature map Grad-CAM should hook.
        self.target_layer_name = "features"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Transfer-learning backbones
# --------------------------------------------------------------------------- #
def _swap_head(model: nn.Module, name: str, num_classes: int, dropout: float) -> str:
    """Replace a torchvision classifier head with a fresh ``num_classes`` layer.

    Returns the name of the convolutional layer Grad-CAM should hook.
    """
    if name == "mobilenet_v2":
        in_f = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return "features"
    if name == "resnet18":
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return "layer4"
    if name == "efficientnet_b0":
        in_f = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return "features"
    raise ValueError(f"Unknown transfer model: {name}")


def _build_transfer(
    name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    dropout: float,
) -> nn.Module:
    """Instantiate a torchvision backbone and attach a fresh classification head.

    When ``freeze_backbone`` is set, all pretrained weights are frozen and only the
    new head is trained — the standard, fast, low-data transfer-learning recipe
    that is the point of comparison against the from-scratch CNN.
    """
    weights = "DEFAULT" if pretrained else None
    builders = {
        "mobilenet_v2": tvm.mobilenet_v2,
        "resnet18": tvm.resnet18,
        "efficientnet_b0": tvm.efficientnet_b0,
    }
    if name not in builders:
        raise ValueError(f"Unknown transfer model: {name}")
    model = builders[name](weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    target = _swap_head(model, name, num_classes, dropout)  # head params stay trainable
    model.target_layer_name = target
    return model


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(
    name: str,
    num_classes: int = NUM_CLASSES_DEFAULT,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    width: int = 32,
    dropout: float = 0.3,
) -> nn.Module:
    """Construct a model by name.

    ``name`` is one of: ``custom_cnn``, ``mobilenet_v2``, ``resnet18``,
    ``efficientnet_b0``. Unknown names raise ``ValueError``.
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, width=width, dropout=dropout)
    return _build_transfer(name, num_classes, pretrained, freeze_backbone, dropout)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
