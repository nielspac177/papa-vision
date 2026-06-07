"""Tests for the model zoo: construction, forward shapes, parameter counts, hooks."""
import pytest
import torch

from papavision.gradcam import _resolve_layer
from papavision.models import build_model, count_parameters

TRANSFER = ["mobilenet_v2", "resnet18", "efficientnet_b0"]


def test_custom_cnn_forward_shape():
    model = build_model("custom_cnn", num_classes=3, width=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3)


def test_custom_cnn_is_compact():
    model = build_model("custom_cnn", num_classes=3, width=32)
    n = count_parameters(model)["total"]
    # The whole point is a *small* from-scratch net (< 1M params).
    assert n < 1_000_000


@pytest.mark.parametrize("name", TRANSFER)
def test_transfer_models_forward_shape(name):
    # pretrained=False keeps the test offline.
    model = build_model(name, num_classes=3, pretrained=False, freeze_backbone=True)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 3)


@pytest.mark.parametrize("name", TRANSFER)
def test_frozen_backbone_has_few_trainable_params(name):
    model = build_model(name, num_classes=3, pretrained=False, freeze_backbone=True)
    counts = count_parameters(model)
    # Only the new head should be trainable -> far fewer trainable than total.
    assert counts["trainable"] < counts["total"]


@pytest.mark.parametrize("name", ["custom_cnn"] + TRANSFER)
def test_target_layer_resolves(name):
    model = build_model(name, num_classes=3, pretrained=False)
    layer = _resolve_layer(model, model.target_layer_name)
    assert isinstance(layer, torch.nn.Module)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not_a_model")
