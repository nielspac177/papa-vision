"""Grad-CAM (Selvaraju et al., 2017) for visual model interpretability.

Grad-CAM highlights the image regions most responsible for a class prediction by
weighting the activations of a convolutional layer with the gradients of the class
score w.r.t. those activations. We use it to ask a pointed question: *does the model
look at the leaf lesion, or at the background?* — directly probing the PlantVillage
background-bias pitfall discussed in the paper.

Implementation is dependency-free (forward/backward hooks only) so it works
identically for the from-scratch CNN and the torchvision backbones.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_layer(model: nn.Module, name: str) -> nn.Module:
    """Resolve a (possibly dotted) attribute path to a submodule."""
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


class GradCAM:
    """Compute Grad-CAM saliency maps for a model + target conv layer.

    Usage::

        cam = GradCAM(model, target_layer="features")
        heatmap = cam(input_tensor, target_class=2)   # (H, W) in [0, 1]
        cam.remove()                                   # detach hooks when done
    """

    def __init__(self, model: nn.Module, target_layer: str | None = None):
        self.model = model
        self.model.eval()
        layer_name = target_layer or getattr(model, "target_layer_name", None)
        if layer_name is None:
            raise ValueError("No target_layer given and model has no target_layer_name.")
        self.layer = _resolve_layer(model, layer_name)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activation),
            self.layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Return a normalized Grad-CAM heatmap for a single image.

        ``input_tensor`` is ``(1, C, H, W)``. If ``target_class`` is None the
        model's top prediction is explained.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        # The input MUST be part of the autograd graph. For the transfer models the
        # entire backbone is frozen (requires_grad=False); without a grad-requiring
        # input, the target conv layer's output has no grad_fn, the backward hook
        # never fires, and self._gradients stays None. Marking the input as
        # grad-requiring makes gradients flow back to the target layer for both the
        # from-scratch CNN and the frozen-backbone backbones.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]
            score.backward()

        # Grad-CAM: channel weights = global-average-pooled gradients.
        grads = self._gradients          # (1, K, h, w)
        acts = self._activations         # (1, K, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)

        # Upsample to the input resolution and normalize to [0, 1].
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self) -> None:
        """Detach the forward/backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Invert ImageNet normalization for display; return an HxWx3 array in [0, 1]."""
    t = tensor.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    img = t.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)
