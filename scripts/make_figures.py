#!/usr/bin/env python
"""Render every publication figure from the committed results.

Figures are written to ``results/figures/`` and mirrored into ``paper/figures/``
so the LaTeX build always uses the latest plots. The script degrades gracefully:
each figure is attempted independently and missing inputs are reported, not fatal.

Figures produced (when their inputs exist):
    dataset_samples.png      sample leaf per class
    training_curves.png      validation macro-F1 vs. epoch, per model
    confusion_matrices.png    per-model test confusion matrices
    model_comparison.png     accuracy + macro-F1 with error bars across seeds
    reliability_diagrams.png  calibration (reliability) per model
    gradcam_panel.png         Grad-CAM overlays probing lesion vs. background

Usage:
    uv run python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from papavision import CLASS_LABELS, CLASSES  # noqa: E402
from papavision.data import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders  # noqa: E402
from papavision.utils import (  # noqa: E402
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    ROOT,
    get_logger,
    load_json,
)

log = get_logger()

# Colourblind-safe palette (Okabe-Ito) + consistent model ordering/labels.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MODEL_ORDER = ["custom_cnn", "mobilenet_v2", "resnet18", "efficientnet_b0"]
MODEL_LABELS = {
    "custom_cnn": "Custom CNN\n(from scratch)",
    "mobilenet_v2": "MobileNetV2",
    "resnet18": "ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
}
PAPER_FIG_DIR = ROOT / "paper" / "figures"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.autolayout": True,
        }
    )


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for d in (FIGURES_DIR, PAPER_FIG_DIR):
        fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", name)


def _run_records() -> list[dict]:
    return [load_json(p) for p in sorted((METRICS_DIR / "runs").glob("*.json"))
            if "_smoke" not in p.name]


def _test_metrics() -> list[dict]:
    return [load_json(p) for p in sorted((METRICS_DIR / "test").glob("*.json"))
            if "_smoke" not in p.name]


def _seed0(records: list[dict], model: str) -> dict | None:
    cands = [r for r in records if r.get("model", r.get("config", {}).get("model")) == model]
    if not cands:
        return None
    return sorted(cands, key=lambda r: r.get("seed", r.get("config", {}).get("seed", 0)))[0]


# --------------------------------------------------------------------------- #
# 1. Dataset samples
# --------------------------------------------------------------------------- #
def fig_dataset_samples() -> None:
    try:
        loaders, _ = get_dataloaders(batch_size=64, img_size=128)
    except FileNotFoundError:
        log.warning("  [dataset_samples] no data — run `make data`. Skipping.")
        return
    # Grab one example per class from the test set.
    ds = loaders["test"].dataset
    by_class: dict[int, np.ndarray] = {}
    for img, label in ds:
        if label not in by_class:
            arr = img.numpy().transpose(1, 2, 0)
            arr = np.clip(arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
            by_class[label] = arr
        if len(by_class) == len(CLASSES):
            break
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(3 * len(CLASSES), 3.2))
    for i, ax in enumerate(axes):
        if i in by_class:
            ax.imshow(by_class[i])
        ax.set_title(CLASS_LABELS[i])
        ax.axis("off")
    fig.suptitle("PlantVillage potato leaf classes", y=1.02)
    _save(fig, "dataset_samples.png")


# --------------------------------------------------------------------------- #
# 2. Training curves
# --------------------------------------------------------------------------- #
def fig_training_curves() -> None:
    records = _run_records()
    if not records:
        log.warning("  [training_curves] no run records. Skipping.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, model in enumerate(MODEL_ORDER):
        r = _seed0(records, model)
        if not r:
            continue
        hist = r["history"]
        epochs = [h["epoch"] for h in hist]
        f1 = [h["val_macro_f1"] for h in hist]
        ax.plot(epochs, f1, marker="o", ms=3, color=PALETTE[i],
                label=MODEL_LABELS[model].replace("\n", " "))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 during training (seed 0)")
    ax.legend(fontsize=8)
    _save(fig, "training_curves.png")


# --------------------------------------------------------------------------- #
# 3. Confusion matrices
# --------------------------------------------------------------------------- #
def fig_confusion_matrices() -> None:
    metrics = _test_metrics()
    if not metrics:
        log.warning("  [confusion_matrices] no test metrics. Skipping.")
        return
    models = [m for m in MODEL_ORDER if _seed0(metrics, m)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.3))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        r = _seed0(metrics, model)
        cm = np.array(r["confusion_matrix"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASS_LABELS, fontsize=7)
        ax.set_title(f"{MODEL_LABELS[model].replace(chr(10),' ')}\nacc={r['accuracy']:.3f}",
                     fontsize=9)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        if model == models[0]:
            ax.set_ylabel("True", fontsize=8)
        ax.grid(False)
    fig.suptitle("Test-set confusion matrices (counts; colour = row-normalized)", y=1.04)
    _save(fig, "confusion_matrices.png")


# --------------------------------------------------------------------------- #
# 4. Model comparison (accuracy + macro-F1 with error bars)
# --------------------------------------------------------------------------- #
def fig_model_comparison() -> None:
    summary_path = METRICS_DIR / "summary.json"
    if not summary_path.exists():
        log.warning("  [model_comparison] no summary.json. Skipping.")
        return
    summary = load_json(summary_path)["models"]
    models = [m for m in MODEL_ORDER if m in summary]
    x = np.arange(len(models))
    w = 0.38

    acc = [summary[m]["accuracy_mean"] for m in models]
    acc_e = [summary[m]["accuracy_std"] for m in models]
    f1 = [summary[m]["macro_f1_mean"] for m in models]
    f1_e = [summary[m]["macro_f1_std"] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, acc, w, yerr=acc_e, capsize=4, color=PALETTE[0], label="Accuracy")
    ax.bar(x + w / 2, f1, w, yerr=f1_e, capsize=4, color=PALETTE[1], label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test performance (mean ± SD over seeds)")
    ax.legend()
    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax.text(xi - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
    _save(fig, "model_comparison.png")


# --------------------------------------------------------------------------- #
# 5. Reliability diagrams
# --------------------------------------------------------------------------- #
def fig_reliability() -> None:
    metrics = _test_metrics()
    if not metrics:
        log.warning("  [reliability] no test metrics. Skipping.")
        return
    models = [m for m in MODEL_ORDER if _seed0(metrics, m)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        r = _seed0(metrics, model)
        cal = r["calibration"]
        conf = np.array(cal["bin_confidence"])
        acc = np.array(cal["bin_accuracy"])
        counts = np.array(cal["bin_count"])
        mask = counts > 0
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.bar(conf[mask], acc[mask], width=1 / cal["n_bins"] * 0.9,
               color=PALETTE[2], alpha=0.8, edgecolor="black", lw=0.4)
        ax.set_title(f"{MODEL_LABELS[model].replace(chr(10),' ')}\nECE={r['ece']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Confidence", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if model == models[0]:
            ax.set_ylabel("Accuracy", fontsize=8)
    fig.suptitle("Reliability diagrams (calibration)", y=1.04)
    _save(fig, "reliability_diagrams.png")


# --------------------------------------------------------------------------- #
# 6. Grad-CAM panel
# --------------------------------------------------------------------------- #
def fig_gradcam() -> None:
    import torch

    from papavision.gradcam import GradCAM, denormalize
    from papavision.models import build_model

    records = _run_records()
    panel_models = [m for m in ["custom_cnn", "mobilenet_v2"] if _seed0(records, m)]
    if not panel_models:
        log.warning("  [gradcam] no trained models. Skipping.")
        return
    try:
        loaders, _ = get_dataloaders(batch_size=1, img_size=128)
    except FileNotFoundError:
        log.warning("  [gradcam] no data. Skipping.")
        return

    # One representative correctly-handled image per class from the test set.
    examples: dict[int, torch.Tensor] = {}
    for img, label in loaders["test"].dataset:
        lab = int(label)
        if lab not in examples:
            examples[lab] = img
        if len(examples) == len(CLASSES):
            break

    n_rows = 1 + len(panel_models)
    fig, axes = plt.subplots(n_rows, len(CLASSES), figsize=(3 * len(CLASSES), 3 * n_rows))

    # Row 0: original images.
    for j, cls in enumerate(range(len(CLASSES))):
        ax = axes[0, j]
        ax.imshow(denormalize(examples[cls], IMAGENET_MEAN, IMAGENET_STD))
        ax.set_title(CLASS_LABELS[cls])
        ax.axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=10)

    for row, model_name in enumerate(panel_models, start=1):
        r = _seed0(records, model_name)
        cfg = r["config"]
        model = build_model(model_name, num_classes=len(CLASSES),
                            pretrained=cfg.get("pretrained", True),
                            freeze_backbone=cfg.get("freeze_backbone", True),
                            width=cfg.get("width", 32), dropout=cfg.get("dropout", 0.3))
        ckpt = CHECKPOINTS_DIR / f"{r['run_id']}.pt"
        if not ckpt.exists():
            continue
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        cam = GradCAM(model)
        for j, cls in enumerate(range(len(CLASSES))):
            ax = axes[row, j]
            inp = examples[cls].unsqueeze(0)
            heat = cam(inp, target_class=cls)
            base = denormalize(examples[cls], IMAGENET_MEAN, IMAGENET_STD)
            ax.imshow(base)
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.axis("off")
            if j == 0:
                ax.set_title(MODEL_LABELS[model_name].replace("\n", " "),
                             loc="left", fontsize=9)
        cam.remove()

    fig.suptitle("Grad-CAM: where each model looks (per class)", y=1.0)
    _save(fig, "gradcam_panel.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    _style()
    log.info("Rendering figures ...")
    for fn in (
        fig_dataset_samples,
        fig_training_curves,
        fig_confusion_matrices,
        fig_model_comparison,
        fig_reliability,
        fig_gradcam,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one bad figure shouldn't kill the rest
            log.warning("  [%s] failed: %s", fn.__name__, e)
    log.info("Figures written to %s and %s", FIGURES_DIR, PAPER_FIG_DIR)


if __name__ == "__main__":
    main()
