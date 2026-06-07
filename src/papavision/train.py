"""Training entry point.

    uv run python -m papavision.train --config configs/custom_cnn.yaml --seed 0

Trains one model for one seed, doing model selection on the validation macro-F1
with early stopping, and writes:

    results/checkpoints/<run_id>.pt        best-val weights
    results/metrics/runs/<run_id>.json     config + meta + per-epoch history

The fixed ``split_seed`` (separate from the training ``--seed``) keeps the
train/val/test partition identical across every run so the test set is shared.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from . import CLASSES
from .data import compute_class_weights, get_dataloaders, make_synthetic_dataset
from .evaluate import collect_predictions
from .models import build_model, count_parameters
from .utils import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    save_json,
    set_seed,
)

log = get_logger()


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one training epoch; return the mean per-sample loss."""
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss_and_f1(model, loader, criterion, device) -> tuple[float, float]:
    """Compute mean loss and macro-F1 on a loader (used for validation)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        all_pred.append(logits.argmax(1).cpu().numpy())
        all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=list(range(len(CLASSES))), zero_division=0)
    return total_loss / max(n, 1), float(macro_f1)


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def train(cfg: dict, data_root: str | None = None) -> dict:
    """Train a single model defined by ``cfg`` and return the run record."""
    seed = cfg["seed"]
    run_id = cfg["run_id"]
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = get_device(cfg.get("device", "auto"))
    ensure_dirs(CHECKPOINTS_DIR, METRICS_DIR / "runs")

    loaders, meta = get_dataloaders(
        data_root=data_root,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        split_seed=cfg["split_seed"],
        aug_strength=cfg.get("aug_strength", 1.0),
    )

    model = build_model(
        cfg["model"],
        num_classes=len(CLASSES),
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", True),
        width=cfg.get("width", 32),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    params = count_parameters(model)
    log.info("Model %s: %s params (%s trainable)",
             cfg["model"], f"{params['total']:,}", f"{params['trainable']:,}")

    # Class-weighted cross-entropy counters class imbalance + optional smoothing.
    class_weights = compute_class_weights(meta["train_class_counts"]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.get("label_smoothing", 0.0)
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = None
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_f1, best_epoch, patience_left = -1.0, -1, cfg.get("patience", 8)
    history = []
    ckpt_path = CHECKPOINTS_DIR / f"{run_id}.pt"
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_f1 = eval_loss_and_f1(model, loaders["val"], criterion, device)
        if scheduler:
            scheduler.step()

        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_macro_f1": val_f1}
        )
        log.info("  epoch %02d/%d  train_loss=%.4f  val_loss=%.4f  val_macroF1=%.4f",
                 epoch, cfg["epochs"], tr_loss, val_loss, val_f1)

        # Model selection + early stopping on validation macro-F1.
        if val_f1 > best_f1:
            best_f1, best_epoch, patience_left = val_f1, epoch, cfg.get("patience", 8)
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("  early stopping at epoch %d (best val macroF1=%.4f @ %d)",
                         epoch, best_f1, best_epoch)
                break

    record = {
        "run_id": run_id,
        "config": cfg,
        "data_root": str(data_root) if data_root else None,
        "meta": meta,
        "params": params,
        "history": history,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 2),
        "device": str(device),
        "checkpoint": str(ckpt_path),
    }
    save_json(record, METRICS_DIR / "runs" / f"{run_id}.json")
    log.info("Done %s in %.1fs (best val macroF1=%.4f).", run_id, record["train_seconds"], best_f1)
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_cfg(args) -> tuple[dict, str | None]:
    """Resolve the effective config from the YAML file + CLI overrides."""
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg.setdefault("split_seed", 42)

    data_root = args.data_root
    tag = ""
    if args.smoke:
        # Tiny, fast, self-contained run for CI: synthetic data, 1 epoch.
        synth_root = RESULTS_DIR / "_smoke_data"
        make_synthetic_dataset(synth_root, n_per_class=24, size=cfg["img_size"], seed=0)
        data_root = str(synth_root)
        cfg["epochs"] = 1
        cfg["batch_size"] = min(cfg["batch_size"], 16)
        cfg["pretrained"] = False  # no weight download in CI
        tag = "_smoke"

    cfg["run_id"] = f"{cfg['model']}_seed{args.seed}{tag}"
    return cfg, data_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one papa-vision model.")
    parser.add_argument("--config", required=True, help="Path to a model YAML config.")
    parser.add_argument("--seed", type=int, default=0, help="Training seed (model init).")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root (defaults to data/potato).")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast synthetic 1-epoch run for CI smoke testing.")
    args = parser.parse_args()

    cfg, data_root = build_cfg(args)
    train(cfg, data_root=data_root)


if __name__ == "__main__":
    main()
