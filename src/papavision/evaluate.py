"""Evaluation: test-set inference, metrics, calibration, bootstrap CIs, McNemar.

Run as a module to evaluate every trained run and aggregate across seeds:

    uv run python -m papavision.evaluate --all

Outputs (under ``results/metrics/``):
    test/<run_id>.json   per-run test metrics (accuracy, macro-F1, per-class, ECE)
    preds/<run_id>.npz   raw test predictions (for paired McNemar tests)
    summary.json         per-model mean +/- 95% CI and pairwise McNemar p-values
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from . import CLASS_LABELS, CLASSES
from .data import get_dataloaders
from .models import build_model, count_parameters
from .stats import bootstrap_metric_ci, expected_calibration_error, mcnemar_test
from .utils import METRICS_DIR, CHECKPOINTS_DIR, get_device, get_logger, load_json, save_json

log = get_logger()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device) -> dict:
    """Run a model over a loader and return labels, predictions, and probabilities.

    Used both for validation during training and for final test evaluation, so the
    two paths are guaranteed to compute predictions identically.
    """
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = probs.argmax(axis=1)
    return {"y_true": y_true, "y_pred": y_pred, "probs": probs}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    """Compute the full set of test metrics for a single run."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=list(range(len(CLASSES))), zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    per_class = {
        CLASS_LABELS[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(CLASSES))
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    # Bootstrap CIs for the two headline metrics.
    acc_ci = bootstrap_metric_ci(y_true, y_pred, accuracy_score)
    f1_ci = bootstrap_metric_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, average="macro",
                                labels=list(range(len(CLASSES))), zero_division=0),
    )
    cal = expected_calibration_error(probs, y_true)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "accuracy_ci": acc_ci,
        "macro_f1_ci": f1_ci,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "ece": cal["ece"],
        "calibration": cal,
    }


# --------------------------------------------------------------------------- #
# Per-run evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(run_record_path: Path, device=None) -> dict:
    """Load a trained run, evaluate on the test set, and persist metrics + preds."""
    record = load_json(run_record_path)
    cfg = record["config"]
    run_id = record["run_id"]
    device = device or get_device(cfg.get("device", "auto"))

    # Rebuild the exact dataloaders (same fixed split_seed => same test set).
    loaders, meta = get_dataloaders(
        data_root=record.get("data_root"),
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
    ckpt_path = CHECKPOINTS_DIR / f"{run_id}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    out = collect_predictions(model, loaders["test"], device)
    metrics = classification_metrics(out["y_true"], out["y_pred"], out["probs"])
    metrics.update(
        {
            "run_id": run_id,
            "model": cfg["model"],
            "seed": cfg["seed"],
            "params": count_parameters(model),
            "n_test": meta["n_test"],
        }
    )

    save_json(metrics, METRICS_DIR / "test" / f"{run_id}.json")
    preds_dir = METRICS_DIR / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        preds_dir / f"{run_id}.npz",
        y_true=out["y_true"], y_pred=out["y_pred"], probs=out["probs"],
    )
    log.info("[eval] %-22s acc=%.4f macroF1=%.4f ECE=%.4f",
             run_id, metrics["accuracy"], metrics["macro_f1"], metrics["ece"])
    return metrics


# --------------------------------------------------------------------------- #
# Aggregation across seeds + pairwise McNemar
# --------------------------------------------------------------------------- #
def aggregate() -> dict:
    """Aggregate per-run metrics into per-model summaries with CIs + McNemar tests."""
    test_dir = METRICS_DIR / "test"
    records = [load_json(p) for p in sorted(test_dir.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"No per-run metrics in {test_dir}. Train + evaluate first.")

    # Group by model.
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    summary = {"models": {}, "mcnemar": {}}
    for model, runs in by_model.items():
        accs = np.array([r["accuracy"] for r in runs])
        f1s = np.array([r["macro_f1"] for r in runs])
        eces = np.array([r["ece"] for r in runs])
        summary["models"][model] = {
            "n_seeds": len(runs),
            "params": runs[0]["params"],
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "ece_mean": float(eces.mean()),
            "seeds": [r["seed"] for r in runs],
        }

    # Pairwise McNemar on the representative (seed 0, else first) run of each model.
    preds_dir = METRICS_DIR / "preds"

    def rep_preds(model: str):
        runs = sorted(by_model[model], key=lambda r: r["seed"])
        rid = runs[0]["run_id"]
        d = np.load(preds_dir / f"{rid}.npz")
        return d["y_true"], d["y_pred"], rid

    model_names = sorted(by_model.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            yt_a, yp_a, rid_a = rep_preds(model_names[i])
            yt_b, yp_b, rid_b = rep_preds(model_names[j])
            if not np.array_equal(yt_a, yt_b):
                log.warning("Test labels differ between %s and %s; skipping McNemar.",
                            rid_a, rid_b)
                continue
            res = mcnemar_test(yt_a, yp_a, yp_b)
            summary["mcnemar"][f"{model_names[i]}__vs__{model_names[j]}"] = res

    save_json(summary, METRICS_DIR / "summary.json")
    log.info("[eval] wrote summary.json for %d models.", len(by_model))
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained papa-vision runs.")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate every run record then aggregate.")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to a single run record JSON.")
    args = parser.parse_args()

    device = get_device()
    runs_dir = METRICS_DIR / "runs"

    if args.run:
        evaluate_run(Path(args.run), device)
    elif args.all:
        # Only evaluate canonical experiment runs (skip HPO/smoke scratch records).
        run_records = [p for p in sorted(runs_dir.glob("*.json"))
                       if not p.name.startswith(("hpo_", "_")) and "_smoke" not in p.name]
        if not run_records:
            raise FileNotFoundError(f"No run records in {runs_dir}. Train first.")
        for rr in run_records:
            evaluate_run(rr, device)

    aggregate()


if __name__ == "__main__":
    main()
