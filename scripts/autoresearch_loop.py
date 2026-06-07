#!/usr/bin/env python
"""Autonomous hyperparameter-search loop for the from-scratch CNN.

This is the reproducible engine behind the project's `/autoresearch` step. It runs
a seeded **random search** over the custom CNN's hyperparameters, selecting on the
**validation** macro-F1 (never the test set, to avoid leakage), and writes the full
trajectory plus the winning configuration to ``results/metrics/autoresearch.json``.

Each trial trains a short run; its temporary checkpoint/record are deleted after the
validation score is read, so the canonical ``results/metrics/runs/`` stays clean.

Usage:
    uv run python scripts/autoresearch_loop.py --trials 12 --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from papavision.train import train  # noqa: E402
from papavision.utils import CHECKPOINTS_DIR, METRICS_DIR, ROOT, get_logger, save_json  # noqa: E402

log = get_logger()

# Search space — deliberately small and interpretable for a course project.
SEARCH_SPACE = {
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "width": [24, 32, 48],
    "dropout": [0.2, 0.3, 0.5],
    "aug_strength": [0.5, 1.0, 1.5],
}

BASE = {
    "model": "custom_cnn",
    "seed": 0,              # fixed during search: compare hyperparameters, not init noise
    "pretrained": False,
    "freeze_backbone": False,
    "img_size": 128,
    "batch_size": 32,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "split_seed": 42,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "label_smoothing": 0.05,
    "patience": 4,
    "device": "auto",
    "deterministic": True,
}


def sample_config(rng: np.random.Generator, epochs: int) -> dict:
    """Draw one hyperparameter configuration from the search space."""
    cfg = dict(BASE)
    cfg["epochs"] = epochs
    for k, choices in SEARCH_SPACE.items():
        cfg[k] = choices[int(rng.integers(len(choices)))]
    return cfg


def _cleanup(run_id: str) -> None:
    """Remove a trial's temporary artifacts."""
    for p in (CHECKPOINTS_DIR / f"{run_id}.pt", METRICS_DIR / "runs" / f"{run_id}.json"):
        Path(p).unlink(missing_ok=True)


def write_protocol_artifacts(result: dict) -> None:
    """Emit the autoresearch protocol files (JSONL + dashboard + worklog) from a
    completed search trajectory, so the run is auditable in the standard format."""
    import json
    import time

    history = result["history"]
    best = result["best"]
    # Sequential keep/discard decisions vs. running best (higher F1 is better).
    running_best = -1.0
    lines = [
        json.dumps({
            "type": "config", "name": "custom_cnn_hpo",
            "metricName": "val_macro_f1", "metricUnit": "", "bestDirection": "higher",
        })
    ]
    rows = []
    for i, h in enumerate(history):
        score = h["val_macro_f1"]
        status = "keep" if score > running_best else "discard"
        if status == "keep":
            running_best = score
        lines.append(json.dumps({
            "run": i + 1, "commit": "n/a", "metric": round(score, 4), "metrics": {},
            "status": status, "description": str(h["hyperparameters"]),
            "timestamp": int(time.time()), "segment": 0,
        }))
        rows.append((i + 1, score, status, h["hyperparameters"]))

    (ROOT / "autoresearch.jsonl").write_text("\n".join(lines) + "\n")

    # Dashboard.
    base = history[0]["val_macro_f1"]
    kept = sum(1 for r in rows if r[2] == "keep")
    dash = [
        "# Autoresearch Dashboard: custom_cnn_hpo",
        "",
        f"**Runs:** {len(rows)} | **Kept:** {kept} | **Discarded:** {len(rows) - kept}",
        f"**Baseline:** val_macro_f1: {base:.4f} (#1)",
        f"**Best:** val_macro_f1: {best['val_macro_f1']:.4f} (#{best['trial'] + 1}, "
        f"{(best['val_macro_f1'] - base) / max(base, 1e-9) * 100:+.1f}%)",
        "",
        "| # | val_macro_f1 | status | hyperparameters |",
        "|---|--------------|--------|-----------------|",
    ]
    for n, score, status, hp in rows:
        delta = (score - base) / max(base, 1e-9) * 100
        dash.append(f"| {n} | {score:.4f} ({delta:+.1f}%) | {status} | `{hp}` |")
    (ROOT / "autoresearch-dashboard.md").write_text("\n".join(dash) + "\n")

    # Worklog.
    wl = [
        "# Autoresearch Worklog: custom_cnn hyperparameter search",
        "",
        f"Random search, {result['trials']} trials x {result['epochs_per_trial']} epochs, "
        f"search_seed={result['search_seed']}. Metric: validation macro-F1 (higher better).",
        "",
    ]
    for n, score, status, hp in rows:
        wl += [
            f"### Run {n}: {hp} — val_macro_f1={score:.4f} ({status.upper()})",
            f"- Result: {score:.4f} (delta vs baseline {(score - base) / max(base, 1e-9) * 100:+.1f}%)",
            "",
        ]
    wl += [
        "## Key Insights",
        f"- Best configuration (trial #{best['trial'] + 1}): `{best['hyperparameters']}` "
        f"reaching val macro-F1 = {best['val_macro_f1']:.4f}.",
        "- These winning hyperparameters are folded into `configs/custom_cnn.yaml` for the",
        "  final 3-seed evaluation runs.",
        "",
    ]
    (ROOT / "experiments").mkdir(exist_ok=True)
    (ROOT / "experiments" / "worklog.md").write_text("\n".join(wl) + "\n")
    log.info("Wrote autoresearch.jsonl, autoresearch-dashboard.md, experiments/worklog.md")


def run_search(trials: int, epochs: int, search_seed: int = 7) -> dict:
    """Run the random search and return the result record."""
    rng = np.random.default_rng(search_seed)
    history, best = [], None

    for t in range(trials):
        cfg = sample_config(rng, epochs)
        cfg["run_id"] = f"hpo_trial{t:02d}"
        hp = {k: cfg[k] for k in SEARCH_SPACE}
        log.info("Trial %02d/%d  %s", t + 1, trials, hp)

        record = train(cfg)  # trains on data/potato, selects on val macro-F1
        score = record["best_val_macro_f1"]
        history.append({"trial": t, "hyperparameters": hp, "val_macro_f1": score})
        if best is None or score > best["val_macro_f1"]:
            best = {"trial": t, "hyperparameters": hp, "val_macro_f1": score}
            log.info("  -> new best val macro-F1 = %.4f", score)
        _cleanup(cfg["run_id"])

    result = {
        "search": "random",
        "trials": trials,
        "epochs_per_trial": epochs,
        "search_seed": search_seed,
        "space": SEARCH_SPACE,
        "history": history,
        "best": best,
    }
    save_json(result, METRICS_DIR / "autoresearch.json")
    write_protocol_artifacts(result)
    log.info("Search complete. Best: %s (val macro-F1=%.4f)",
             best["hyperparameters"], best["val_macro_f1"])
    log.info("Wrote %s", METRICS_DIR / "autoresearch.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for the custom CNN.")
    parser.add_argument("--trials", type=int, default=12, help="Number of random trials.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial (short).")
    parser.add_argument("--search-seed", type=int, default=7, help="Seed for sampling.")
    args = parser.parse_args()
    run_search(args.trials, args.epochs, args.search_seed)


if __name__ == "__main__":
    main()
