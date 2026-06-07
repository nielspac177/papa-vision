"""End-to-end smoke test: train one epoch on synthetic data, then evaluate it."""
from pathlib import Path

from papavision.data import make_synthetic_dataset
from papavision.evaluate import evaluate_run
from papavision.train import train
from papavision.utils import CHECKPOINTS_DIR, METRICS_DIR


def _cfg(run_id: str) -> dict:
    return {
        "model": "custom_cnn",
        "run_id": run_id,
        "seed": 0,
        "width": 8,
        "dropout": 0.3,
        "pretrained": False,
        "freeze_backbone": False,
        "img_size": 48,
        "batch_size": 8,
        "val_frac": 0.2,
        "test_frac": 0.2,
        "split_seed": 42,
        "aug_strength": 1.0,
        "epochs": 1,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "label_smoothing": 0.0,
        "patience": 3,
        "device": "cpu",
        "deterministic": True,
    }


def test_train_and_evaluate_roundtrip(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=24, size=48, seed=0)
    run_id = "custom_cnn_seed0_pytest"
    cfg = _cfg(run_id)

    record = train(cfg, data_root=str(root))
    assert (CHECKPOINTS_DIR / f"{run_id}.pt").exists()
    assert (METRICS_DIR / "runs" / f"{run_id}.json").exists()
    assert len(record["history"]) == 1
    assert record["best_val_macro_f1"] >= 0.0

    metrics = evaluate_run(METRICS_DIR / "runs" / f"{run_id}.json", device="cpu")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "confusion_matrix" in metrics
    assert "calibration" in metrics

    # Cleanup the artifacts this test created.
    for p in [
        CHECKPOINTS_DIR / f"{run_id}.pt",
        METRICS_DIR / "runs" / f"{run_id}.json",
        METRICS_DIR / "test" / f"{run_id}.json",
        METRICS_DIR / "preds" / f"{run_id}.npz",
    ]:
        Path(p).unlink(missing_ok=True)
