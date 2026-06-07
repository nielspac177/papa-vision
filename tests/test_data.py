"""Tests for the data pipeline: discovery, leakage-safe splits, transforms, loaders."""
import numpy as np

from papavision import CLASSES
from papavision.data import (
    build_transforms,
    class_counts,
    compute_class_weights,
    discover_samples,
    get_dataloaders,
    make_synthetic_dataset,
    stratified_split,
)


def test_synthetic_dataset_has_all_classes(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=10, size=32, seed=0)
    samples = discover_samples(root)
    assert len(samples) == 30
    counts = class_counts(samples)
    assert all(counts[c] == 10 for c in CLASSES)


def test_stratified_split_is_leakage_free(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=20, size=32, seed=1)
    samples = discover_samples(root)
    parts = stratified_split(samples, val_frac=0.2, test_frac=0.2, split_seed=42)

    # Partitions cover everything and never overlap (no leakage).
    train_paths = {p for p, _ in parts["train"]}
    val_paths = {p for p, _ in parts["val"]}
    test_paths = {p for p, _ in parts["test"]}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert len(train_paths | val_paths | test_paths) == len(samples)


def test_split_is_deterministic(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=20, size=32, seed=2)
    samples = discover_samples(root)
    a = stratified_split(samples, split_seed=42)
    b = stratified_split(samples, split_seed=42)
    assert [p for p, _ in a["test"]] == [p for p, _ in b["test"]]


def test_transforms_output_shape(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=4, size=64, seed=3)
    samples = discover_samples(root)
    tf = build_transforms(img_size=96, train=True)
    from PIL import Image

    out = tf(Image.open(samples[0][0]).convert("RGB"))
    assert out.shape == (3, 96, 96)


def test_class_weights_upweight_rare_classes():
    counts = {CLASSES[0]: 100, CLASSES[1]: 1000, CLASSES[2]: 1000}
    w = compute_class_weights(counts).numpy()
    # The rare (healthy) class must get the largest weight.
    assert w[0] > w[1] and w[0] > w[2]
    assert np.isclose(w.mean(), 1.0, atol=1e-5)


def test_get_dataloaders_on_synthetic(tmp_path):
    root = make_synthetic_dataset(tmp_path / "ds", n_per_class=20, size=48, seed=4)
    loaders, meta = get_dataloaders(data_root=root, img_size=48, batch_size=8)
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (3, 48, 48)
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == 60
