"""Data pipeline: dataset discovery, leakage-safe stratified splits, transforms,
DataLoaders, and a synthetic dataset generator used by the tests / CI / smoke runs.

Design notes
------------
* **One fixed split for everyone.** The train/val/test partition is governed by a
  dedicated ``split_seed`` that is held constant across all models and all training
  seeds. This guarantees every model is evaluated on the *identical* test images,
  which is what makes the paired McNemar test (see ``stats.py``) valid.
* **Leakage-safe.** Splitting is performed once over file paths; the same image can
  never appear in two partitions.
* **ImageNet normalization.** Applied for both the from-scratch and transfer models
  so the only thing that differs between them is the architecture/weights, not the
  input statistics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import CLASSES
from .utils import DATA_DIR, get_logger

log = get_logger()

# ImageNet statistics — required for the pretrained backbones, harmless for the
# from-scratch CNN, and applied uniformly so comparisons are apples-to-apples.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class PotatoLeafDataset(Dataset):
    """A simple image-classification dataset backed by a list of ``(path, label)``.

    Splitting happens *outside* this class (on the path list), so a dataset
    instance only ever sees the samples assigned to its partition — there is no
    way for it to leak across the train/val/test boundary.
    """

    def __init__(self, samples: Sequence[tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # Convert to RGB so grayscale/CMYK images don't break the conv stack.
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Discovery + splitting
# --------------------------------------------------------------------------- #
def discover_samples(root: str | Path) -> list[tuple[Path, int]]:
    """Scan ``root/<class_name>/*`` and return ``(path, label)`` pairs.

    Labels follow the canonical ``CLASSES`` ordering from the package root, so
    index 0 is always *healthy*, 1 *early blight*, 2 *late blight*.
    """
    root = Path(root)
    samples: list[tuple[Path, int]] = []
    for label, cls in enumerate(CLASSES):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix in IMG_EXTENSIONS:
                samples.append((p, label))
    return samples


def stratified_split(
    samples: Sequence[tuple[Path, int]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_seed: int = 42,
) -> dict[str, list[tuple[Path, int]]]:
    """Partition samples into train/val/test, stratified by class.

    The split is deterministic in ``split_seed`` and must use the SAME seed for
    every experiment so the test set is shared (a prerequisite for paired
    significance testing).
    """
    labels = [lbl for _, lbl in samples]
    idx = np.arange(len(samples))

    # First carve off the test set.
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=split_seed, stratify=labels
    )
    # Then split the remainder into train/val (val_frac expressed w.r.t. the whole).
    rel_val = val_frac / (1.0 - test_frac)
    tv_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=split_seed, stratify=tv_labels
    )

    take = lambda ids: [samples[i] for i in ids]
    return {"train": take(train_idx), "val": take(val_idx), "test": take(test_idx)}


def class_counts(samples: Sequence[tuple[Path, int]]) -> dict[str, int]:
    """Count samples per class name (for logging + class-weighted loss)."""
    counts = {c: 0 for c in CLASSES}
    for _, lbl in samples:
        counts[CLASSES[lbl]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int = 128, train: bool = False, aug_strength: float = 1.0):
    """Compose torchvision transforms.

    Training augmentation (flip, rotation, colour jitter, random resized crop) is
    scaled by ``aug_strength`` so the hyperparameter search can dial it up/down.
    Evaluation uses a deterministic resize + centre crop.
    """
    if train:
        s = aug_strength
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(1.0 - 0.2 * s, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5 * min(s, 1.0)),
                transforms.RandomRotation(degrees=15 * s),
                transforms.ColorJitter(
                    brightness=0.2 * s, contrast=0.2 * s, saturation=0.2 * s
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def get_dataloaders(
    data_root: str | Path | None = None,
    img_size: int = 128,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_seed: int = 42,
    aug_strength: float = 1.0,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], dict]:
    """Build train/val/test DataLoaders plus a metadata dict.

    If ``data_root`` is ``None`` it defaults to ``data/potato``. The function
    raises a clear error (pointing at ``make data``) when no images are found,
    rather than silently training on nothing.
    """
    root = Path(data_root) if data_root else (DATA_DIR / "potato")
    samples = discover_samples(root)
    if not samples:
        raise FileNotFoundError(
            f"No images found under {root}. Run `make data` to download the "
            f"PlantVillage potato subset, or pass data_root to a synthetic set."
        )

    parts = stratified_split(samples, val_frac, test_frac, split_seed)

    tf_train = build_transforms(img_size, train=True, aug_strength=aug_strength)
    tf_eval = build_transforms(img_size, train=False)

    datasets = {
        "train": PotatoLeafDataset(parts["train"], tf_train),
        "val": PotatoLeafDataset(parts["val"], tf_eval),
        "test": PotatoLeafDataset(parts["test"], tf_eval),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
        for split, ds in datasets.items()
    }

    train_counts = class_counts(parts["train"])
    meta = {
        "n_total": len(samples),
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "train_class_counts": train_counts,
        "classes": list(CLASSES),
        "img_size": img_size,
        "split_seed": split_seed,
    }
    log.info(
        "Data: %d train / %d val / %d test (split_seed=%d)",
        meta["n_train"], meta["n_val"], meta["n_test"], split_seed,
    )
    return loaders, meta


def compute_class_weights(train_counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency class weights (normalized to mean 1) for the loss.

    PlantVillage's potato classes are imbalanced (far fewer healthy leaves), so we
    up-weight rare classes to keep the from-scratch model from collapsing to the
    majority class.
    """
    counts = np.array([max(train_counts[c], 1) for c in CLASSES], dtype=np.float64)
    weights = counts.sum() / (len(counts) * counts)  # inverse frequency
    weights = weights / weights.mean()  # normalize to mean 1 (keeps loss scale stable)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Synthetic dataset (tests / CI / smoke runs — no download required)
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(
    root: str | Path,
    n_per_class: int = 30,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Generate a small, *learnable* synthetic potato-leaf stand-in.

    Each class gets a distinct, deterministic visual signature so that a model can
    actually learn something (making smoke tests meaningful) without needing the
    real dataset:

    * healthy      — smooth green leaf
    * early blight — green leaf with bright concentric (bullseye) spots
    * late blight  — green leaf with dark irregular water-soaked lesions

    Returns the dataset root (``root/<class>/<img>.png``).
    """
    root = Path(root)
    rng = np.random.default_rng(seed)

    def base_leaf(local_rng) -> np.ndarray:
        # Green-dominant background with mild texture.
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[..., 1] = 0.45 + 0.1 * local_rng.standard_normal((size, size))  # green
        img[..., 0] = 0.20 + 0.05 * local_rng.standard_normal((size, size))  # red
        img[..., 2] = 0.15 + 0.05 * local_rng.standard_normal((size, size))  # blue
        return img

    yy, xx = np.mgrid[0:size, 0:size]

    for label, cls in enumerate(CLASSES):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            local_rng = np.random.default_rng(seed * 1000 + label * 100 + i)
            img = base_leaf(local_rng)

            if label == 1:  # early blight — bright concentric rings
                for _ in range(local_rng.integers(2, 5)):
                    cy, cx = local_rng.integers(0, size, 2)
                    r = local_rng.integers(4, 10)
                    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                    ring = (np.cos(d / max(r, 1) * 3.0) > 0.3) & (d < r * 2)
                    img[ring] = [0.55, 0.40, 0.20]  # tan bullseye
            elif label == 2:  # late blight — dark irregular lesions
                for _ in range(local_rng.integers(2, 5)):
                    cy, cx = local_rng.integers(0, size, 2)
                    r = local_rng.integers(5, 12)
                    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                    blob = d < (r + 2 * local_rng.standard_normal())
                    img[blob] = [0.12, 0.10, 0.10]  # dark water-soaked

            img = np.clip(img, 0, 1)
            Image.fromarray((img * 255).astype(np.uint8)).save(cls_dir / f"{cls}_{i:03d}.png")

    log.info("Synthetic dataset written to %s (%d imgs/class).", root, n_per_class)
    return root
