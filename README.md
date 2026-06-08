# papa-vision 🥔🔬

**Lightweight convolutional neural networks for potato leaf disease diagnosis on
commodity hardware: a from-scratch versus transfer-learning study with Grad-CAM
interpretability.**

Author: **Niels Pacheco** · MIA-07 *Redes Neuronales y Aprendizaje Profundo* (Sección C) · Final project, 2026.

[![CI](https://github.com/nielspacheco1997/papa-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/nielspacheco1997/papa-vision/actions)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Late blight (*tizón tardío*, *Phytophthora infestans*) and early blight
(*Alternaria solani*) are the two most economically damaging foliar diseases of
the potato (*Solanum tuberosum*) — the staple crop domesticated in the Peruvian
Andes. This repository trains and rigorously compares convolutional neural
networks that classify a potato leaf as **healthy**, **early blight**, or
**late blight** from a single RGB photograph.

The study asks a practical question: *how well can a small CNN trained from
scratch on a GPU-less laptop compete with ImageNet-pretrained transfer-learning
backbones?* — and answers it with honest statistics (multiple seeds, bootstrap
confidence intervals, McNemar's test), calibration analysis, and **Grad-CAM**
interpretability that probes the well-documented *background-bias* pitfall of the
PlantVillage dataset.

Everything runs on **CPU or Apple Silicon (MPS)** — no GPU required.

### Models compared

| Key | Architecture | Source | Params (approx.) |
|-----|--------------|--------|------------------|
| `custom_cnn`      | 4-block CNN built from scratch | this repo | ~0.33 M |
| `mobilenet_v2`    | MobileNetV2 | ImageNet-pretrained, frozen backbone + fine-tuned head | ~2.2 M |
| `resnet18`        | ResNet-18 | ImageNet-pretrained, frozen backbone + fine-tuned head | ~11 M |
| `efficientnet_b0` | EfficientNet-B0 | ImageNet-pretrained, frozen backbone + fine-tuned head | ~4 M |

### Headline result (test set, mean ± SD over 3 seeds)

| Model | Params | Accuracy (%) | Macro-F1 (%) | ECE |
|-------|-------:|:------------:|:------------:|:---:|
| **Custom CNN (scratch)** | **328,587** | **97.3 ± 2.3** | **96.7 ± 2.1** | 0.123 |
| EfficientNet-B0 | 4,011,391 | 94.9 ± 0.5 | 91.7 ± 1.4 | 0.157 |
| MobileNetV2 | 2,227,715 | 94.9 ± 0.8 | 91.6 ± 1.2 | 0.134 |
| ResNet-18 | 11,178,051 | 91.7 ± 1.5 | 86.5 ± 2.1 | 0.171 |

The from-scratch CNN **significantly outperforms** all three frozen-backbone transfer
baselines (paired McNemar *p* = 0.017 / 0.004 / 7×10⁻⁵) with **7–34× fewer
parameters** — under an equal low-compute budget, end-to-end domain-specific features
beat frozen ImageNet features. Grad-CAM shows all models partly attend to background
(PlantVillage's known bias), so these lab accuracies are an upper bound on field
performance. Numbers are auto-generated from `results/metrics/` — see the paper.

---

## Reproducible quickstart

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). Python 3.12 is
fetched automatically and the exact dependency versions are frozen in `uv.lock`.

```bash
# 1. Build the environment (Python 3.12 + torch/torchvision, MPS-enabled)
make setup

# 2. Run the unit tests on a tiny synthetic dataset (no download needed)
make test

# 3. Download the PlantVillage potato subset (~3 classes, ~2,150 images)
make data

# 4. Train every model across 3 seeds (CPU/MPS; ~1-2 h total)
make train-all

# 5. Compute metrics, run statistical tests, and render all figures
make eval
make figures

# 6. Build the paper (PDF) and the slide deck
make paper
make slides
```

Single-model training:

```bash
make train CONFIG=configs/custom_cnn.yaml
```

Or call the package directly inside the environment:

```bash
uv run python -m papavision.train --config configs/resnet18.yaml --seed 0
```

---

## Repository layout

```
papa-vision/
├── configs/            # one YAML per model (hyperparameters, augmentation)
├── src/papavision/     # the installable package (data, models, train, eval, gradcam, stats, utils)
├── scripts/            # download_data, autoresearch_loop, make_figures, make_paper_tables
├── tests/              # pytest suite (runs on synthetic data, no network)
├── paper/              # Nature Scientific Reports-style LaTeX manuscript
├── slides/             # Marp presentation deck
├── results/            # metrics (JSON) + figures (PNG) — real, committed numbers
└── notebooks/          # end-to-end demo
```

---

## Reproducibility guarantees

- **Pinned environment** — `uv.lock` freezes every transitive dependency.
- **Fixed seeds** — Python / NumPy / PyTorch RNGs are seeded; each result is the
  mean ± 95% CI over 3 seeds.
- **Config-driven** — no magic numbers in code; every run is fully described by a
  YAML file under `configs/`.
- **Leakage-safe splits** — stratified train/val/test partition with a fixed seed;
  the test set is never touched during model selection.
- **Committed artifacts** — `results/metrics/` (per-run and aggregate JSON, test
  predictions) and `results/figures/` (rendered PNGs) are committed, so the paper's
  tables and plots are reproducible from the repository. Model checkpoints are *not*
  committed (size); regenerate them with `make train-all`, after which `make figures`
  reproduces every plot — including the Grad-CAM panel — from scratch.
- **CI** — GitHub Actions runs the test suite on every push using a synthetic
  dataset, so the pipeline is verified without large downloads.

---

## Data

The PlantVillage potato subset is **not** redistributed in this repository. The
`make data` target downloads it from the Hugging Face Hub (with a GitHub mirror
fallback) into `data/potato/`. See `data/README.md` for details and licensing.

---

## Citation

```bibtex
@misc{pacheco2026papavision,
  author = {Niels Pacheco},
  title  = {papa-vision: Lightweight CNNs for potato leaf disease diagnosis on commodity hardware},
  year   = {2026},
  note   = {MIA-07 Redes Neuronales y Aprendizaje Profundo, Final Project},
  url    = {https://github.com/nielspacheco1997/papa-vision}
}
```

## License

[MIT](LICENSE) © 2026 Niels Pacheco.
