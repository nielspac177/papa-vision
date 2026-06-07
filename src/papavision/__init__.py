"""papavision — Lightweight CNNs for potato leaf disease diagnosis.

A small, fully reproducible research package comparing a from-scratch CNN against
ImageNet-pretrained transfer-learning backbones for 3-class potato leaf disease
classification (healthy / early blight / late blight), with statistically rigorous
evaluation and Grad-CAM interpretability. Designed to train on commodity hardware
(CPU / Apple Silicon MPS, no GPU required).

Author: Niels Pacheco.
"""

__version__ = "1.0.0"
__author__ = "Niels Pacheco"

# Canonical class ordering used everywhere (data, models, metrics, figures).
CLASSES = ("Potato___healthy", "Potato___Early_blight", "Potato___Late_blight")
CLASS_LABELS = ("Healthy", "Early blight", "Late blight")
