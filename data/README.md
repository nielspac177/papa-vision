# Data

This directory holds the **PlantVillage potato subset** used by the experiments.
The images are **not** committed to the repository (see `.gitignore`); run the
download script to populate it:

```bash
make data          # or: uv run python scripts/download_data.py
```

## Expected layout after download

```
data/
└── potato/
    ├── Potato___healthy/        *.jpg
    ├── Potato___Early_blight/   *.jpg
    └── Potato___Late_blight/    *.jpg
```

## Source

The PlantVillage dataset (Hughes & Salathé, 2015, *arXiv:1511.08060*) is a public
collection of leaf images photographed under controlled laboratory conditions.
The download script fetches the three potato classes from the Hugging Face Hub,
falling back to a GitHub mirror of the original dataset if the Hub is unavailable.

Approximate class counts (subject to the exact mirror):

| Class                  | Images |
|------------------------|--------|
| `Potato___healthy`     | ~152   |
| `Potato___Early_blight`| ~1000  |
| `Potato___Late_blight` | ~1000  |

## Important limitation

PlantVillage images were captured against fairly uniform backgrounds. Models
trained on it can latch onto **background cues** rather than leaf lesions, which
inflates accuracy and harms field generalization. We probe this directly with
Grad-CAM (see the paper's Discussion). Treat reported accuracies as an
*upper bound* on real-world field performance.

## License

The PlantVillage dataset is distributed by its authors for research use. Please
consult the original release for terms. This repository redistributes **no**
image data.
