# Autoresearch Dashboard: custom_cnn_hpo

**Runs:** 10 | **Kept:** 3 | **Discarded:** 7
**Baseline:** val_macro_f1: 0.9410 (#1)
**Best:** val_macro_f1: 0.9687 (#10, +2.9%)

| # | val_macro_f1 | status | hyperparameters |
|---|--------------|--------|-----------------|
| 1 | 0.9410 (+0.0%) | keep | `{'lr': 0.003, 'weight_decay': 0.0001, 'width': 48, 'dropout': 0.5, 'aug_strength': 1.0}` |
| 2 | 0.9196 (-2.3%) | discard | `{'lr': 0.003, 'weight_decay': 0.001, 'width': 24, 'dropout': 0.2, 'aug_strength': 0.5}` |
| 3 | 0.9677 (+2.8%) | keep | `{'lr': 0.0003, 'weight_decay': 0.001, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.0}` |
| 4 | 0.9308 (-1.1%) | discard | `{'lr': 0.003, 'weight_decay': 1e-05, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.0}` |
| 5 | 0.9336 (-0.8%) | discard | `{'lr': 0.003, 'weight_decay': 1e-05, 'width': 32, 'dropout': 0.2, 'aug_strength': 1.5}` |
| 6 | 0.9500 (+1.0%) | discard | `{'lr': 0.0003, 'weight_decay': 0.001, 'width': 32, 'dropout': 0.3, 'aug_strength': 1.0}` |
| 7 | 0.9448 (+0.4%) | discard | `{'lr': 0.001, 'weight_decay': 0.0001, 'width': 32, 'dropout': 0.5, 'aug_strength': 1.5}` |
| 8 | 0.9312 (-1.0%) | discard | `{'lr': 0.003, 'weight_decay': 0.001, 'width': 32, 'dropout': 0.3, 'aug_strength': 1.5}` |
| 9 | 0.9487 (+0.8%) | discard | `{'lr': 0.001, 'weight_decay': 1e-05, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.5}` |
| 10 | 0.9687 (+2.9%) | keep | `{'lr': 0.001, 'weight_decay': 1e-05, 'width': 24, 'dropout': 0.3, 'aug_strength': 0.5}` |
