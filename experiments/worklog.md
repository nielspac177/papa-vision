# Autoresearch Worklog: custom_cnn hyperparameter search

Random search, 10 trials x 10 epochs, search_seed=7. Metric: validation macro-F1 (higher better).

### Run 1: {'lr': 0.003, 'weight_decay': 0.0001, 'width': 48, 'dropout': 0.5, 'aug_strength': 1.0} — val_macro_f1=0.9410 (KEEP)
- Result: 0.9410 (delta vs baseline +0.0%)

### Run 2: {'lr': 0.003, 'weight_decay': 0.001, 'width': 24, 'dropout': 0.2, 'aug_strength': 0.5} — val_macro_f1=0.9196 (DISCARD)
- Result: 0.9196 (delta vs baseline -2.3%)

### Run 3: {'lr': 0.0003, 'weight_decay': 0.001, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.0} — val_macro_f1=0.9677 (KEEP)
- Result: 0.9677 (delta vs baseline +2.8%)

### Run 4: {'lr': 0.003, 'weight_decay': 1e-05, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.0} — val_macro_f1=0.9308 (DISCARD)
- Result: 0.9308 (delta vs baseline -1.1%)

### Run 5: {'lr': 0.003, 'weight_decay': 1e-05, 'width': 32, 'dropout': 0.2, 'aug_strength': 1.5} — val_macro_f1=0.9336 (DISCARD)
- Result: 0.9336 (delta vs baseline -0.8%)

### Run 6: {'lr': 0.0003, 'weight_decay': 0.001, 'width': 32, 'dropout': 0.3, 'aug_strength': 1.0} — val_macro_f1=0.9500 (DISCARD)
- Result: 0.9500 (delta vs baseline +1.0%)

### Run 7: {'lr': 0.001, 'weight_decay': 0.0001, 'width': 32, 'dropout': 0.5, 'aug_strength': 1.5} — val_macro_f1=0.9448 (DISCARD)
- Result: 0.9448 (delta vs baseline +0.4%)

### Run 8: {'lr': 0.003, 'weight_decay': 0.001, 'width': 32, 'dropout': 0.3, 'aug_strength': 1.5} — val_macro_f1=0.9312 (DISCARD)
- Result: 0.9312 (delta vs baseline -1.0%)

### Run 9: {'lr': 0.001, 'weight_decay': 1e-05, 'width': 48, 'dropout': 0.2, 'aug_strength': 1.5} — val_macro_f1=0.9487 (DISCARD)
- Result: 0.9487 (delta vs baseline +0.8%)

### Run 10: {'lr': 0.001, 'weight_decay': 1e-05, 'width': 24, 'dropout': 0.3, 'aug_strength': 0.5} — val_macro_f1=0.9687 (KEEP)
- Result: 0.9687 (delta vs baseline +2.9%)

## Key Insights
- Best configuration (trial #10): `{'lr': 0.001, 'weight_decay': 1e-05, 'width': 24, 'dropout': 0.3, 'aug_strength': 0.5}` reaching val macro-F1 = 0.9687.
- These winning hyperparameters are folded into `configs/custom_cnn.yaml` for the
  final 3-seed evaluation runs.

