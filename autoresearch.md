# Autoresearch: maximize the from-scratch CNN's validation macro-F1

## Objective
Optimize the hyperparameters of the **from-scratch `custom_cnn`** so it classifies
potato leaf disease (healthy / early blight / late blight) as well as possible —
*without* using pretrained weights. This is the core scientific question of the
project: how close can a small, scratch-trained CNN get to transfer learning?

We search over learning rate, weight decay, network width, dropout, and data-
augmentation strength, selecting purely on the **validation** split so the test set
remains untouched for final, unbiased reporting.

## Metrics
- **Primary**: `val_macro_f1` (higher is better) — macro-averaged F1 on the
  validation split (robust to the class imbalance, unlike raw accuracy).
- **Secondary**: validation loss (monitoring only).

## How to Run
`./autoresearch.sh --trials 12 --epochs 10`

Runs a seeded random search via `scripts/autoresearch_loop.py`, trains each
candidate on `data/potato`, and writes:
- `results/metrics/autoresearch.json` — full trajectory + winner
- `autoresearch.jsonl` — per-trial state in the autoresearch protocol
- `autoresearch-dashboard.md` — human-readable leaderboard
- `experiments/worklog.md` — narrative log + insights

## Files in Scope
- `configs/custom_cnn.yaml` — the config that receives the winning hyperparameters.
- `scripts/autoresearch_loop.py` — the search engine + search space.

## Off Limits
- The test split (never used for selection).
- `src/papavision/data.py` splitting logic (the fixed `split_seed=42` guarantees a
  shared, stable test set across all experiments).
- The transfer-learning configs (the from-scratch model is what we're tuning).

## Constraints
- No pretrained weights for `custom_cnn` (`pretrained: false`).
- The unit test suite must keep passing (`make test`).
- Selection on validation macro-F1 only — no peeking at the test set.

## What's Been Tried
See `experiments/worklog.md` (auto-updated by the loop) and
`autoresearch-dashboard.md` for the leaderboard. Summary of insights is appended
there after the search completes.
