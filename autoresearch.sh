#!/usr/bin/env bash
# Autoresearch driver: seeded random search over the from-scratch CNN's
# hyperparameters, selecting on validation macro-F1.
#
# Usage: ./autoresearch.sh [--trials N] [--epochs E]
set -euo pipefail

cd "$(dirname "$0")"

# Fast pre-check: the search script must at least import cleanly (< 1s).
uv run python -c "import ast,sys; ast.parse(open('scripts/autoresearch_loop.py').read())"

exec uv run python scripts/autoresearch_loop.py "$@"
