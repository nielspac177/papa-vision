# papa-vision — reproducible build automation.
# Every target runs inside the pinned uv environment (Python 3.12 + torch/MPS).

PY      := uv run python
CONFIGS := configs/custom_cnn.yaml configs/mobilenet_v2.yaml configs/resnet18.yaml configs/efficientnet_b0.yaml
SEEDS   := 0 1 2
CONFIG  ?= configs/custom_cnn.yaml   # default for `make train`

.PHONY: help setup test data train train-all eval figures autoresearch paper slides clean all

help:
	@echo "papa-vision targets:"
	@echo "  setup       Build the uv environment (Python 3.12 + torch)"
	@echo "  test        Run the pytest suite on synthetic data"
	@echo "  data        Download the PlantVillage potato subset"
	@echo "  train       Train one model        (CONFIG=configs/<model>.yaml)"
	@echo "  train-all   Train every model x every seed"
	@echo "  eval        Aggregate metrics + run statistical tests"
	@echo "  figures     Render all publication figures"
	@echo "  autoresearch  Run the hyperparameter-search loop"
	@echo "  paper       Compile the Nature-style PDF"
	@echo "  slides      Build the Marp slide deck"
	@echo "  all         data -> train-all -> eval -> figures -> paper"

setup:
	uv python pin 3.12
	uv sync --extra dev --extra download

test:
	uv run pytest

data:
	$(PY) scripts/download_data.py

# Train a single model (override CONFIG=...). Loops over all seeds.
train:
	@for s in $(SEEDS); do \
		echo ">>> $(CONFIG) seed=$$s"; \
		$(PY) -m papavision.train --config $(CONFIG) --seed $$s; \
	done

# Full experimental grid: every architecture x every seed.
train-all:
	@for c in $(CONFIGS); do \
		for s in $(SEEDS); do \
			echo ">>> $$c seed=$$s"; \
			$(PY) -m papavision.train --config $$c --seed $$s; \
		done; \
	done

eval:
	$(PY) -m papavision.evaluate --all

figures:
	$(PY) scripts/make_figures.py

autoresearch:
	$(PY) scripts/autoresearch_loop.py --trials 10

paper:
	$(MAKE) -C paper

slides:
	@command -v marp >/dev/null 2>&1 \
		&& marp slides/presentation.md --pdf --allow-local-files -o slides/presentation.pdf \
		|| echo "marp not installed — view slides/presentation.md directly or 'npm i -g @marp-team/marp-cli'"

clean:
	rm -rf results/checkpoints
	$(MAKE) -C paper clean

all: data train-all eval figures paper
