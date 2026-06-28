# Reproduce the analysis end to end. Each target is idempotent and reads/writes
# parquet checkpoints. The default `make analyze`/`make test` path is OFFLINE and
# uses the committed sample, so CI needs no network. `make data` (network) pulls
# the real catalogues for a science run.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: all venv install asset sample analyze completeness forecast figures test lint data clean

all: analyze completeness forecast figures

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

asset:
	$(PY) scripts/make_bergeron_asset.py

# --- Offline (sample-based) reproduction: the CI gate ----------------------
sample:
	$(PY) -m seti.cli make-sample

analyze: sample
	$(PY) -m seti.cli analyze

completeness: sample
	$(PY) -m seti.cli completeness

forecast:
	$(PY) -m seti.cli forecast

contamination-budget:
	$(PY) -m seti.cli contamination-budget

paper-numbers:
	$(PY) -m seti.cli paper-numbers

figures: analyze forecast contamination-budget
	$(PY) -m seti.cli figures

# Build the manuscript PDF (requires a LaTeX toolchain). Regenerates numbers and
# figures first so the paper is always in sync with the code.
paper: paper-numbers figures
	cd paper && latexmk -pdf -bibtex -interaction=nonstopmode main.tex

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

# --- Online (real-catalogue) science run -----------------------------------
# Pulls Gaia WD x CatWISE2020 x 2MASS + control catalogues into data/cache,
# then runs the same analysis on the real analysis-ready table. Network + time.
data:
	$(PY) -m seti.acquire_run   # see src/seti/acquire/* ; wire up for science runs

clean:
	rm -rf data/cache data/interim data/processed results/tables/*.parquet \
	       results/figures/*.pdf
