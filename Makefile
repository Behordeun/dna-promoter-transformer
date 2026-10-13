.PHONY: setup data ecoli human test lint

setup:
	uv venv .venv --python 3.11
	uv pip install --python .venv/bin/python -e ".[dev]"

data:
	mkdir -p data
	curl -sL "https://archive.ics.uci.edu/static/public/67/molecular+biology+promoter+gene+sequences.zip" -o data/uci67.zip
	cd data && unzip -o uci67.zip >/dev/null

ecoli:
	.venv/bin/python -m src.train

human:
	.venv/bin/python -m src.train_human

test:
	.venv/bin/python -m pytest tests/ -q

lint:
	.venv/bin/ruff check src tests
