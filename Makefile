.PHONY: install ingest silver gold train pipeline test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python -m flows.job --stage ingestion

silver:
	python -m flows.job --stage silver

gold:
	python -m flows.job --stage gold

train:
	python -m flows.job --stage train

pipeline:
	python -m flows.job --stage all

test:
	pytest -q

lint:
	ruff check .

clean:
	rm -rf data models .pytest_cachemake test