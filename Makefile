.PHONY: setup run run-script test lint format type

setup:
	poetry env use 3.11
	poetry install

run:
	poetry run streamlit run src/cz_elections_live/streamlit_app.py

run-script:
	poetry run cz-live

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run black .

type:
	poetry run mypy src

