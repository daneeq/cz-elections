# CZ Elections Live

Live dashboard + Monte Carlo forecast for Czech Parliamentary Elections 2025.
Built with Python/Streamlit and Poetry (Python ≥ 3.11).

## Quickstart

```bash
# 1) Install Poetry (https://python-poetry.org/docs/#installation)
# 2) In repo root:
poetry env use 3.11
poetry install

cp .env.example .env
# Edit DATA_MODE in .env: ps2021_fixture (default) or ps2025_live

# 3) Run
poetry run streamlit run src/cz_elections_live/streamlit_app.py
# or
poetry run cz-live
```

## Modes
- **ps2021_fixture**: uses synthetic partial counts generated from sample 2021-like data in `data/raw/`.
- **ps2025_live**: parses live XML from volby.cz (set `VOLBY2025_TOPLINE` in `.env` if needed).

## Structure
See repo tree in the project root. Edit coalition presets in `config/coalitions.yaml`.
