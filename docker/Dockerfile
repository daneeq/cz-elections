FROM python:3.11-slim

ENV POETRY_VERSION=1.8.3     POETRY_VIRTUALENVS_CREATE=false     PIP_NO_CACHE_DIR=off     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN pip install --upgrade pip && pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml ./
COPY README.md ./
RUN poetry install --no-root --only main

COPY . .

EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "src/cz_elections_live/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
