# The builder image, used to build the virtual environment
FROM python:3.12-slim-bookworm as builder

RUN apt-get update && apt-get install -y git build-essential

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8080
EXPOSE 8080

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.12-slim-bookworm as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
RUN pip install poetry

WORKDIR /app

COPY ./src ./src
COPY ./.chainlit ./.chainlit
COPY ./config.yaml ./
COPY chainlit.md ./
COPY ./pyproject.toml ./

ENV PYTHONPATH=/app

CMD ["poetry", "run", "chainlit", "run", "/app/src/app.py", "-w", "-h", "--host", "0.0.0.0", "--port", "8080", "--debug"]
