ARG DOCKER_PROXY_PREFIX
FROM ${DOCKER_PROXY_PREFIX}python:3.10-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN touch README.md
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev

FROM ${DOCKER_PROXY_PREFIX}python:3.10-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV REGULARIZER_CONFIG="/etc/regularizer/config.yml"
ENV REGULARIZER_ENV="/etc/regularizer/default.env"

COPY --from=builder /app/.venv /app/.venv
COPY docker/etc/regularizer /etc/regularizer
COPY regularizer ./regularizer

ENTRYPOINT ["python", "-m", "regularizer"]
