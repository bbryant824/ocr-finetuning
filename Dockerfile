# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

FROM base AS controller
RUN pip install --no-cache-dir .
CMD ["active-ocr-controller"]

FROM base AS gpu
RUN pip install --no-cache-dir ".[gpu]"
CMD ["active-ocr-gpu"]
