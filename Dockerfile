# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 python:3.11-slim AS base

# -------- STAGE 1: Builder --------
FROM base AS builder
WORKDIR /build

# Install binutils only for stripping
RUN apt-get update && apt-get install -y --no-install-recommends binutils && rm -rf /var/lib/apt/lists/*

# Copy and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile --prefix=/python -r requirements.txt

# Strip unused symbols from .so files
RUN find /python -name "*.so" -exec strip --strip-unneeded {} + || true

# -------- STAGE 2: Runtime --------
FROM base AS runtime
WORKDIR /app

# COPY stripped dependencies
COPY --from=builder /python /usr/local

# Copy application code
COPY extractor.py main.py .

# 🧹 Remove __pycache__, docs, and pip cache (extra cleanup)
RUN find /usr/local -type d -name '__pycache__' -exec rm -rf {} + && \
    rm -rf /usr/local/lib/python3.11/test /usr/share/doc /usr/share/man /root/.cache

CMD ["python", "main.py"]
