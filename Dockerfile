# ==============================================================================
# BUILDER STAGE
# ==============================================================================
FROM python:3.12-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=100

WORKDIR /app

# Install build dependencies (only what's needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    pkg-config \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/

# Create virtual environment
RUN /uv/bin/uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for maximum Docker layer caching
COPY requirements.txt .

# Install Python dependencies with uv
# Use CPU-only torch to avoid NVIDIA CUDA bloat
RUN /uv/bin/uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    /uv/bin/uv pip install --no-cache torchvision --index-url https://download.pytorch.org/whl/cpu && \
    /uv/bin/uv pip install --no-cache -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Strip binaries and clean up Python cache
RUN find /opt/venv -name '*.so' -type f -exec strip --strip-unneeded '{}' + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && \
    find /opt/venv -name "*.pyc" -delete 2>/dev/null

# ==============================================================================
# RUNTIME STAGE
# ==============================================================================
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y --auto-remove && \
    apt-get clean

# Copy only the optimized virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]
