# ==============================================================================
# BUILDER STAGE
# ==============================================================================
FROM python:3.12-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=100 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

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

# ==============================================================================
# MODEL DOWNLOAD STAGE - Pre-download all ML models
# ==============================================================================
FROM builder AS model-downloader

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Download all required models during build (not at runtime)
COPY download_models.py .
RUN python download_models.py || echo "⚠️ Some models failed to download (will download at runtime)"

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
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install runtime dependencies + Node.js 20.x (required for Airbnb MCP via npx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove \
    && apt-get clean \
    && npm cache clean --force

# Verify Node.js and npx are available
RUN node -v && npx --version

# Copy only the optimized virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy pre-downloaded models from model downloader stage
COPY --from=model-downloader /app/.cache/huggingface /app/.cache/huggingface

# Copy application code (including HPVdb)
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app/.cache

USER appuser

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]
