FROM python:3.12-slim

LABEL maintainer="streamlit-ai-portfolio"

# Install system deps + Node.js 20 (native for linux/amd64 or linux/arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for maximum Docker layer caching
COPY requirements.txt .

# Install Python deps with uv
RUN uv pip install --system --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -exec rm -rf {} + \
    || true

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=true
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]
