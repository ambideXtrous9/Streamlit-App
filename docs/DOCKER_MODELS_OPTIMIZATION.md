# Docker Model Pre-Download Optimization

## Problem
Previously, every time a Docker container started, it would download ML models on-the-fly:
- `sentence-transformers/all-MiniLM-L6-v2` (~90MB)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (~60MB)
- YOLO models (~10-50MB each)
- HPVdb vector database (~22MB)

This caused:
- **Slow startup times** (30s-2min depending on network)
- **Redundant downloads** on every container recreate
- **Network dependency** at runtime

## Solution

### Multi-Stage Docker Build
The Dockerfile now uses a **3-stage build**:

```
┌─────────────────────────────────────┐
│  Stage 1: builder                   │
│  - Install Python dependencies      │
│  - CPU-only PyTorch                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Stage 2: model-downloader          │
│  - Pre-download all ML models       │
│  - Cache in /app/.cache/huggingface │
│  - Strip binaries, clean cache      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Stage 3: runtime (final image)     │
│  - Copy venv from builder           │
│  - Copy models from model-downloader│
│  - Copy application code + HPVdb    │
│  - No runtime downloads needed!     │
└─────────────────────────────────────┘
```

### Key Optimizations

1. **Model Pre-Download During Build**
   - `download_models.py` runs during Docker build
   - All HuggingFace models cached to `/app/.cache/huggingface`
   - YOLO models downloaded and cached
   - HPVdb included via `COPY . .`

2. **Removed Unnecessary Volume Mounts**
   - Removed `hf-cache` volume (models now baked in)
   - Only `sqlite-data` persisted (for app data)

3. **CPU-Only PyTorch**
   - Uses `https://download.pytorch.org/whl/cpu`
   - Saves ~1-2GB by avoiding CUDA libraries

4. **Binary Stripping**
   - Strips shared libraries in venv
   - Removes `__pycache__` and `.pyc` files

## Models Pre-Downloaded

| Model | Size | Used By |
|-------|------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | ~90MB | HarryAgent (embeddings) |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | ~60MB | HarryAgent (reranker) |
| `yolov8n.pt` | ~6MB | LogoYolo (inference) |
| `yolov8s.pt` | ~22MB | LogoYolo (optional) |
| `yolov8m.pt` | ~52MB | LogoYolo (optional) |
| `HPVdb/` | ~22MB | HarryAgent (vector DB) |

**Total model size**: ~252MB (downloaded once during build)

## Benefits

### Before Optimization
```bash
$ docker-compose up
# First start: Download models (60-120s)
# Every restart: Download models again (60-120s)
```

### After Optimization
```bash
$ docker-compose up
# First start: Use cached models (instant)
# Every restart: Still instant (models in image)
```

**Startup time**: Reduced from 60-120s → 5-10s

## Files Changed

- `Dockerfile` - Added model-downloader stage + Node.js 20.x installation
- `download_models.py` - New script to pre-download models
- `.dockerignore` - Excluded `.rp360/` and `nodev20/` from build
- `docker-compose.yml` - Removed `hf-cache` volume
- `AirbnbAgent/tourAgent.py` - Simplified Node.js check (no longer uses bundled nodev20)

## Node.js for Airbnb MCP

The Docker image now includes **Node.js 20.x** installed via apt during build:

- ✅ **No bundled `nodev20/` folder** (excluded via `.dockerignore`)
- ✅ **Clean installation** via NodeSource repository
- ✅ **Verified at build time** (`node -v && npx --version`)
- ✅ **Automatic detection** at runtime (no apt install needed in Docker)

This enables the Airbnb MCP (`@openbnb/mcp-server-airbnb`) to work without:
- Bundling large Node.js binaries (~50MB)
- Runtime apt installation delays
- Platform-specific path handling

## Customization

To add more models to pre-download:

1. Edit `download_models.py` and add to `models_to_download` list
2. Rebuild: `docker-compose build --no-cache`

## Verification

Check models are cached in running container:
```bash
docker exec streamlit-ai-portfolio ls -lh /app/.cache/huggingface
docker exec streamlit-ai-portfolio du -sh /app/.cache/huggingface
```
