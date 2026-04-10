#!/usr/bin/env python3
"""
Pre-download all ML models during Docker build to avoid runtime downloads.
This script runs during Docker build phase to cache all models.

Usage: python download_models.py
"""
import os
import sys

def download_huggingface_models():
    """Download all required HuggingFace models."""
    print("=" * 60)
    print("Downloading HuggingFace Models...")
    print("=" * 60)
    
    models_to_download = [
        # Embeddings (HarryAgent)
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence_transformers"),
        # Cross-Encoder reranker (HarryAgent)
        ("cross-encoder/ms-marco-TinyBERT-L-2-v2", "sentence_transformers"),
    ]
    
    for model_name, lib in models_to_download:
        try:
            print(f"\n📥 Downloading: {model_name}")
            if lib == "sentence_transformers":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
            print(f"✅ Successfully downloaded: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to download {model_name}: {e}")

def download_ultralytics_models():
    """Download YOLO models used in LogoYolo."""
    print("\n" + "=" * 60)
    print("Downloading Ultralytics YOLO Models...")
    print("=" * 60)
    
    yolo_models = [
        'yolov8n.pt',      # YOLOv8 nano (commonly used)
        'yolov8s.pt',      # YOLOv8 small
        'yolov8m.pt',      # YOLOv8 medium
    ]
    
    try:
        from ultralytics import YOLO
        for model_file in yolo_models:
            print(f"\n📥 Downloading {model_file}...")
            model = YOLO(model_file)
            print(f"✅ Successfully downloaded {model_file}")
    except Exception as e:
        print(f"⚠️ Failed to download YOLO models: {e}")

def verify_downloads():
    """Verify models are cached."""
    print("\n" + "=" * 60)
    print("Verifying Model Cache...")
    print("=" * 60)
    
    cache_dirs = [
        os.environ.get("TRANSFORMERS_CACHE", "/app/.cache/huggingface"),
        os.environ.get("HF_HOME", "/app/.cache/huggingface"),
    ]
    
    for cache_dir in set(cache_dirs):
        if os.path.exists(cache_dir):
            print(f"\n✅ Cache exists at: {cache_dir}")
            import subprocess
            try:
                result = subprocess.run(["du", "-sh", cache_dir], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"📦 Cache size: {result.stdout.split()[0]}")
            except:
                print("📦 (could not determine size)")
        else:
            print(f"\n⚠️ Cache not found at: {cache_dir}")
    
    # List downloaded models
    print("\n📋 Cached models:")
    try:
        import subprocess
        result = subprocess.run(
            ["find", "/app/.cache/huggingface", "-name", "*.bin", "-o", "-name", "*.safetensors"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            for model_file in result.stdout.strip().split('\n')[:10]:
                print(f"  ✓ {model_file}")
        else:
            print("  (no models found in cache)")
    except Exception as e:
        print(f"  ⚠️ Could not list models: {e}")

def check_existing_files():
    """Check for existing model files in the project."""
    print("\n" + "=" * 60)
    print("Checking for existing model files...")
    print("=" * 60)
    
    important_files = [
        "HPVdb",
    ]
    
    for filepath in important_files:
        if os.path.exists(filepath):
            import subprocess
            result = subprocess.run(["du", "-sh", filepath], capture_output=True, text=True)
            size = result.stdout.split()[0] if result.returncode == 0 else "unknown"
            print(f"✅ {filepath} ({size})")
        else:
            print(f"⚠️ {filepath} not found (will be copied from build context)")

if __name__ == "__main__":
    check_existing_files()
    download_huggingface_models()
    download_ultralytics_models()
    verify_downloads()
    
    print("\n" + "=" * 60)
    print("✅ Model pre-download complete!")
    print("Models will be cached in the Docker image, avoiding runtime downloads.")
    print("=" * 60)

