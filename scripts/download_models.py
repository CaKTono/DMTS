#!/usr/bin/env python3
"""Download DMTS model checkpoints from HuggingFace."""

import argparse
import os
import sys

# Allow importing config from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def model_exists(path):
    """Check if a model directory exists and has files."""
    return path.is_dir() and any(path.iterdir())


def main():
    parser = argparse.ArgumentParser(description="Download DMTS models from HuggingFace")
    parser.add_argument(
        "--backend",
        choices=["nllb", "hunyuan", "hybrid"],
        default="hybrid",
        help="Which backend to download models for (default: hybrid = all models)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=f"Target directory for models (default: {config.MODELS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if they already exist",
    )
    args = parser.parse_args()

    models_dir = config.MODELS_DIR
    if args.models_dir:
        from pathlib import Path
        models_dir = Path(args.models_dir)

    models_dir.mkdir(parents=True, exist_ok=True)

    required = config.BACKEND_MODELS[args.backend]

    print(f"\nDMTS Model Download (backend: {args.backend})")
    print(f"Target directory: {models_dir}\n")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run:   pip install huggingface_hub")
        sys.exit(1)

    skipped = 0
    downloaded = 0

    for key in required:
        repo_id = config.HF_REPOS.get(key)
        if not repo_id:
            continue

        local_dir = models_dir / config.DEFAULT_MODELS[key].name

        if model_exists(local_dir) and not args.force:
            print(f"  SKIP  {key} — already exists at {local_dir.name}/")
            skipped += 1
            continue

        print(f"  DOWNLOADING  {key} ({repo_id}) -> {local_dir.name}/")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
            )
            downloaded += 1
            print(f"  DONE  {key}")
        except Exception as e:
            print(f"  ERROR  {key}: {e}")
            sys.exit(1)

    print(f"\nFinished: {downloaded} downloaded, {skipped} skipped (already exist).")
    print("Next: python scripts/preflight.py\n")


if __name__ == "__main__":
    main()
