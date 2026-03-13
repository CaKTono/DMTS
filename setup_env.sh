#!/bin/bash
# Setup script for DMTS conda environment
# Usage: conda activate dmts && bash setup_env.sh

set -e

echo "[1/4] Installing PyTorch with CUDA 12.8..."
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "[2/4] Installing numpy 1.23.5..."
pip install numpy==1.23.5

echo "[3/4] Installing TTS==0.22.0 (--no-deps to bypass overly strict numpy==1.22.0 pin)..."
# TTS 0.22.0 metadata pins numpy==1.22.0 on Python 3.10, but it works fine with 1.23.5.
# All required TTS transitive deps are listed in requirements.txt.
pip install TTS==0.22.0 --no-deps

echo "[4/4] Installing all remaining requirements..."
pip install -r requirements.txt

echo "Done. Test with: python dmts_mk4.py --help"
