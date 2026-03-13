#!/bin/bash
# Setup script for DMTS conda environment
# Usage: conda activate dmts && bash setup_env.sh

export PYTHONNOUSERSITE=1

# --- Pre-flight checks ---
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda environment active."
    echo "Run:  conda create -n dmts python=3.10 -y && conda activate dmts"
    exit 1
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "ERROR: Python $PYTHON_VERSION detected. Python 3.10 is required for TTS compatibility."
    echo "Run:  conda create -n dmts python=3.10 -y && conda activate dmts"
    exit 1
fi

set -e

echo "[1/4] Installing PyTorch CUDA 12.8 wheels via pip..."
# The conda PyTorch 2.5.1 + MKL/OpenMP stack pulled into fresh envs on this machine
# fails at import time with: undefined symbol: iJIT_NotifyEvent.
# Match the known-good realtimestt environment instead.
python -m pip install --force-reinstall \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[2/4] Installing pinned NumPy and PyAudio..."
# Torch wheels pull in numpy 2.x, but DMTS currently needs numpy 1.23.5 for the
# pandas/scikit-learn/TTS stack. Install PyAudio explicitly so clean envs do not
# depend on user-site packages leaked in via RealtimeSTT.
python -m pip install --force-reinstall numpy==1.23.5 PyAudio==0.2.14

echo "[3/4] Installing TTS==0.22.0 (--no-deps to bypass overly strict numpy==1.22.0 pin)..."
# TTS 0.22.0 metadata pins numpy==1.22.0 on Python 3.10, but it works fine with 1.23.5.
# All required TTS transitive deps are listed in requirements.txt.
python -m pip install TTS==0.22.0 --no-deps

echo "[4/4] Installing all remaining requirements..."
python -m pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download models:   python scripts/download_models.py"
echo "  2. Verify everything: python scripts/preflight.py"
echo "  3. Start server:      bash run_server.sh"
