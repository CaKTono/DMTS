# DMTS MK4 - Linux Server Setup Guide

This guide covers setting up DMTS MK4 on a Linux server with CUDA support.

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: ~20GB for models

### Software Requirements
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA drivers + CUDA 11.8+
- Python 3.10+
- Conda (recommended for environment management)

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DMTS.git
cd DMTS
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n dmts python=3.10
conda activate dmts

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Models

You need to download the following models:

#### Whisper Models (Transcription)
```bash
# Create models directory
mkdir -p /path/to/models

# Download faster-whisper-large-v3
# From: https://huggingface.co/Systran/faster-whisper-large-v3

# Download faster-whisper-large-v3-turbo-ct2 (for real-time/verification)
# From: https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2
```

#### Diarization Model (Speaker ID)
```bash
# Download XTTS-v2 for speaker embeddings
# From: https://huggingface.co/coqui/XTTS-v2
```

#### Translation Models
```bash
# NLLB models (for NLLB/Hybrid backends)
# - nllb-200-distilled-600M: https://huggingface.co/facebook/nllb-200-distilled-600M
# - nllb-200-3.3B: https://huggingface.co/facebook/nllb-200-3.3B

# Hunyuan model (for Hunyuan/Hybrid backends)
# - Hunyuan-MT-7B: https://huggingface.co/tencent/Hunyuan-MT-7B
```

### 4. Configure Model Paths

Edit the run script (e.g., `run_server_hybrid.sh`) to set your model paths:

```bash
# ============================================================
# MODEL PATH CONFIGURATION - EDIT THESE PATHS
# ============================================================
WHISPER_MODEL="/path/to/models/faster-whisper-large-v3"
WHISPER_MODEL_REALTIME="/path/to/models/faster-whisper-large-v3-turbo-ct2"
VERIFICATION_MODEL="/path/to/models/faster-whisper-large-v3-turbo-ct2"
DIARIZATION_MODEL="/path/to/models/XTTS-v2/v2.0.2"
NLLB_600M="/path/to/models/nllb-200-distilled-600M"
NLLB_3_3B="/path/to/models/nllb-200-3.3B"
HUNYUAN_MODEL="/path/to/models/Hunyuan-MT-7B"
```

### 5. Run the Server

```bash
# Make script executable
chmod +x run_server_hybrid.sh

# Run with hybrid backend (recommended)
./run_server_hybrid.sh

# Or with specific backends:
./run_server_nllb.sh      # NLLB only (fast, 200+ languages)
./run_server_hunyuan.sh   # Hunyuan only (high quality, 38 languages)
```

### 6. Access the Web Interface

Open your browser to: `http://localhost:8890`

---

## Configuration

### Translation Backends

| Backend | Script | Speed | Languages | VRAM |
|---------|--------|-------|-----------|------|
| Hybrid (Recommended) | `run_server_hybrid.sh` | Medium | 200+ | ~16GB |
| NLLB | `run_server_nllb.sh` | Fast | 200+ | ~8GB |
| Hunyuan | `run_server_hunyuan.sh` | Slow | 38 | ~14GB |

### Key Parameters

```bash
# Server
--port 8890                    # WebSocket port

# Device
--device "cuda"                # Use GPU (or "cpu")
--compute_type "float16"       # Precision (int8, float16, float32)

# Translation
--translation_target_language "eng_Latn"  # Target language (NLLB code)
--translation_backend hybrid              # Backend: nllb, hunyuan, hybrid

# Verification (MK4)
--enable_verification                     # Enable hallucination detection
--verification_word_overlap_threshold 0.3 # Verification threshold
```

### Environment Variables

```bash
# Set HuggingFace cache directory
export HF_HOME="/path/to/models"
export HUGGINGFACE_HUB_CACHE="/path/to/models"
export TRANSFORMERS_CACHE="/path/to/models"
```

---

## Troubleshooting

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8890

# Kill it
kill -9 <PID>

# Or kill all DMTS processes
pkill -f "dmts_mk4"
```

### CUDA Out of Memory

Try these solutions:
```bash
# Use CPU instead
--device "cpu" --compute_type "float32"

# Use 8-bit quantization (for Hunyuan)
--translation_load_in_8bit

# Use smaller models
--model "faster-whisper-medium"
```

### Model Loading Timeout

First run may take 5-10 minutes to load all models. Check logs for progress.

### Permission Denied on Script

```bash
chmod +x run_server*.sh
```

---

## Directory Structure

```
DMTS/
├── dmts_mk4.py              # Main server
├── language_codes.py        # Language mappings
├── index.html               # Web interface
├── translation/
│   ├── manager_nllb.py      # NLLB backend
│   ├── manager_hybrid.py    # Hybrid backend
│   └── manager_hunyuan.py   # Hunyuan backend
├── run_server.sh            # Default launch script
├── run_server_hybrid.sh     # Hybrid backend
├── run_server_nllb.sh       # NLLB backend
├── run_server_hunyuan.sh    # Hunyuan backend
├── saved_audio/             # Audio recordings (created on run)
└── logs/                    # Transcription logs (created on run)
```

---

## Testing

### Test WebSocket Connection

```bash
# Check if server is running
curl http://localhost:8890/

# View logs
tail -f logs/transcript.log
```

### Test Python Imports

```bash
conda activate dmts
python -c "from RealtimeSTT import AudioToTextRecorder; print('RealtimeSTT OK')"
python -c "from sklearn.cluster import AgglomerativeClustering; print('sklearn OK')"
python -c "from TTS.tts.models import setup_model; print('TTS OK')"
```

---

## Quick Commands Reference

```bash
# Activate environment
conda activate dmts

# Start server
./run_server_hybrid.sh

# Stop server
pkill -f "dmts_mk4"

# Check status
lsof -i :8890
ps aux | grep dmts_mk4

# View logs
tail -f logs/transcript.log
```

---

## Notes

1. **First Run**: Expect 5-10 minutes for model loading
2. **VRAM**: Hybrid backend needs ~16GB VRAM for all models
3. **CPU Fallback**: Server auto-falls back to CPU if CUDA unavailable
4. **Logs**: Transcription logs saved to `logs/transcript.log`
5. **Audio**: Sentence audio saved to `saved_audio/` directory
