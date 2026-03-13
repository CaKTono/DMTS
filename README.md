# DMTS - Diarization, Multilingual Transcription & Translation Server

A real-time speech-to-text server with **speaker diarization**, **multilingual translation**, and **hallucination detection**. Built on top of [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT).

https://github.com/user-attachments/assets/2d5dfd22-d5c2-4c88-a334-324eb5ae4741

## Reproducibility Report

- [Linux CUDA Reproducibility Report (2026-03-13)](docs/2026-03-13-linux-cuda-reproducibility-report.md)

## Features

### Core Capabilities
- **Real-time Transcription**: Live speech-to-text using Whisper models
- **Speaker Diarization**: Identifies who is speaking using Coqui TTS embeddings with agglomerative clustering
- **Multilingual Translation**: Support for 200+ languages via NLLB, with optional high-quality Hunyuan-MT support

### MK4 Enhancements
- **Hallucination Detection**: Contextual consistency verification to detect and recover from Whisper hallucinations
- **MK4 Recovery Protocol**: Extracts valid speech from combined audio when hallucinations are detected
- **Text Correction**: Retroactive text and translation corrections via WebSocket updates
- **Speech Density Filtering**: Flags suspicious transcriptions for verification instead of immediate discard

### Translation Backends

| Backend | Real-time Model | Final Model | Languages | Speed |
|---------|----------------|-------------|-----------|-------|
| **NLLB** | NLLB-600M | NLLB-3.3B | 200+ | Fast |
| **Hunyuan** | Hunyuan-MT-7B | Hunyuan-MT-7B | 38 | High Quality |
| **Hybrid** (Recommended) | NLLB-600M | Hunyuan + NLLB fallback | 200+ | Best of Both |

## Installation

### Prerequisites
- Linux (tested on Ubuntu 22.04+) with an NVIDIA GPU and working driver (`nvidia-smi`)
- [Conda](https://docs.anaconda.com/miniconda/) (Miniconda or Anaconda)
- GPU VRAM: ~19GB (NLLB-only), ~22GB (Hunyuan-only), ~38GB (Hybrid)

### Step 1: Create conda environment

```bash
conda create -n dmts python=3.10 -y
conda activate dmts
```

### Step 2: Install dependencies

```bash
bash setup_env.sh
```

This installs the CUDA 12.8 PyTorch wheels that match the working `realtimestt`
environment, pins `numpy==1.23.5` for the pandas/scikit-learn/TTS stack,
installs `PyAudio`, and installs all remaining requirements.

### Step 3: Download models

```bash
# All models for Hybrid backend (recommended)
python scripts/download_models.py

# Or just the models you need
python scripts/download_models.py --backend nllb      # NLLB-only (~19GB VRAM)
python scripts/download_models.py --backend hunyuan    # Hunyuan-only (~22GB VRAM)
```

Models are downloaded to `DMTS/models/` by default. To use a custom directory
with the helper scripts and direct `python dmts_mk4.py ...` usage:

```bash
python scripts/download_models.py --models-dir /path/to/models

# Then set the env var so config.py, preflight.py, and dmts_mk4.py defaults find them:
export DMTS_MODELS_DIR=/path/to/models
```

The `run_server*.sh` wrappers also honor `DMTS_MODELS_DIR`. If you do not set
it, they default to `./models`.

For Linux CUDA stability, the run scripts currently enable
`--use_main_model_for_realtime` by default. That reuses the main Whisper model
for realtime updates instead of starting a second CUDA faster-whisper model.

If one model lives outside `DMTS_MODELS_DIR`, you can override it with an env
var such as `DMTS_DIARIZATION_MODEL_PATH=/path/to/XTTS-v2`.

### Step 4: Verify environment

```bash
python scripts/preflight.py
```

This checks Python version, CUDA availability, cuDNN/ctranslate2 imports, and that all required model directories exist.
It also checks whether cuDNN runtime libraries are discoverable (a common Linux CUDA runtime issue).

### Step 5: Start the server

```bash
bash run_server.sh           # Hybrid backend (default, recommended)
bash run_server_nllb.sh      # NLLB-only (fast, 200+ languages)
bash run_server_hunyuan.sh   # Hunyuan-only (high quality, 38 languages)
```

Open the web UI at **http://localhost:8890**.

## Usage

### Run Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `run_server.sh` | Hybrid | Default configuration (recommended) |
| `run_server_hybrid.sh` | Hybrid | NLLB real-time + Hunyuan/NLLB final |
| `run_server_nllb.sh` | NLLB | NLLB-only (fastest, 200+ languages) |
| `run_server_hunyuan.sh` | Hunyuan | Hunyuan-only (highest quality, 38 languages) |

### Configuration Options

Key command-line arguments:

```bash
# Transcription
--model PATH              # Whisper model for final transcription
--realtime_model_type PATH  # Whisper model for real-time
--device cuda             # Device (cuda/cpu)
--compute_type float16    # Compute type

# Diarization
--enable_diarization      # Enable speaker identification
--diarization_model_path PATH  # Path to XTTS-v2

# Translation
--enable_translation      # Enable translation
--translation_backend hybrid|nllb|hunyuan
--translation_target_language eng_Latn  # Target language (NLLB code)

# MK4 Verification
--enable_verification     # Enable hallucination detection
--verification_model_path PATH
--verification_word_overlap_threshold 0.3
```

### Custom Model Paths

If your models are not in `./models/`, you have three options:

1. **Environment variable** (used by `config.py`, `preflight.py`, `dmts_mk4.py`, and `run_server*.sh`):
   ```bash
   export DMTS_MODELS_DIR=/storage/ckpt
   ```

2. **Run script variables** (per-model control for `run_server*.sh`):
   Edit `STORAGE_PATH` and individual model variables in your run script.

3. **CLI flags** (per-run override):
   ```bash
   python dmts_mk4.py --model /path/to/whisper --diarization_model_path /path/to/xtts ...
   ```

### Language Codes

The server supports both ISO codes and NLLB codes:
- ISO: `en`, `zh`, `ja`, `ko`, `fr`, `de`, etc.
- NLLB: `eng_Latn`, `zho_Hans`, `jpn_Jpan`, `kor_Hang`, etc.

See `language_codes.py` for the full mapping.

## WebSocket Protocol

### Message Types

**Diarization Update** (server -> client):
```json
{
    "type": "diarization_update",
    "new_sentence": {
        "index": 5,
        "text": "Hello world",
        "speaker_id": 1,
        "translation": {"text": "Hola mundo"}
    },
    "updates": [
        {
            "index": 3,
            "speaker_id": 0,
            "text": "corrected text",
            "translation": {"text": "texto corregido"}
        }
    ]
}
```

**Real-time Transcription** (server -> client):
```json
{
    "type": "realtime",
    "text": "partial transcription...",
    "translation": {"text": "traduccion parcial..."}
}
```

## Project Structure

```
DMTS/
├── dmts_mk4.py              # Main server
├── config.py                # Shared model path definitions
├── language_codes.py        # ISO/NLLB language mappings
├── index.html               # Web UI
├── translation/
│   ├── manager_nllb.py      # NLLB-only backend
│   ├── manager_hybrid.py    # Hybrid backend
│   └── manager_hunyuan.py   # Hunyuan-only backend
├── scripts/
│   ├── configure_cuda_libs.sh # CUDA/cuDNN runtime path setup for launch scripts
│   ├── download_models.py   # Model downloader (HuggingFace)
│   └── preflight.py         # Environment verification
├── run_server*.sh           # Launch scripts
├── setup_env.sh             # Dependency installer (conda)
├── requirements.txt
├── LICENSE
├── NOTICE
└── models/                  # Model storage (created by download_models.py)
    ├── faster-whisper-large-v3/
    ├── faster-whisper-large-v3-turbo-ct2/
    ├── XTTS-v2/
    ├── nllb-200-distilled-600M/
    ├── nllb-200-3.3B/
    └── Hunyuan-MT-7B/
```

## Troubleshooting

### torch import fails with `iJIT_NotifyEvent`
This means the fresh environment pulled in a broken PyTorch + MKL/OpenMP runtime mix.

**Fix**: Reinstall in a fresh conda env using `setup_env.sh`:
```bash
conda create -n dmts python=3.10 -y
conda activate dmts
bash setup_env.sh
```

### `pyaudio` / `pandas` / `scikit-learn` import fails in a fresh env
Run the setup and preflight checks with user site-packages disabled:

```bash
export PYTHONNOUSERSITE=1
python scripts/preflight.py
```

If that fixes the discrepancy, your shell was leaking packages from `~/.local/`.

### ctranslate2 import fails
Reinstall in a fresh env with `setup_env.sh`, then rerun `python scripts/preflight.py`.

### `AudioToTextRecorder` aborts with `Unable to load any of {libcudnn_ops...}`
This usually means cuDNN runtime libraries are installed but not visible to
`ctranslate2` in your current shell/process.

**Fix**:
```bash
# Recommended launch path (auto-configures CUDA/cuDNN library paths):
bash run_server.sh

# Or configure library lookup manually:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
python scripts/preflight.py
```

The provided `run_server*.sh` wrappers also avoid the dual-model warmup path by using
`--use_main_model_for_realtime` by default.

### Model not found errors
Run `python scripts/preflight.py` to see which models are missing, then run `python scripts/download_models.py` to download them.

### Diarization shows `Speaker -1`
`Speaker -1` means the XTTS embedding extraction failed for that sentence.
With current DMTS, diarization uses in-memory audio embeddings and does not require `torchcodec`.

**Fix**:
```bash
# Verify --diarization_model_path points to a valid XTTS-v2 model directory.
# Check server logs for "Error generating speaker embedding" details.
python scripts/preflight.py
```

## Acknowledgments

This project is built upon and extends [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) by Kolja Beigel, which provides the core audio-to-text recording functionality.

Additional components used:
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Speaker embeddings for diarization
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Meta's multilingual translation
- [Hunyuan-MT](https://huggingface.co/tencent/Hunyuan-MT-7B) - Tencent's translation LLM

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This project uses RealtimeSTT which is licensed under the MIT License.
Copyright (c) 2023 Kolja Beigel
