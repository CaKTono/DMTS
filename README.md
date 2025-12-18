# DMTS - Diarization, Multilingual Transcription & Translation Server

A real-time speech-to-text server with **speaker diarization**, **multilingual translation**, and **hallucination detection**. Built on top of [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT).

https://github.com/CaKTono/DMTS/raw/main/dmts_demo.mp4

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
- Python 3.10 (required for TTS compatibility)
- NVIDIA GPU with CUDA support
- ~40GB GPU VRAM for Hybrid backend (or ~12GB for NLLB-only)
- Conda (required for cuDNN installation)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/CaKTono/DMTS.git
cd DMTS

# Create conda environment
conda create -n dmts python=3.10 -y
conda activate dmts

# Install cuDNN (required for faster-whisper)
conda install cudnn -y

# Install PyTorch with CUDA support (adjust cu124 for your CUDA version)
# Check your CUDA version with: nvidia-smi
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install PyAudio via conda (requires portaudio system library)
conda install -c conda-forge pyaudio -y

# Install all dependencies (--no-deps bypasses version conflicts)
pip install -r requirements.txt --no-deps
```

### Dependency Conflicts Note

This project has inherent dependency conflicts between package requirements:
- **TTS 0.22.0** requires `numpy==1.22.0`

The `requirements.txt` resolves these by pinning `numpy>=1.23.5`. The `--no-deps` flag bypasses pip's dependency resolver to allow these pinned versions. The application works correctly despite version warnings at startup.

### Download Models

**Default setup**: Models are stored in `DMTS/models` (recommended, pre-configured in all run scripts).

```bash
# Install huggingface CLI if needed
pip install huggingface_hub[cli] hf_transfer

# Create models directory (from DMTS root)
mkdir -p models && cd models

# Whisper Models (required)
hf download Systran/faster-whisper-large-v3 --local-dir faster-whisper-large-v3
hf download deepdml/faster-whisper-large-v3-turbo-ct2 --local-dir faster-whisper-large-v3-turbo-ct2

# Diarization Model (required)
hf download coqui/XTTS-v2 --local-dir XTTS-v2

# Translation Models (choose based on backend)
# For NLLB/Hybrid:
hf download facebook/nllb-200-distilled-600M --local-dir nllb-200-distilled-600M
hf download facebook/nllb-200-3.3B --local-dir nllb-200-3.3B

# For Hunyuan/Hybrid:
hf download tencent/Hunyuan-MT-7B --local-dir Hunyuan-MT-7B

cd ..
```

**Custom path setup**: If you prefer a different location, edit the `STORAGE_PATH` variable in the run scripts:

```bash
# Edit your preferred run script
nano run_server_hybrid.sh

# Change this line (default is /root/DMTS/models):
STORAGE_PATH="/root/DMTS/models"  # Change to your custom path
```

**Model Requirements by Backend:**

| Backend | Required Models | VRAM |
|---------|----------------|------|
| NLLB-only | Whisper, XTTS-v2, NLLB-600M, NLLB-3.3B | ~19GB |
| Hunyuan-only | Whisper, XTTS-v2, Hunyuan-MT-7B | ~22GB |
| Hybrid | All models | ~38GB |

## Usage

### Quick Start

1. **Download models** - Ensure all models are in `DMTS/models/` (see [Download Models](#download-models) section). All run scripts are pre-configured to use this path.

2. **Run the server**:
```bash
chmod +x run_server_hybrid.sh
./run_server_hybrid.sh
```

3. **Open the web UI**: http://localhost:8890

> **Note**: The run scripts are pre-configured with `STORAGE_PATH="/root/DMTS/models"`. If you need a custom path, edit the `STORAGE_PATH` variable in your chosen run script.

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
├── language_codes.py        # ISO/NLLB language mappings
├── index.html               # Web UI
├── translation/
│   ├── manager_nllb.py      # NLLB-only backend
│   ├── manager_hybrid.py    # Hybrid backend
│   └── manager_hunyuan.py   # Hunyuan-only backend
├── run_server*.sh           # Launch scripts (pre-configured for models/ directory)
├── requirements.txt
├── LICENSE
├── NOTICE
└── models/                  # Model storage directory (required)
    ├── faster-whisper-large-v3/
    ├── faster-whisper-large-v3-turbo-ct2/
    ├── XTTS-v2/
    ├── nllb-200-distilled-600M/
    ├── nllb-200-3.3B/
    └── Hunyuan-MT-7B/
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
