# DMTS - Diarization, Multilingual Transcription & Translation Server

A real-time speech-to-text server with **speaker diarization**, **multilingual translation**, and **hallucination detection**. Built on top of [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT).

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
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda (recommended for environment management)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/DMTS.git
cd DMTS

# Create conda environment
conda create -n dmts python=3.10
conda activate dmts

# Install dependencies
pip install -r requirements.txt
```

### Download Models

You'll need to download the following models:

1. **Whisper Models** (for transcription):
   - [faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)
   - [faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) (for real-time/verification)

2. **Diarization Model**:
   - [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (for speaker embeddings)

3. **Translation Models** (based on backend choice):
   - NLLB: [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M), [nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)
   - Hunyuan: [Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B)

## Usage

### Quick Start

1. Edit the run script to set your model paths:
```bash
# Edit run_server_hybrid.sh (or your preferred backend)
nano run_server_hybrid.sh
```

2. Configure the model paths at the top of the script:
```bash
WHISPER_MODEL="/path/to/faster-whisper-large-v3"
WHISPER_MODEL_REALTIME="/path/to/faster-whisper-large-v3-turbo-ct2"
MODEL_PATH="/path/to/XTTS-v2"
# ... etc
```

3. Run the server:
```bash
./run_server_hybrid.sh
```

4. Open the web UI:
```
http://localhost:8890
```

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
├── run_server*.sh           # Launch scripts
├── requirements.txt
├── LICENSE
└── NOTICE
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
