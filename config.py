import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("DMTS_MODELS_DIR", REPO_ROOT / "models"))

DEFAULT_MODELS = {
    "whisper":          MODELS_DIR / "faster-whisper-large-v3",
    "whisper_realtime": MODELS_DIR / "faster-whisper-large-v3",  # same model as whisper by default
    "nllb_realtime":    MODELS_DIR / "nllb-200-distilled-600M",
    "nllb_full":        MODELS_DIR / "nllb-200-3.3B",
    "xtts":             MODELS_DIR / "XTTS-v2",
    "verification":     MODELS_DIR / "faster-whisper-large-v3-turbo-ct2",
    "hunyuan":          MODELS_DIR / "Hunyuan-MT-7B",
}

# HuggingFace repo IDs for model downloads
HF_REPOS = {
    "whisper":          "Systran/faster-whisper-large-v3",
    "verification":     "deepdml/faster-whisper-large-v3-turbo-ct2",
    "xtts":             "coqui/XTTS-v2",
    "nllb_realtime":    "facebook/nllb-200-distilled-600M",
    "nllb_full":        "facebook/nllb-200-3.3B",
    "hunyuan":          "tencent/Hunyuan-MT-7B",
}

# Which models each backend needs
BACKEND_MODELS = {
    "nllb":    ["whisper", "verification", "xtts", "nllb_realtime", "nllb_full"],
    "hunyuan": ["whisper", "verification", "xtts", "hunyuan"],
    "hybrid":  ["whisper", "verification", "xtts", "nllb_realtime", "nllb_full", "hunyuan"],
}


def get_model_path(name: str) -> str:
    return str(DEFAULT_MODELS[name])
