#!/usr/bin/env python3
"""DMTS environment and model preflight checks."""

import os
import site
import sys
from pathlib import Path

# Allow importing config from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

failures = 0

MODEL_ENV_OVERRIDES = {
    "whisper": "DMTS_WHISPER_MODEL",
    "whisper_realtime": "DMTS_WHISPER_MODEL_REALTIME",
    "verification": "DMTS_VERIFICATION_MODEL",
    "xtts": "DMTS_DIARIZATION_MODEL_PATH",
    "nllb_realtime": "DMTS_NLLB_600M_MODEL",
    "nllb_full": "DMTS_NLLB_3_3B_MODEL",
    "hunyuan": "DMTS_HUNYUAN_MODEL",
}


def check(name, passed, msg_pass="", msg_fail=""):
    global failures
    if passed:
        detail = f" — {msg_pass}" if msg_pass else ""
        print(f"  {GREEN}PASS{RESET}  {name}{detail}")
    else:
        detail = f" — {msg_fail}" if msg_fail else ""
        print(f"  {RED}FAIL{RESET}  {name}{detail}")
        failures += 1


def warn(name, msg):
    print(f"  {YELLOW}WARN{RESET}  {name} — {msg}")


def resolve_model_path(model_key: str) -> Path:
    override_env = MODEL_ENV_OVERRIDES.get(model_key)
    if override_env:
        override_value = os.environ.get(override_env)
        if override_value:
            return Path(override_value)
    return config.DEFAULT_MODELS[model_key]


def main():
    global failures

    backend = "hybrid"
    if len(sys.argv) > 1 and sys.argv[1] in ("nllb", "hunyuan", "hybrid"):
        backend = sys.argv[1]

    print(f"\nDMTS Preflight Checks (backend: {backend})\n")

    # 1. Python version
    v = sys.version_info
    check(
        "Python version",
        v.major == 3 and v.minor == 10,
        f"{v.major}.{v.minor}.{v.micro}",
        f"{v.major}.{v.minor}.{v.micro} — Python 3.10 required for TTS compatibility",
    )

    # 2. Conda env active
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    check(
        "Conda environment",
        bool(conda_prefix),
        os.path.basename(conda_prefix),
        "No conda env active. Run: conda activate dmts",
    )

    if site.ENABLE_USER_SITE:
        warn(
            "User site-packages",
            "enabled. For clean-room checks, run with PYTHONNOUSERSITE=1",
        )

    # 3. PyTorch + CUDA
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            check("PyTorch CUDA", True, f"{gpu_name}, CUDA {cuda_ver}")
        else:
            check("PyTorch CUDA", False, msg_fail="torch.cuda.is_available() == False")
    except Exception as e:
        check("PyTorch CUDA", False, msg_fail=str(e))

    # 4. ctranslate2 (fails when cuDNN is missing)
    try:
        import ctranslate2  # noqa: F401

        check("ctranslate2 import", True)
    except Exception as e:
        check(
            "ctranslate2 import",
            False,
            msg_fail=f"{e}\n         Reinstall with setup_env.sh in a clean env",
        )

    # 5. cuDNN runtime lookup (common ctranslate2 runtime failure)
    try:
        import ctypes

        cudnn_names = [
            "libcudnn_ops.so.9.1.0",
            "libcudnn_ops.so.9.1",
            "libcudnn_ops.so.9",
            "libcudnn_ops.so",
        ]
        loaded_name = None
        for lib_name in cudnn_names:
            try:
                ctypes.CDLL(lib_name)
                loaded_name = lib_name
                break
            except OSError:
                continue

        if loaded_name is None:
            check(
                "cuDNN runtime lookup",
                False,
                msg_fail=(
                    "Unable to load libcudnn_ops from LD_LIBRARY_PATH. "
                    "Use run_server*.sh (it configures CUDA lib paths) "
                    "or export your env CUDA library paths manually."
                ),
            )
        else:
            check("cuDNN runtime lookup", True, loaded_name)
            try:
                cudnn = ctypes.CDLL("libcudnn.so.9")
                symbol_ok = hasattr(cudnn, "cudnnCreateTensorDescriptor")
                check(
                    "cuDNN symbol",
                    symbol_ok,
                    "cudnnCreateTensorDescriptor",
                    "libcudnn.so.9 loaded but required symbol is missing",
                )
            except OSError as e:
                check("cuDNN symbol", False, msg_fail=str(e))
    except Exception as e:
        check("cuDNN runtime lookup", False, msg_fail=str(e))

    # 6. faster_whisper
    try:
        from faster_whisper import WhisperModel  # noqa: F401

        check("faster_whisper import", True)
    except Exception as e:
        check("faster_whisper import", False, msg_fail=str(e))

    # 7. DMTS runtime imports
    try:
        import pyaudio  # noqa: F401

        check("pyaudio import", True)
    except Exception as e:
        check("pyaudio import", False, msg_fail=str(e))

    try:
        import pandas  # noqa: F401

        check("pandas import", True)
    except Exception as e:
        check("pandas import", False, msg_fail=str(e))

    try:
        from sklearn.cluster import AgglomerativeClustering  # noqa: F401

        check("scikit-learn import", True)
    except Exception as e:
        check("scikit-learn import", False, msg_fail=str(e))

    # 8. Model directories
    required = config.BACKEND_MODELS.get(backend, config.BACKEND_MODELS["hybrid"])
    print(f"\n  Model directories (MODELS_DIR={config.MODELS_DIR}):\n")
    for key in required:
        path = resolve_model_path(key)
        exists = path.is_dir() and any(path.iterdir())
        check(
            f"  {key}",
            exists,
            str(path.name),
            f"missing: {path}\n           Run: python scripts/download_models.py --backend {backend}",
        )

    # 9. Output directories writable
    print()
    for dirname in ["saved_audio", "logs"]:
        dirpath = config.REPO_ROOT / dirname
        try:
            dirpath.mkdir(exist_ok=True)
            testfile = dirpath / ".preflight_test"
            testfile.write_text("ok")
            testfile.unlink()
            check(f"{dirname}/ writable", True)
        except OSError as e:
            check(f"{dirname}/ writable", False, msg_fail=str(e))

    # Summary
    print()
    if failures == 0:
        print(f"{GREEN}All checks passed.{RESET} Ready to start: bash run_server.sh\n")
    else:
        print(f"{RED}{failures} check(s) failed.{RESET} Fix the issues above before starting the server.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
