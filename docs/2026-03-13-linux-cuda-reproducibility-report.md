# DMTS Linux CUDA Reproducibility Report (2026-03-13)

## Purpose

Document the full environment-hardening and runtime-stability work done to make DMTS reproducible on Linux with CUDA, including the concrete failure modes we hit and the final stable path.

## Final Recommended Run Path

```bash
conda create -n dmts python=3.10 -y
conda activate dmts
cd DMTS
bash setup_env.sh
python scripts/preflight.py
bash run_server.sh
```

Notes:
- `run_server*.sh` now configures CUDA/cuDNN runtime library paths automatically.
- Realtime Whisper is configured to reuse main Whisper by default (`--use_main_model_for_realtime`) for stability.

## Trial-and-Error Timeline (What We Tried)

1. Started from clean env rebuild and hit `torch` import error:
   `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`.
2. Switched baseline to pip CUDA wheels (`torch/torchaudio/torchvision 2.9.1+cu128`) to align with known-working runtime.
3. Verified server could start and parse args, but first real transcription path crashed with:
   `Unable to load any of {libcudnn_ops...}`.
4. Ran A/B test with a minimal forced transcription script:
   - `dmts` env aborted
   - `realtimestt` env did not abort
5. Checked dynamic library visibility and found:
   - `dmts` had cuDNN libs under `site-packages/nvidia/cudnn/lib`
   - but those dirs were not in `LD_LIBRARY_PATH` during launch.
6. Validated hypothesis by testing `ctypes.CDLL('libcudnn_ops.so.9')`:
   - failed before path injection
   - succeeded after path injection.
7. Implemented launch-time CUDA/cuDNN path injection helper and wired it into all `run_server*.sh`.
8. Added explicit cuDNN runtime/symbol checks into preflight so this fails fast before runtime.
9. Next issue: diarization produced `Speaker -1` with embedding errors from `torchcodec`.
10. Tried adding `torchcodec` as dependency; this failed in practice due FFmpeg/shared-lib and ABI mismatch (`libtorchcodec_core*.so` load errors).
11. Inspected XTTS API and changed diarization embedding path to use in-memory audio tensor directly (no temp file load path).
12. Smoke-tested new embedding path in `dmts`; valid embedding returned, no `torchcodec` requirement.
13. Removed the temporary `torchcodec` requirement and removed no-longer-used RealtimeSTT patch workflow from baseline flow.

## Issues Found and Resolved

### 1) `torch` import failure (`iJIT_NotifyEvent`)

Observed:
- Fresh env with conda PyTorch stack failed at import (`libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`).

Resolution:
- `setup_env.sh` now installs PyTorch CUDA wheels via pip (`torch/torchaudio/torchvision 2.9.1 + cu128`) to match known-working runtime.

### 2) Runtime abort: `Unable to load any of {libcudnn_ops...}`

Observed:
- Server started, then aborted inside transcription path with cuDNN symbol/load errors.
- cuDNN existed in pip-installed NVIDIA package dirs but was not visible in `LD_LIBRARY_PATH`.

CUDA 12.8 vs 12.4 note:
- It is reasonable to suspect a version difference because the older `realtimestt` environment had been working with a CUDA 12.4-based setup earlier in the process.
- However, the strongest direct evidence from this debugging session was not "12.8 is inherently broken." The decisive signal was that `libcudnn_ops.so.9` failed to load before path injection and succeeded immediately after path injection.
- In other words, the observed production failure was primarily a Linux dynamic-library discovery problem in the new environment layout, not just a raw CUDA version mismatch.
- The CUDA version change still matters operationally because moving to pip `cu128` wheels changed where the runtime libraries lived (`site-packages/nvidia/...` instead of a simpler conda-style `$CONDA_PREFIX/lib` layout).

Resolution:
- Added `scripts/configure_cuda_libs.sh`.
- All `run_server*.sh` scripts source this helper before launching `python dmts_mk4.py`.
- `scripts/preflight.py` now checks cuDNN runtime lookup and required symbol availability.

### 3) Diarization returning `Speaker -1`

Observed:
- Logs showed embedding failures (`torchcodec`/FFmpeg runtime mismatch), causing unknown speaker assignment.

Root cause:
- XTTS embedding path depended on loading temp WAV file via stack that routed into fragile `torchcodec` runtime requirements in this env.

Resolution:
- `dmts_mk4.py` diarization embedding path was changed to use in-memory audio tensor directly with XTTS speaker encoder.
- This removes runtime dependency on `torchcodec` for diarization in DMTS.
- `torchcodec` was removed from DMTS requirements.

## Reproducibility Improvements Added

### Shared model path config

- Added `config.py` as single source of truth for model defaults.
- Default model root is repo-local `./models` unless `DMTS_MODELS_DIR` is set.
- CLI defaults in `dmts_mk4.py` are now config-driven (no machine-local fallback paths).

### Scripted verification and model bootstrap

- Added `scripts/preflight.py` to verify:
  - Python/conda baseline
  - PyTorch/CUDA availability
  - ctranslate2/faster-whisper imports
  - cuDNN runtime symbol lookup
  - runtime imports (`pyaudio`, `pandas`, `scikit-learn`)
  - required model directories and writable log/audio dirs
- Added `scripts/download_models.py` with backend-specific model download support.

### Launch script standardization

- `run_server.sh`, `run_server_hybrid.sh`, `run_server_nllb.sh`, `run_server_hunyuan.sh` now:
  - use repo-relative defaults (`./models`) with env overrides
  - auto-configure CUDA/cuDNN runtime paths
  - default to `USE_MAIN_MODEL_FOR_REALTIME=true` for Linux CUDA stability

### Setup simplification

- `setup_env.sh` now focuses on deterministic env setup.
- RealtimeSTT patch workflow was removed as baseline path (no longer required for stable DMTS launch).

## Important Environment Variables

- `DMTS_MODELS_DIR`: override model root directory.
- `DMTS_DIARIZATION_MODEL_PATH`: override XTTS model path only.
- Optional per-model overrides also supported in run scripts:
  - `DMTS_WHISPER_MODEL`
  - `DMTS_WHISPER_MODEL_REALTIME`
  - `DMTS_VERIFICATION_MODEL`
  - `DMTS_NLLB_600M_MODEL`
  - `DMTS_NLLB_3_3B_MODEL`
  - `DMTS_HUNYUAN_MODEL`

## What We Removed as No Longer Needed

- RealtimeSTT startup patch script from the standard flow (`scripts/patch_realtimestt.py`).
- Patch-related setup/preflight requirements.
- `torchcodec` requirement for DMTS diarization path.

## Residual Warnings (Non-Blocking)

Some warnings may still appear and are currently non-blocking for normal server operation:
- transformers model class warning about `GenerationMixin`
- sequential pipeline usage warning from transformers
- occasional `librosa`/`pkg_resources` deprecation warning

## Verification Evidence (Session Summary)

- Shell syntax validation passed for all run scripts and setup script.
- Python compile checks passed for changed Python files.
- cuDNN lookup validation passed when launch helper is applied.
- XTTS speaker embedding smoke test succeeded in `dmts` env with in-memory embedding path (valid embedding tensor returned).
- User runtime confirmed server now runs and diarization works after final fixes.
