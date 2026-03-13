#!/bin/bash

# Ensure CUDA/cuDNN shared libraries are discoverable for ctranslate2/faster-whisper.
configure_cuda_library_path() {
    if [ -z "${CONDA_PREFIX:-}" ]; then
        return 0
    fi

    local py_ver
    py_ver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ -z "${py_ver}" ]; then
        return 0
    fi

    local nvidia_site="${CONDA_PREFIX}/lib/python${py_ver}/site-packages/nvidia"
    local -a extra_paths=("${CONDA_PREFIX}/lib")
    local subdir

    for subdir in \
        "cudnn/lib" \
        "cublas/lib" \
        "cuda_runtime/lib" \
        "cufft/lib" \
        "curand/lib" \
        "cusolver/lib" \
        "cusparse/lib" \
        "nvjitlink/lib"
    do
        if [ -d "${nvidia_site}/${subdir}" ]; then
            extra_paths+=("${nvidia_site}/${subdir}")
        fi
    done

    local path_item
    for path_item in "${extra_paths[@]}"; do
        if [ -d "${path_item}" ] && [[ ":${LD_LIBRARY_PATH:-}:" != *":${path_item}:"* ]]; then
            LD_LIBRARY_PATH="${path_item}:${LD_LIBRARY_PATH:-}"
        fi
    done

    export LD_LIBRARY_PATH
}
