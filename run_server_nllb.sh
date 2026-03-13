#!/bin/bash

# ============================================================
# DMTS MK4 - NLLB Only Backend with Hallucination Detection
# ============================================================
# Uses NLLB-600M for real-time and NLLB-3.3B for final translation
# Fast translation, ~200 languages supported

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DMTS MK4 - NLLB Backend${NC}"
echo -e "${BLUE}  (Fast, 200+ Languages)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure cuDNN/CUDA libs are visible to ctranslate2/faster-whisper.
if [ -f "${SCRIPT_DIR}/scripts/configure_cuda_libs.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/scripts/configure_cuda_libs.sh"
    configure_cuda_library_path
fi

# ============================================================
# MODEL PATH CONFIGURATION - EDIT THESE PATHS
# ============================================================
# Set STORAGE_PATH to where you downloaded your models
STORAGE_PATH="${DMTS_MODELS_DIR:-./models}"
WHISPER_MODEL="${DMTS_WHISPER_MODEL:-${STORAGE_PATH}/faster-whisper-large-v3}"
WHISPER_MODEL_REALTIME="${DMTS_WHISPER_MODEL_REALTIME:-${STORAGE_PATH}/faster-whisper-large-v3-turbo-ct2}"
VERIFICATION_MODEL="${DMTS_VERIFICATION_MODEL:-${STORAGE_PATH}/faster-whisper-large-v3-turbo-ct2}"
DIARIZATION_MODEL="${DMTS_DIARIZATION_MODEL_PATH:-${STORAGE_PATH}/XTTS-v2}"
NLLB_600M="${DMTS_NLLB_600M_MODEL:-${STORAGE_PATH}/nllb-200-distilled-600M}"
NLLB_3_3B="${DMTS_NLLB_3_3B_MODEL:-${STORAGE_PATH}/nllb-200-3.3B}"

# ============================================================
# SERVER CONFIGURATION
# ============================================================
PORT=8890
DEVICE="cuda"
COMPUTE_TYPE="float16"
TARGET_LANGUAGE="eng_Latn"
USE_MAIN_MODEL_FOR_REALTIME="true"

# ============================================================
# MK4 VERIFICATION CONFIGURATION
# ============================================================
ENABLE_VERIFICATION="true"
VERIFICATION_COMPUTE_TYPE="float16"
VERIFICATION_WORD_OVERLAP_THRESHOLD=0.05
VERIFICATION_FIRST_N_SENTENCES=2
TRANSLATION_CONSISTENCY_THRESHOLD=0.3

# Create output directories
mkdir -p "${SCRIPT_DIR}/saved_audio"
mkdir -p "${SCRIPT_DIR}/logs"

# Check port
echo -e "${YELLOW}Checking if port ${PORT} is available...${NC}"
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Warning: Port ${PORT} is already in use!${NC}"
    echo "Kill existing process? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        lsof -ti:${PORT} | xargs kill -9
        echo -e "${GREEN}Process killed. Waiting 2 seconds...${NC}"
        sleep 2
    else
        echo "Exiting. Please free port ${PORT} manually."
        exit 1
    fi
fi

# Display configuration
echo -e "${BLUE}Server Configuration:${NC}"
echo -e "  Backend: NLLB (fast, ~200 languages)"
echo -e "  Real-time Model: NLLB-600M"
echo -e "  Final Model: NLLB-3.3B"
echo -e "  Verification: ${ENABLE_VERIFICATION}"
echo -e "  Port: ${PORT}"
echo -e "  Realtime Whisper: ${USE_MAIN_MODEL_FOR_REALTIME} (reuse main model)"
echo ""
echo -e "${YELLOW}Starting server...${NC}"
echo ""

# Build verification args (MK4)
VERIFICATION_ARGS=""
if [ "$ENABLE_VERIFICATION" = "true" ]; then
    VERIFICATION_ARGS="--enable_verification --verification_model_path ${VERIFICATION_MODEL} --verification_compute_type ${VERIFICATION_COMPUTE_TYPE} --verification_word_overlap_threshold ${VERIFICATION_WORD_OVERLAP_THRESHOLD} --verification_first_n_sentences ${VERIFICATION_FIRST_N_SENTENCES} --translation_consistency_threshold ${TRANSLATION_CONSISTENCY_THRESHOLD}"
fi

REALTIME_ARGS=""
if [ "$USE_MAIN_MODEL_FOR_REALTIME" = "true" ]; then
    REALTIME_ARGS="--use_main_model_for_realtime"
fi

# Run the server
cd "${SCRIPT_DIR}"
python dmts_mk4.py \
    --diarization_model_path "${DIARIZATION_MODEL}" \
    --enable_diarization \
    --enable_translation \
    --translation_target_language "${TARGET_LANGUAGE}" \
    --translation_backend nllb \
    --translation_model_realtime "${NLLB_600M}" \
    --translation_model_full "${NLLB_3_3B}" \
    ${VERIFICATION_ARGS} \
    ${REALTIME_ARGS} \
    --model "${WHISPER_MODEL}" \
    --realtime_model_type "${WHISPER_MODEL_REALTIME}" \
    --audio-log-dir "./saved_audio" \
    --transcription-log "./logs/transcript.log" \
    --port ${PORT} \
    --device "${DEVICE}" \
    --compute_type "${COMPUTE_TYPE}" \
    --pre_recording_buffer_duration 0.35
