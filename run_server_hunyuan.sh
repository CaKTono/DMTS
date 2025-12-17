#!/bin/bash

# ============================================================
# DMTS MK4 - Hunyuan Only Backend with Hallucination Detection
# ============================================================
# Uses Hunyuan-MT-7B for both real-time and final translation
# High quality but slower, supports 38 languages

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DMTS MK4 - Hunyuan Backend${NC}"
echo -e "${BLUE}  (High Quality LLM, 38 Languages)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================
# MODEL PATH CONFIGURATION - EDIT THESE PATHS
# ============================================================
# Set STORAGE_PATH to where you downloaded your models
STORAGE_PATH="./models"
WHISPER_MODEL="${STORAGE_PATH}/faster-whisper-large-v3"
WHISPER_MODEL_REALTIME="${STORAGE_PATH}/faster-whisper-large-v3-turbo-ct2"
VERIFICATION_MODEL="${STORAGE_PATH}/faster-whisper-large-v3-turbo-ct2"
DIARIZATION_MODEL="${STORAGE_PATH}/XTTS-v2"
HUNYUAN_MODEL="${STORAG E_PATH}/Hunyuan-MT-7B"

# ============================================================
# SERVER CONFIGURATION
# ============================================================
PORT=8890
DEVICE="cuda"
COMPUTE_TYPE="float16"
TARGET_LANGUAGE="eng_Latn"

# Hunyuan configuration
TRANSLATION_GPU_DEVICE=0
TRANSLATION_LOAD_8BIT="false"
# Recommended: Skip real-time translation due to slow LLM inference
SKIP_REALTIME_TRANSLATION="true"

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
echo -e "  Backend: Hunyuan-MT-7B (LLM)"
echo -e "  Supported Languages: 38"
echo -e "  GPU Device: ${TRANSLATION_GPU_DEVICE}"
echo -e "  Load 8-bit: ${TRANSLATION_LOAD_8BIT}"
echo -e "  Skip RT Translation: ${SKIP_REALTIME_TRANSLATION}"
echo -e "  Verification: ${ENABLE_VERIFICATION}"
echo -e "  Port: ${PORT}"
echo ""
echo -e "${YELLOW}Starting server...${NC}"
echo ""

# Build translation args
TRANSLATION_ARGS="--translation_backend hunyuan --translation_model ${HUNYUAN_MODEL} --translation_gpu_device ${TRANSLATION_GPU_DEVICE}"
if [ "$TRANSLATION_LOAD_8BIT" = "true" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --translation_load_in_8bit"
fi
if [ "$SKIP_REALTIME_TRANSLATION" = "true" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --skip_realtime_translation"
fi

# Build verification args (MK4)
VERIFICATION_ARGS=""
if [ "$ENABLE_VERIFICATION" = "true" ]; then
    VERIFICATION_ARGS="--enable_verification --verification_model_path ${VERIFICATION_MODEL} --verification_compute_type ${VERIFICATION_COMPUTE_TYPE} --verification_word_overlap_threshold ${VERIFICATION_WORD_OVERLAP_THRESHOLD} --verification_first_n_sentences ${VERIFICATION_FIRST_N_SENTENCES} --translation_consistency_threshold ${TRANSLATION_CONSISTENCY_THRESHOLD}"
fi

# Run the server
cd "${SCRIPT_DIR}"
python dmts_mk4.py \
    --diarization_model_path "${DIARIZATION_MODEL}" \
    --enable_diarization \
    --enable_translation \
    --translation_target_language "${TARGET_LANGUAGE}" \
    ${TRANSLATION_ARGS} \
    ${VERIFICATION_ARGS} \
    --model "${WHISPER_MODEL}" \
    --realtime_model_type "${WHISPER_MODEL_REALTIME}" \
    --audio-log-dir "./saved_audio" \
    --transcription-log "./logs/transcript.log" \
    --port ${PORT} \
    --device "${DEVICE}" \
    --compute_type "${COMPUTE_TYPE}" \
    --pre_recording_buffer_duration 0.35
