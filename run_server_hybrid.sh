#!/bin/bash

# ============================================================
# DMTS MK4 - Hybrid Backend with Hallucination Detection
# ============================================================
# Uses NLLB-600M for real-time, Hunyuan for final (38 langs), NLLB-3.3B fallback
# Recommended for best balance of speed and quality.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DMTS MK4 - Hybrid Backend${NC}"
echo -e "${BLUE}  (Hallucination Detection Enabled)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================
# MODEL PATH CONFIGURATION - EDIT THESE PATHS
# ============================================================
WHISPER_MODEL="/path/to/faster-whisper-large-v3"
WHISPER_MODEL_REALTIME="/path/to/faster-whisper-large-v3-turbo-ct2"
VERIFICATION_MODEL="/path/to/faster-whisper-large-v3-turbo-ct2"
DIARIZATION_MODEL="/path/to/XTTS-v2/v2.0.2"
NLLB_600M="/path/to/nllb-200-distilled-600M"
NLLB_3_3B="/path/to/nllb-200-3.3B"
HUNYUAN_MODEL="/path/to/Hunyuan-MT-7B"

# ============================================================
# SERVER CONFIGURATION
# ============================================================
PORT=8890
DEVICE="cuda"
COMPUTE_TYPE="float16"
TARGET_LANGUAGE="eng_Latn"
TRANSLATION_GPU_DEVICE=0
TRANSLATION_LOAD_8BIT="false"

# ============================================================
# MK4 VERIFICATION CONFIGURATION
# ============================================================
ENABLE_VERIFICATION="true"
VERIFICATION_COMPUTE_TYPE="float16"
VERIFICATION_WORD_OVERLAP_THRESHOLD=0.3
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
echo -e "  Backend: Hybrid (NLLB + Hunyuan)"
echo -e "  Real-time: NLLB-600M"
echo -e "  Final (38 langs): Hunyuan-MT-7B"
echo -e "  Final (fallback): NLLB-3.3B"
echo -e "  Verification: ${ENABLE_VERIFICATION}"
echo -e "  Port: ${PORT}"
echo ""
echo -e "${YELLOW}Starting server...${NC}"
echo ""

# Build args
TRANSLATION_ARGS="--translation_backend hybrid --translation_model ${HUNYUAN_MODEL} --translation_model_realtime ${NLLB_600M} --translation_model_full ${NLLB_3_3B} --translation_gpu_device ${TRANSLATION_GPU_DEVICE}"
if [ "$TRANSLATION_LOAD_8BIT" = "true" ]; then
    TRANSLATION_ARGS="${TRANSLATION_ARGS} --translation_load_in_8bit"
fi

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
