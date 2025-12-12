"""
DMTS MK4 - Diarization + Multilingual Transcription & Translation System (Combined Transcription)

This server provides real-time speech-to-text transcription with speaker diarization,
multi-backend translation, and advanced hallucination prevention with combined transcription
accuracy improvements. MK4 is the recommended version for production use.

### What's New in MK4:
- Combined Transcription Output: Uses combined audio transcription for improved accuracy
- Text Correction Protocol: Retroactively corrects previous sentences when better transcription available
- Translation Re-sync: Automatically re-translates corrected text and sends updates to clients
- Extended Updates Array: Updates can now contain corrected text and translation, not just speaker ID

### Inherited from MK3:
- Contextual Consistency Verification: Detects Whisper hallucinations ("Thanks for watching!")
- Heuristic-Triggered Verification: Only verifies suspicious sentences to reduce latency
- Double-Verification for Segment 0: First sentence verified twice for accuracy
- Translation Consistency Check: Prevents "jumpy" real-time translations
- Robust Concurrency Model: Hybrid threading with ThreadPoolExecutor for blocking tasks
- Non-Blocking Operations: CPU-intensive tasks offloaded to background worker pool

### Features:
- Real-time transcription using Faster-Whisper models
- Speaker diarization using Coqui TTS embeddings (agglomerative clustering)
- Multi-backend translation: NLLB-only, Hunyuan-only, or Hybrid (NLLB + Hunyuan)
- ISO language code support (zh, ja, ko, etc.)
- WebSocket-based communication for control and data handling
- Flexible recording and transcription options with configurable pauses for sentence detection
- Supports Silero and WebRTC VAD for robust voice activity detection
- Multi-layer hallucination prevention (speech density, stuck-loop, contextual verification)

### Starting the Server:
You can start the server using Python or via the provided shell scripts:

```bash
python dmts_mk4.py [OPTIONS]
# Or use provided scripts:
./run_server_hybrid.sh    # Recommended: NLLB realtime + Hunyuan final + verification
./run_server_nllb.sh      # NLLB only (fast, 200+ languages)
./run_server_hunyuan.sh   # Hunyuan only (high quality, 38 languages)
```

### Available Parameters:

#### Core Model Settings:
    - `-m, --model`: Path to Whisper model; default 'faster-whisper-large-v3'.
    - `-r, --realtime_model_type`: Path to real-time model; default same as --model.
    - `-l, --lang, --language`: Language code for transcription (empty = auto-detect).
    - `--compute_type`: Computation type: 'int8', 'float16', 'float32'.
    - `--device`: Device: 'cuda' or 'cpu'.

#### Server Configuration:
    - `-p, --port`: WebSocket server port; default 8890.
    - `--audio-log-dir`: Directory to save audio for each sentence.
    - `--transcription-log`: File path to log all transcription events as JSON.

#### Diarization Settings:
    - `--enable_diarization`: Enable speaker diarization; default True.
    - `--diarization_model_path`: Path to Coqui TTS model for speaker embeddings (REQUIRED).
    - `--diarization_speaker_threshold`: Distance threshold for speaker detection; default 17.0.
    - `--diarization_silhouette_threshold`: Improvement threshold for additional speakers; default 0.0001.

#### Translation Settings (ISO codes supported):
    - `--enable_translation`: Enable translation layer; default True.
    - `--translation_target_language`: Target language - supports ISO ("zh", "ja") or NLLB codes.
    - `--translation_backend`: 'nllb', 'hunyuan', or 'hybrid' (recommended).
    - `--translation_model`: Hunyuan model path (for hunyuan/hybrid backends).
    - `--translation_model_realtime`: NLLB model for real-time (fast, ~600M).
    - `--translation_model_full`: NLLB model for final sentences (~3.3B).
    - `--translation_gpu_device`: GPU device index for translation; default 0.
    - `--translation_load_in_8bit`: Load Hunyuan in 8-bit to save VRAM.
    - `--skip_realtime_translation`: Only translate final sentences (for slow LLMs).

#### MK3 Verification Settings (Hallucination Prevention):
    - `--enable_verification`: Enable contextual consistency verification; default True.
    - `--verification_model_path`: Whisper model for verification; default 'faster-whisper-large-v3-turbo-ct2'.
    - `--verification_compute_type`: Compute type for verification model; default 'float16'.
    - `--verification_word_overlap_threshold`: Word overlap for consistency (0-1); default 0.3.
    - `--verification_first_n_sentences`: Always verify first N sentences; default 2.
    - `--translation_consistency_threshold`: Similarity threshold for real-time translation; default 0.3.

#### VAD & Timing Settings:
    - `--silero_sensitivity`: Silero VAD sensitivity (0-1); default 0.05.
    - `--webrtc_sensitivity`: WebRTC VAD sensitivity (0-3); default 3.
    - `--min_length_of_recording`: Minimum recording duration; default 1.1s.
    - `--end_of_sentence_detection_pause`: Silence for sentence end; default 0.45s.
    - `--pre_recording_buffer_duration`: Audio buffer before speech; default 0.35s.

#### Debug Options:
    - `-D, --debug`: Enable debug logging.
    - `--use_extended_logging`: Enable extensive log messages.
    - `--logchunks`: Log incoming audio chunks.

Run with `--help` for the complete list of all parameters.


### WebSocket Interface:
The server exposes three WebSocket endpoints on a single port (default 8890):
1. **/control**: Send commands (set language, toggle features, etc.) and receive responses.
2. **/data**: Send audio data and receive real-time transcription, diarization, and translation updates.
3. **/stream**: Simplified endpoint for basic streaming (audio in, transcription out).

The server broadcasts real-time transcription updates to all connected clients.
"""

# --- Diarization Imports (Add near other imports) ---
# Ensure TTS library is installed: pip install TTS
# You might need to adjust these imports based on the exact TTS version
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config import load_config
from TTS.tts.configs.xtts_config import XttsConfig # Might be needed for add_safe_globals
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs # Might be needed for add_safe_globals
from TTS.config.shared_configs import BaseDatasetConfig # Might be needed for add_safe_globals
import torch
# import numpy as np # Already imported
import tempfile
import os
# --- End Diarization Imports ---

# Add with other imports at the top of the file
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# new import
from aiohttp import web, WSMsgType
import aiohttp_cors

import sys
import subprocess
import importlib
import logging
from functools import partial

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# check and install package
def check_and_install_packages(packages):
    """
    Checks if specified Python packages are installed, and installs them if not.

    Args:
        packages (list of dict): A list of packages to check. Each dictionary
                                 should contain 'module_name', 'install_name',
                                 and optionally 'attribute'.
    """
    for package in packages:
        module_name = package['module_name'] 
        attribute = package.get('attribute')
        install_name = package['install_name']

        try:
            if attribute:
                module = importlib.import_module(module_name)
                getattr(module, attribute)
            else:
                importlib.import_module(module_name)
        except (ImportError, AttributeError):
            print(f"Module '{module_name}' not found. Installing '{install_name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
                print(f"Package '{install_name}' installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"FATAL: Failed to install '{install_name}'. Please install it manually using 'pip install {install_name}'. Error: {e}", file=sys.stderr)
                sys.exit(1)

from difflib import SequenceMatcher
from collections import deque
from datetime import datetime
import asyncio
import pyaudio
import base64
import sys

# --- On/Off Switches (Boolean Flags) ---

# [CONCEPT] Debug Verbosity -> Controls how much "behind-the-scenes" info is printed to the console.
debug_logging = False         # Enables basic debug messages throughout the app.
extended_logging = False      # Enables even MORE verbose logs, specifically from the audio processing worker.

# [CONCEPT] Data Passthrough -> Controls if extra data is sent back to the client or saved.
send_recorded_chunk = False   # If True, sends raw audio chunks back to the client over the websocket.
log_incoming_chunks = False   # If True, prints a simple "." to the console for every audio chunk received. Useful for seeing if the audio stream is "alive".

# [CONCEPT] Algorithm Toggles -> Enables or disables specific processing features.
silence_timing = False        # If True, enables the dynamic silence detection algorithm in the `text_detected` function.
writechunks = False           # If True, saves the entire stream of incoming raw audio into one single WAV file.
write_sentence_audio = False  # If True, saves the audio of each *completed sentence* as a separate, individual WAV file.

# [CONCEPT] State Holder -> A placeholder variable to hold a live object if a feature is enabled.
wav_file = None               # If `writechunks` is True, this variable will hold the actual file object that Python is writing to.

# --- Fine-Tuning Dials (Heuristic Parameters) ---

# [CONCEPT] Stuck-Loop Detection -> A heuristic to break out of a transcription loop caused by repetitive background noise.
hard_break_even_on_background_noise = 3.0 # The time window in seconds to watch for repetition.
hard_break_even_on_background_noise_min_texts = 3 # How many transcription updates must occur in that window to trigger the check.
hard_break_even_on_background_noise_min_similarity = 0.99 # How similar (99%) the first and last text must be to be considered a "stuck loop".
hard_break_even_on_background_noise_min_chars = 15 # The text must be at least this long to avoid firing on short, common words.

# [CONCEPT] Speech Density Heuristic -> Detects hallucinations by checking if the text is unrealistically long for the audio duration.
# Human speech rarely exceeds ~20 characters/second. Hallucinations often generate long text from short noise/silence.
# If (text_length / audio_duration) > threshold, the sentence is likely a hallucination and will be discarded.
max_speech_density_threshold = 25.0  # Maximum characters per second allowed. Exceeding this triggers hallucination detection.
min_audio_duration_for_density_check = 0.3  # Minimum audio duration in seconds to apply the density check. Very short audio is skipped.

# [CONCEPT] Hallucination Keywords -> Known phrases that Whisper frequently hallucinates from silence/noise.
# Used to trigger targeted verification instead of verifying every sentence.
HALLUCINATION_KEYWORDS = {
    # English
    "thanks for watching", "thank you for watching", "subscribe",
    "like and subscribe", "see you next time", "see you in the next",
    "don't forget to", "mbc", "subtitles by", "captions by",
    # Chinese
    "感谢观看", "订阅", "字幕", "谢谢收看",
    # Japanese
    "ご視聴ありがとう", "チャンネル登録",
    # Korean
    "시청해주셔서 감사합니다", "구독",
}

# [CONCEPT] Translation Consistency -> Track previous translation for real-time consistency checks.
previous_realtime_translation = None
previous_realtime_source = None

text_time_deque = deque()
loglevel = logging.WARNING

FORMAT = pyaudio.paInt16
CHANNELS = 1


if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


check_and_install_packages([
    # {
    #     'module_name': 'RealtimeSTT',                 # Import module
    #     'attribute': 'AudioToTextRecorder',           # Specific class to check
    #     'install_name': 'RealtimeSTT',                # Package name for pip install
    # },
    {
        'module_name': 'websockets',                  # Import module
        'install_name': 'websockets',                 # Package name for pip install
    },
    {
        'module_name': 'numpy',                       # Import module
        'install_name': 'numpy',                      # Package name for pip install
    },
    {
        'module_name': 'scipy.signal',                # Submodule of scipy
        'attribute': 'resample',                      # Specific function to check
        'install_name': 'scipy',                      # Package name for pip install
    },
    {
        'module_name': 'transformers',
        'install_name': 'transformers[sentencepiece]'
    },
    {
        'module_name': 'torch',
        'install_name': 'torch'
    }
])

# Define ANSI color codes for terminal output
class bcolors:
    HEADER = '\033[95m'   # Magenta
    OKBLUE = '\033[94m'   # Blue
    OKCYAN = '\033[96m'   # Cyan
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'     # Red
    ENDC = '\033[0m'      # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f"{bcolors.BOLD}{bcolors.OKCYAN}Starting server, please wait...{bcolors.ENDC}")

# Initialize colorama
from colorama import init, Fore, Style
init()

from RealtimeSTT import AudioToTextRecorder # from __init__.py
from scipy.signal import resample 
import numpy as np
import websockets
import threading
import logging
import wave
import json
import time
import queue  # <-- mk6

# --- Concurrency & System Utilities ---
# [CONCEPT] Parallel Processing & System Integration → Prevents the server from freezing on heavy tasks.
# threading, concurrent.futures.ThreadPoolExecutor: While asyncio is great for waiting on the network, it can't handle CPU-heavy tasks. These libraries are used to move blocking tasks (like saving a file or complex audio resampling) to a separate thread, so the main server loop remains responsive.

# --- ADD THESE ---
from concurrent.futures import ThreadPoolExecutor
# Translation manager will be imported dynamically based on --translation_backend
# --- END ADD ---

# Global variables initialization
global_args = None
recorder = None
recorder_config = {}
recorder_ready = threading.Event()
recorder_thread = None
stop_recorder = False
prev_text = ""
pending_audio_queue = deque() # mk5: Use a queue instead of a single variable
transcription_index = 0
shared_executor = None

# mk6: Queue for passing full sentence data from ASR thread to FullSentenceProcessor thread
# --- ADD THIS LINE ---
full_sentence_queue = queue.Queue() # For passing full sentence data from ASR thread to FullSentenceProcessor thread
# --- END ADD ---

# --- Diarization Global ---
tts_model = None
# --- End Diarization Global ---

# --- MK3: Verification Model Global ---
verification_model = None  # Whisper model for contextual consistency verification
# --- End Verification Model Global ---

# --- ADD THESE ---
translation_manager = None
translation_queue = asyncio.Queue()
target_translation_language = None
# --- END ADD ---

# For control settings 
# Define allowed methods and parameters for security
allowed_methods = [
    'set_microphone',
    'abort',
    'stop',
    'clear_audio_queue',
    'wakeup',
    'shutdown',
    'text',
]
allowed_parameters = [
    'language',
    'silero_sensitivity',
    'wake_word_activation_delay',
    'post_speech_silence_duration',
    'listen_start',
    'recording_stop_time',
    'last_transcription_bytes',
    'last_transcription_bytes_b64',
    'speech_end_silence_start',
    'is_recording',
    'use_wake_words',
]

# [CONCEPT] Connection Pooling -> A dynamic list of all currently active clients.
# A `set` is used because adding/removing clients is very fast, and a client can't be in the list twice.
control_connections = set() # → Will hold all client connections to the "/control" websocket.
data_connections = set()    # → Will hold all client connections to the "/data" websocket. This is the list of clients who will receive transcription broadcasts.
stream_connections = set()  # → MK4: Will hold all client connections to the "/stream" websocket for debug logging.

# [CONCEPT] Asynchronous Message Passing -> A thread-safe "inbox" for passing data between tasks.
# This works like a real-world queue: first-in, first-out (FIFO).
control_queue = asyncio.Queue() # → (Note: This is declared but never used in the code. It's likely leftover from a previous design.)
audio_queue = asyncio.Queue()   # → The central message bus. The recorder callbacks (`text_detected`, etc.) `put` transcription results here. A separate `broadcast_audio_messages` function `gets` results from here to send to clients.

# Preprocessing function
def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    # Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    if text.endswith("...'."):
        text = text[:-1]

    if text.endswith("...'"):
        text = text[:-1]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

# Debugging for timestamp detection
def debug_print(message):
    if debug_logging:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        thread_name = threading.current_thread().name
        print(f"{Fore.CYAN}[DEBUG][{timestamp}][{thread_name}] {message}{Style.RESET_ALL}", file=sys.stderr)

def format_timestamp_ns(timestamp_ns: int) -> str:
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# In stt_server.py (global scope)

def map_lang_to_nllb(lang_code):
    """ Maps Whisper language codes to NLLB-compatible codes. """
    mapping = {
        'en': 'eng_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn',
        'de': 'deu_Latn', 'it': 'ita_Latn', 'pt': 'por_Latn',
        'ru': 'rus_Cyrl', 'ja': 'jpn_Jpan', 'ko': 'kor_Hang',
        'zh': 'zho_Hans', 'ar': 'arb_Arab', 'hi': 'hin_Deva',
    }
    return mapping.get(lang_code, 'eng_Latn') # Default to English if not found

def text_detected(text, loop): # real time text
    global prev_text

    text = preprocess_text(text)

    if text == prev_text:
        return
    
    if silence_timing:
        def ends_with_ellipsis(text: str):
            if text.endswith("..."):
                return True
            if len(text) > 1 and text[:-1].endswith("..."):
                return True
            return False

        def sentence_end(text: str):
            sentence_end_marks = ['.', '!', '?', '。']
            if text and text[-1] in sentence_end_marks:
                return True
            return False

        # [CONCEPT] Contextual Pause Adjustment -> Changes the 'end of speech' timer based on punctuation clues.
        if ends_with_ellipsis(text):
            # Why: The user is probably just thinking.
            # → Do: Give them a lot of time to continue their thought (e.g., 2.0 seconds).
            recorder.post_speech_silence_duration = global_args.mid_sentence_detection_pause
        elif sentence_end(text) and sentence_end(prev_text) and not ends_with_ellipsis(prev_text):
            # Why: The user has finished a sentence and is starting a new one.
            # → Do: Use a very short pause to be responsive (e.g., 0.25 seconds).
            recorder.post_speech_silence_duration = global_args.end_of_sentence_detection_pause
        else:
            # Why: We're not sure if the sentence is over. It's the most common state.
            # → Do: Use a moderate, default pause (e.g., 0.6 seconds).
            recorder.post_speech_silence_duration = global_args.unknown_sentence_detection_pause

        # Append the new text with its timestamp
        current_time = time.time()
        text_time_deque.append((current_time, text))

        # Remove texts older than hard_break_even_on_background_noise seconds
        while text_time_deque and text_time_deque[0][0] < current_time - hard_break_even_on_background_noise:
            text_time_deque.popleft()

        # Check if at least hard_break_even_on_background_noise_min_texts texts have arrived within the last hard_break_even_on_background_noise seconds
        if len(text_time_deque) >= hard_break_even_on_background_noise_min_texts:
            texts = [t[1] for t in text_time_deque]
            first_text = texts[0]
            last_text = texts[-1]

            # Compute the similarity ratio between the first and last texts
            similarity = SequenceMatcher(None, first_text, last_text).ratio()

            if similarity > hard_break_even_on_background_noise_min_similarity and len(first_text) > hard_break_even_on_background_noise_min_chars:
                recorder.stop()
                recorder.clear_audio_queue()
                prev_text = ""

    prev_text = text

    # # Put the message in the audio queue to be sent to clients
    # message = json.dumps({
    #     'type': 'realtime',
    #     'text': text
    # })
    # asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop) # the magic happens


    # --- START MODIFICATION ---
    if global_args.enable_translation:
        # If translation is ON, put a job on the translation queue
        source_lang_code = map_lang_to_nllb(recorder.language)
        job = {
            'type': 'realtime',
            'text': text,
            'source_lang': source_lang_code
        }
        asyncio.run_coroutine_threadsafe(translation_queue.put(job), loop)
    else:
        # If translation is off, behave as before.
        message = json.dumps({
            'type': 'realtime',
            'text': text
        })
        asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)
        
        # Get current timestamp in HH:MM:SS.nnn format
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if extended_logging:
            print(f"  [{timestamp}] Realtime text: {bcolors.OKCYAN}{text}{bcolors.ENDC}\n", flush=True, end="")
        else:
            print(f"\r[{timestamp}] {bcolors.OKCYAN}{text}{bcolors.ENDC}", flush=True, end='')
    # --- END MODIFICATION ---

# --- Recorder Callbacks ---
def on_recording_start(loop):
    message = json.dumps({
        'type': 'recording_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_recording_stop(loop):
    message = json.dumps({
        'type': 'recording_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_start(loop):
    message = json.dumps({
        'type': 'vad_detect_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_vad_detect_stop(loop):
    message = json.dumps({
        'type': 'vad_detect_stop'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detected(loop):
    message = json.dumps({
        'type': 'wakeword_detected'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_start(loop):
    message = json.dumps({
        'type': 'wakeword_detection_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_wakeword_detection_end(loop):
    message = json.dumps({
        'type': 'wakeword_detection_end'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_transcription_start(_audio_bytes, loop):
    # mk2
    # remove the audio in the message, send a simple notification
    global pending_audio_queue # mk5: Use the queue

    # Store the raw audio bytes for later
    # mk5: Append to queue instead of overwriting
    pending_audio_queue.append(_audio_bytes)
    debug_print(f"Appended audio to queue (size: {len(pending_audio_queue)})")

    message = json.dumps({
        'type': 'transcription_start'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_start(loop):
    print("&&& stt_server on_turn_detection_start")
    message = json.dumps({
        'type': 'start_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

def on_turn_detection_stop(loop):
    print("&&& stt_server on_turn_detection_stop")
    message = json.dumps({
        'type': 'stop_turn_detection'
    })
    asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)


# def on_realtime_transcription_update(text, loop):
#     # Send real-time transcription updates to the client
#     text = preprocess_text(text)
#     message = json.dumps({
#         'type': 'realtime_update',
#         'text': text
#     })
#     asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

# def on_recorded_chunk(chunk, loop):
#     if send_recorded_chunk:
#         bytes_b64 = base64.b64encode(chunk.tobytes()).decode('utf-8')
#         message = json.dumps({
#             'type': 'recorded_chunk',
#             'bytes': bytes_b64
#         })
#         asyncio.run_coroutine_threadsafe(audio_queue.put(message), loop)

# Define the server's arguments
def parse_arguments():
    global debug_logging, extended_logging, loglevel, writechunks, log_incoming_chunks, dynamic_silence_timing

    import argparse
    parser = argparse.ArgumentParser(description='Start the Speech-to-Text (STT) server with various configuration options.')

    parser.add_argument('-m', '--model', type=str, default='/home/zhouzhencheng/storage/ckpt/faster-whisper-large-v3',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='/home/zhouzhencheng/storage/ckpt/faster-whisper-large-v3',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')
    
    parser.add_argument('-l', '--lang', '--language', type=str, default='',
                help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')

    parser.add_argument('-i', '--input-device', '--input-device-index', type=int, default=1,
                    help='Index of the audio input device to use. Use this option to specify a particular microphone or audio input device based on your system. Default is 1.')

    parser.add_argument('-c', '--control', '--control_port', type=int, default=8011,
                        help='The port number used for the control WebSocket connection. Control connections are used to send and receive commands to the server. Default is port 8011.')

    parser.add_argument('-d', '--data', '--data_port', type=int, default=8012,
                        help='The port number used for the data WebSocket connection. Data connections are used to send audio data and receive transcription updates in real time. Default is port 8012.')

    parser.add_argument('-p', '--port', type=int, default=8888,
                        help='The port number for the main web server (serving index.html). Default is 8888.')

    parser.add_argument('-w', '--wake_words', type=str, default="",
                        help='Specify the wake word(s) that will trigger the server to start listening. For example, setting this to "Jarvis" will make the system start transcribing when it detects the wake word "Jarvis". Default is "Jarvis".')

    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug logging for detailed server operations')

    parser.add_argument('--debug_websockets', action='store_true', help='Enable debug logging for detailed server websocket operations')

    parser.add_argument('-W', '--write', metavar='FILE', help='Save received audio to a WAV file')

    parser.add_argument('--audio-log-dir', type=str, default=None, help='Path to a directory to save the audio for each full sentence transcription.')
    
    parser.add_argument('--transcription-log', type=str, default=None, help='Path to a file to save all transcription and event JSON messages.')

    parser.add_argument('-b', '--batch', '--batch_size', type=int, default=16, help='Batch size for inference. This parameter controls the number of audio chunks processed in parallel during transcription. Default is 16.')

    parser.add_argument('--root', '--download_root', type=str,default=None, help='Specifies the root path where the Whisper models are downloaded to. Default is None.')

    parser.add_argument('-s', '--silence_timing', action='store_true', default=True,
                    help='Enable dynamic adjustment of silence duration for sentence detection. Adjusts post-speech silence duration based on detected sentence structure and punctuation. Default is False.')

    parser.add_argument('--init_realtime_after_seconds', type=float, default=0.2,
                        help='The initial waiting time in seconds before real-time transcription starts. This delay helps prevent false positives at the beginning of a session. Default is 0.2 seconds.')  
    
    parser.add_argument('--realtime_batch_size', type=int, default=16,
                        help='Batch size for the real-time transcription model. This parameter controls the number of audio chunks processed in parallel during real-time transcription. Default is 16.')
    
    parser.add_argument('--initial_prompt_realtime', type=str, default="", help='Initial prompt that guides the real-time transcription model to produce transcriptions in a particular style or format.')

    parser.add_argument('--silero_sensitivity', type=float, default=0.05,
                        help='Sensitivity level for Silero Voice Activity Detection (VAD), with a range from 0 to 1. Lower values make the model less sensitive, useful for noisy environments. Default is 0.05.')

    parser.add_argument('--silero_use_onnx', action='store_true', default=False,
                        help='Enable ONNX version of Silero model for faster performance with lower resource usage. Default is False.')

    parser.add_argument('--webrtc_sensitivity', type=int, default=3,
                        help='Sensitivity level for WebRTC Voice Activity Detection (VAD), with a range from 0 to 3. Higher values make the model less sensitive, useful for cleaner environments. Default is 3.')

    parser.add_argument('--min_length_of_recording', type=float, default=1.1,
                        help='Minimum duration of valid recordings in seconds. This prevents very short recordings from being processed, which could be caused by noise or accidental sounds. Default is 1.1 seconds.')

    parser.add_argument('--min_gap_between_recordings', type=float, default=0,
                        help='Minimum time (in seconds) between consecutive recordings. Setting this helps avoid overlapping recordings when there’s a brief silence between them. Default is 0 seconds.')

    parser.add_argument('--pre_recording_buffer_duration', type=float, default=0.2,
                        help='Duration of the pre-recording buffer in seconds. Default is 0.2 seconds.')

    parser.add_argument('--enable_realtime_transcription', action='store_true', default=True,
                        help='Enable continuous real-time transcription of audio as it is received. When enabled, transcriptions are sent in near real-time. Default is True.')

    parser.add_argument('--realtime_processing_pause', type=float, default=0.01,
                        help='Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness but may put more load on the CPU. Default is 0.02 seconds.')

    parser.add_argument('--silero_deactivity_detection', action='store_true', default=True,
                        help='Use the Silero model for end-of-speech detection. This option can provide more robust silence detection in noisy environments, though it consumes more GPU resources. Default is True.')

    parser.add_argument('--early_transcription_on_silence', type=float, default=0.2,
                        help='Start transcription after the specified seconds of silence. This is useful when you want to trigger transcription mid-speech when there is a brief pause. Should be lower than post_speech_silence_duration. Set to 0 to disable. Default is 0.2 seconds.')

    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for the main transcription model. Larger values may improve transcription accuracy but increase the processing time. Default is 5.')

    parser.add_argument('--beam_size_realtime', type=int, default=3,
                        help='Beam size for the real-time transcription model. A smaller beam size allows for faster real-time processing but may reduce accuracy. Default is 3.')

    parser.add_argument('--initial_prompt', type=str,
                        default="Incomplete thoughts should end with '...'. Examples of complete thoughts: 'The sky is blue.' 'She walked home.' Examples of incomplete thoughts: 'When the sky...' 'Because he...'",
                        help='Initial prompt that guides the transcription model to produce transcriptions in a particular style or format. The default provides instructions for handling sentence completions and ellipsis usage.')

    parser.add_argument('--end_of_sentence_detection_pause', type=float, default=0.25,
                        help='The duration of silence (in seconds) that the model should interpret as the end of a sentence. This helps the system detect when to finalize the transcription of a sentence. Default is 0.45 seconds.')

    parser.add_argument('--unknown_sentence_detection_pause', type=float, default=0.5,
                        help='The duration of pause (in seconds) that the model should interpret as an incomplete or unknown sentence. This is useful for identifying when a sentence is trailing off or unfinished. Default is 0.7 seconds.')

    parser.add_argument('--mid_sentence_detection_pause', type=float, default=2.0,
                        help='The duration of pause (in seconds) that the model should interpret as a mid-sentence break. Longer pauses can indicate a pause in speech but not necessarily the end of a sentence. Default is 2.0 seconds.')

    parser.add_argument('--wake_words_sensitivity', type=float, default=0.5,
                        help='Sensitivity level for wake word detection, with a range from 0 (most sensitive) to 1 (least sensitive). Adjust this value based on your environment to ensure reliable wake word detection. Default is 0.5.')

    parser.add_argument('--wake_word_timeout', type=float, default=5.0,
                        help='Maximum time in seconds that the system will wait for a wake word before timing out. After this timeout, the system stops listening for wake words until reactivated. Default is 5.0 seconds.')

    parser.add_argument('--wake_word_activation_delay', type=float, default=0,
                        help='The delay in seconds before the wake word detection is activated after the system starts listening. This prevents false positives during the start of a session. Default is 0 seconds.')

    parser.add_argument('--wakeword_backend', type=str, default='none',
                        help='The backend used for wake word detection. You can specify different backends such as "default" or any custom implementations depending on your setup. Default is "pvporcupine".')

    parser.add_argument('--openwakeword_model_paths', type=str, nargs='*',
                        help='A list of file paths to OpenWakeWord models. This is useful if you are using OpenWakeWord for wake word detection and need to specify custom models.')

    parser.add_argument('--openwakeword_inference_framework', type=str, default='tensorflow',
                        help='The inference framework to use for OpenWakeWord models. Supported frameworks could include "tensorflow", "pytorch", etc. Default is "tensorflow".')

    parser.add_argument('--wake_word_buffer_duration', type=float, default=1.0,
                        help='Duration of the buffer in seconds for wake word detection. This sets how long the system will store the audio before and after detecting the wake word. Default is 1.0 seconds.')

    parser.add_argument('--use_main_model_for_realtime', action='store_true',
                        help='Enable this option if you want to use the main model for real-time transcription, instead of the smaller, faster real-time model. Using the main model may provide better accuracy but at the cost of higher processing time.')

    parser.add_argument('--use_extended_logging', action='store_true',
                        help='Writes extensive log messages for the recording worker, that processes the audio chunks.')

    parser.add_argument('--compute_type', type=str, default='int8',
                        help='Type of computation to use. See https://opennmt.net/CTranslate2/quantization.html')

    parser.add_argument('--gpu_device_index', type=int, default=0,
                        help='Index of the GPU device to use. Default is None.')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for model to use. Can either be "cuda" or "cpu". Default is cuda.')
    
    parser.add_argument('--handle_buffer_overflow', action='store_true',
                        help='Handle buffer overflow during transcription. Default is False.')

    parser.add_argument('--suppress_tokens', type=int, default=[-1], nargs='*', help='Suppress tokens during transcription. Default is [-1].')

    parser.add_argument('--allowed_latency_limit', type=int, default=100,
                        help='Maximal amount of chunks that can be unprocessed in queue before discarding chunks.. Default is 100.')

    parser.add_argument('--faster_whisper_vad_filter', default=True, action='store_true',
                        help='Enable VAD filter for Faster Whisper. Default is False.')

    parser.add_argument('--logchunks', action='store_true', help='Enable logging of incoming audio chunks (periods)')

    # --- ADD TRANSLATION ARGUMENTS ---
    parser.add_argument('--enable_translation', default=True, action='store_true',
                        help='Enable the translation layer.')
    
    parser.add_argument('--translation_target_language', type=str, default='eng_Latn',
                        help='Target language for translation (e.g., fra_Latn for French, spa_Latn for Spanish). Uses NLLB language codes.')
    
    parser.add_argument('--translation_backend', type=str, default='nllb',
                        choices=['nllb', 'hunyuan', 'hybrid'],
                        help='Translation backend: "nllb" (fast), "hunyuan" (LLM), or "hybrid" (NLLB realtime + Hunyuan final).')

    parser.add_argument('--translation_model', type=str, default='tencent/Hunyuan-MT-7B',
                        help='Path to Hunyuan-MT-7B model (used when --translation_backend=hunyuan).')

    parser.add_argument('--translation_model_realtime', type=str, default='/home/zhouzhencheng/storage/ckpt/nllb-200-distilled-600M',
                        help='Hugging Face model for real-time translation (NLLB backend only).')

    parser.add_argument('--translation_model_full', type=str, default='/home/zhouzhencheng/storage/ckpt/nllb-200-3.3B',
                        help='Hugging Face model for full-sentence translation (NLLB backend only).')

    parser.add_argument('--translation_load_in_8bit', action='store_true', default=False,
                        help='Load Hunyuan model in 8-bit quantization to save VRAM.')

    parser.add_argument('--translation_gpu_device', type=int, default=0,
                        help='GPU device index for Hunyuan translation model (default: 0).')

    parser.add_argument('--skip_realtime_translation', action='store_true', default=False,
                        help='Skip real-time translation (only translate final sentences). Recommended for slow LLM backends.')
    # --- END ADD ---

    # Mk6
    # --- ADD DIARIZATION ARGUMENTS (Inside parse_arguments function) ---
    parser.add_argument('--enable_diarization', action='store_true', default=True,
                        help='Enable speaker diarization using TTS model embeddings.')
    parser.add_argument('--diarization_model_path', type=str, required=True, default='/home/zhouzhencheng/realtime_mt/Real-time_STT/WhoSpeaks/XTTS-v2/v2.0.2',
                        help='Path to the Coqui TTS model directory used for generating speaker embeddings.')
    # ...
    # local_models_path = os.environ.get("COQUI_MODEL_PATH")
    # checkpoint = os.path.join(local_models_path, "v2.0.2")
    # config = load_config((os.path.join(checkpoint, "config.json")))
    # self.tts = setup_tts_model(config)
    
    # Inside the parse_arguments function, near the other diarization arguments

    parser.add_argument('--diarization_speaker_threshold', type=float, default=17.0,
                        help='Embedding distance threshold to determine if there is more than one speaker.')
    parser.add_argument('--diarization_silhouette_threshold', type=float, default=0.0001,
                        help='Similarity improvement required to identify an additional speaker when more than two are present.')
    # --- END ADD DIARIZATION ARGUMENTS ---

    # --- MK3: CONTEXTUAL CONSISTENCY VERIFICATION ARGUMENTS ---
    parser.add_argument('--enable_verification', action='store_true', default=True,
                        help='Enable contextual consistency verification to detect hallucinations.')
    parser.add_argument('--verification_model_path', type=str, 
                        default='/home/zhouzhencheng/storage/ckpt/faster-whisper-large-v3-turbo-ct2',
                        help='Path to the Whisper model for verification (e.g., large-v3-turbo).')
    parser.add_argument('--verification_compute_type', type=str, default='float16',
                        help='Compute type for verification model (float16, int8, etc.).')
    parser.add_argument('--verification_word_overlap_threshold', type=float, default=0.3,
                        help='Minimum word overlap ratio to consider transcription consistent (0.0-1.0).')
    parser.add_argument('--verification_first_n_sentences', type=int, default=2,
                        help='Always verify first N sentences (hallucinations common at start).')
    parser.add_argument('--translation_consistency_threshold', type=float, default=0.3,
                        help='Similarity threshold for real-time translation consistency (0.0-1.0).')
    # --- END MK3 VERIFICATION ARGUMENTS ---

    # Parse arguments
    args = parser.parse_args()
    debug_logging = args.debug
    extended_logging = args.use_extended_logging
    writechunks = args.write
    log_incoming_chunks = args.logchunks
    dynamic_silence_timing = args.silence_timing


    ws_logger = logging.getLogger('websockets') # Create a logger for websockets
    '''
    When app debug is ON (args.debug_websockets is True):
    - The ws_logger outputs all logs, including debug messages, to the console or wherever your logging is configured.
    - ws_logger.propagate = False means these logs are handled only by the ws_logger and not passed up to the root logger, preventing duplicate log entries.
    
    When app debug is OFF:
    - The ws_logger only outputs warnings and errors (not debug/info).
    - ws_logger.propagate = True means these warning/error logs are also sent to the root logger, so they appear in your main application logs.
    '''
    if args.debug_websockets:
        # If app debug is on, let websockets be verbose too
        '''
        Verbosity refers to the amount of detail included in output, especially in logs or messages.
            - High verbosity: Shows lots of detailed information (e.g., debug logs, step-by-step actions).
            - Low verbosity: Shows only essential information (e.g., warnings, errors).
        '''
        ws_logger.setLevel(logging.DEBUG)
        # Ensure it uses the handler configured by basicConfig
        ws_logger.propagate = False # Prevent duplicate messages if it also propagates to root (we already have ws_logger set up, so we don't need to propagate to the root logger)
    else:
        # If app debug is off, silence websockets below WARNING
        ws_logger.setLevel(logging.WARNING)
        ws_logger.propagate = True # Allow WARNING/ERROR messages to reach root logger's handler
    '''
    Why specifically "\\n"?
        When passing arguments via the command line, typing \n is interpreted by the shell as a literal backslash and n, not as a newline character.

        If you type --initial_prompt "Line1\nLine2", most shells will pass the string as Line1nLine2 (the \ is ignored).
        To actually include a newline, you’d have to type a real line break, which is not practical in a command.
        So, users typically write \\n to mean a literal backslash and n in the string, which your code then replaces with an actual newline.
    '''
    # Replace escaped newlines with actual newlines in initial_prompt
    if args.initial_prompt:
        args.initial_prompt = args.initial_prompt.replace("\\n", "\n")

    if args.initial_prompt_realtime:
        args.initial_prompt_realtime = args.initial_prompt_realtime.replace("\\n", "\n")

    return args
# start of the modification
def _save_audio_file(filename, audio_bytes, channels, sample_width, framerate):
    """
    Saves audio bytes to a WAV file. This is a blocking I/O operation.
    Intended to be run in a separate thread to avoid blocking the main threads.
    """
    try:
        # [CONCEPT] Context Manager for Files -> This is the safest way to work with files.
        # The `with` statement guarantees the file is properly closed automatically, even if errors occur.
        # 'wb' means "write binary" mode, which is necessary for non-text files like audio.
        with wave.open(filename, 'wb') as wf:
            
            # These next lines write the WAV file's "header" metadata.
            # This is like writing the "To:" and "From:" on an envelope before putting the letter inside.
            wf.setnchannels(channels)       # Set to 1 for mono, 2 for stereo
            wf.setsampwidth(sample_width)   # Set the bit depth (e.g., 2 bytes for 16-bit audio)
            wf.setframerate(framerate)      # Set the sample rate (e.g., 16000 Hz)

            # This is the main action: writing the actual audio data into the file.
            wf.writeframes(audio_bytes)

        # If verbose logging is on, print a success message.
        if extended_logging:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"  [{timestamp}] {bcolors.OKGREEN}Saved audio to: {filename}{bcolors.ENDC}\n", flush=True, end="")

    # [CONCEPT] Error Handling -> If anything goes wrong with the file operation 
    # (e.g., disk is full, invalid filename, no permissions), this block catches the error.
    except Exception as e:
        # It prints a helpful error message instead of crashing the worker thread.
        print(f"{bcolors.FAIL}Error saving audio file in background thread: {e}{bcolors.ENDC}")
# end of the modification

# new handler to serve index.html
async def handle_index(request):
    """Handler to serve the index.html file."""
    # Assumes index.html is in the same directory as the script
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'index.html')
    
    try:
        with open(file_path, 'r') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except FileNotFoundError:
        return web.Response(status=404, text="Error: index.html not found in the same directory as the script.")

def _recorder_thread(loop):
    # Mk2 -> add pending audio buffer
    # Mk3 -> add shared_executor pool (team of workers)
    # modify the global variable we already define -> commmunicating the status of the recorder to the main thread
    global recorder, stop_recorder, pending_audio_queue, transcription_index, shared_executor # Add pending_audio_queue, transcription_index here
    print(f"{bcolors.OKGREEN}Initializing RealtimeSTT server with parameters:{bcolors.ENDC}")
    for key, value in recorder_config.items():
        print(f"    {bcolors.OKBLUE}{key}{bcolors.ENDC}: {value}")

    # --- (print translation args) ---
    if global_args and global_args.enable_translation:
        # print(f"{bcolors.OKGREEN}Initializing Translation layer with parameters:{bcolors.ENDC}")
        print(f"    {bcolors.OKBLUE}translation_backend{bcolors.ENDC}: {global_args.translation_backend}")
        print(f"    {bcolors.OKBLUE}translation_target_language{bcolors.ENDC}: {global_args.translation_target_language}")
        if global_args.translation_backend == 'hunyuan':
            print(f"    {bcolors.OKBLUE}translation_model{bcolors.ENDC}: {global_args.translation_model}")
            print(f"    {bcolors.OKBLUE}translation_load_in_8bit{bcolors.ENDC}: {global_args.translation_load_in_8bit}")
        else:
            print(f"    {bcolors.OKBLUE}translation_model_realtime{bcolors.ENDC}: {global_args.translation_model_realtime}")
            print(f"    {bcolors.OKBLUE}translation_model_full{bcolors.ENDC}: {global_args.translation_model_full}")
    # --- end ---

    recorder = AudioToTextRecorder(**recorder_config) # this is recorder, that's why we define it globally as None at first
    print(f"{bcolors.OKGREEN}{bcolors.BOLD}RealtimeSTT initialized{bcolors.ENDC}")

    # [CONCEPT] Synchronization Signal -> This is the "I'm ready!" signal.
    # It flips the `recorder_ready` event, which un-pauses the main thread, letting it
    # know that the recorder object now exists and is ready to receive audio.
    recorder_ready.set()
    
    #  This function will be our callback for when a *full sentence* is transcribed.
    def process_text(full_sentence):
        global prev_text, pending_audio_queue, transcription_index
        prev_text = ""
        # It first does some cleanup on the text.
        full_sentence = preprocess_text(full_sentence)
        # mk6
        # --- STEP 2: Refactor - Prepare data packet for FullSentenceProcessorThread ---
        # 1. Create a unique index for this transcription
        current_index = transcription_index
        transcription_index += 1 # Increment for the next sentence

        # 2. Prepare the data packet to send to the FullSentenceProcessorThread
        #    Include the sentence text, the audio buffer, and the index.
        #    The audio buffer might be None if on_transcription_start wasn't called
        #    or if there was an issue capturing it.
        data_packet = {
            'index': current_index,
            'text': full_sentence,
            'audio_buffer': None # Initialized to None, will be updated below
        }

        # --- STEP 2: Get the audio data for this sentence ---
        # mk5: Pop from the queue to ensure we get the *correct* audio for *this* text
        # even if multiple sentences came in quickly.
        current_audio_buffer = None
        try:
            if pending_audio_queue:
                current_audio_buffer = pending_audio_queue.popleft()
                debug_print(f"Retrieved audio from queue (queue size: {len(pending_audio_queue)})")
        except IndexError:
            pass

        # Fallback: If queue was empty, try to get it from the recorder
        if current_audio_buffer is None:
             if hasattr(recorder, 'last_transcription_bytes'):
                 current_audio_buffer = recorder.last_transcription_bytes
                 debug_print(f"Recovered audio buffer from recorder.last_transcription_bytes")

        # --- SPEECH DENSITY HEURISTIC (Hallucination Detection) ---
        # Calculate if the transcription is suspiciously long for the audio duration.
        # MK4: Instead of discarding, flag for mandatory verification to allow recovery
        density_suspicious = False
        if current_audio_buffer is not None and len(full_sentence) > 0:
            # Audio buffer is float32 at 16kHz sample rate
            audio_duration_sec = len(current_audio_buffer) / 16000.0
            
            # Only apply check if audio is long enough to be meaningful
            if audio_duration_sec >= min_audio_duration_for_density_check:
                text_length = len(full_sentence)
                speech_density = text_length / audio_duration_sec
                
                if speech_density > max_speech_density_threshold:
                    # MK4: Don't discard - flag for mandatory verification instead
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    print(f"\n[{timestamp}] {bcolors.WARNING}[DENSITY SUSPICIOUS] \"{full_sentence}\" "
                          f"(density: {speech_density:.1f} chars/sec, audio: {audio_duration_sec:.2f}s) - flagged for MK4 verification{bcolors.ENDC}")
                    density_suspicious = True
                else:
                    debug_print(f"Density check PASSED: {speech_density:.1f} chars/sec (audio: {audio_duration_sec:.2f}s)")
        # --- END SPEECH DENSITY HEURISTIC ---
        
        # MK4: Add density_suspicious flag to data packet
        data_packet['density_suspicious'] = density_suspicious

        # Update data packet with the correct buffer
        data_packet['audio_buffer'] = current_audio_buffer

        full_sentence_queue.put(data_packet)
        debug_print(f"Put sentence (index {current_index}) onto full_sentence_queue")

        # (No need to clear pending_audio_buffer anymore, popping does that)
        # --- END STEP 2 ---

    try:
        # [CONCEPT] Graceful Shutdown Loop -> This loop will run forever as long as the global
        # `stop_recorder` flag is False. The main thread can set this to True to stop the server.
        while not stop_recorder:
            
            # [CONCEPT] Blocking Method with Callback -> This is the core action.
            # The `recorder.text()` method starts listening. The program's execution will PAUSE on this line
            # until a full sentence has been detected.
            # Once a sentence is ready, it will automatically call the `process_text` function
            # that we passed to it as an argument.
            recorder.text(process_text)
    # This allows the thread to be shut down cleanly with Ctrl+C.
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Exiting application due to keyboard interrupt{bcolors.ENDC}")

# --- STEP 3: Create FullSentenceProcessorThread ---
import threading
import queue
import os
import numpy as np
import pyaudio
import wave
import json
import asyncio

# Import the _save_audio_file function if it's defined elsewhere, or define it here.
# Assuming it's already defined in the file.
# from your_module import _save_audio_file # Uncomment if needed and adjust

class FullSentenceProcessorThread(threading.Thread):
    """
    A dedicated thread to process full sentences received from the ASR.
    This includes tasks like saving audio files and preparing data for diarization.
    It consumes from full_sentence_queue and produces to audio_queue or translation_queue.
    MK3: Now includes Contextual Consistency Verification to detect hallucinations.
    """

    def __init__(self, full_sentence_queue, audio_queue, translation_queue, shared_executor, global_args, recorder_ref, loop, tts_model, verification_model=None):
        """
        Initializes the FullSentenceProcessorThread.

        Args:
            full_sentence_queue (queue.Queue): Queue to consume sentence data packets from (_recorder_thread).
            audio_queue (asyncio.Queue): Queue to put non-translated final messages onto (for broadcasting).
            translation_queue (asyncio.Queue): Queue to put translation jobs onto.
            shared_executor (concurrent.futures.ThreadPoolExecutor): Executor for offloading blocking tasks.
            global_args (argparse.Namespace): Parsed command-line arguments.
            recorder_ref (AudioToTextRecorder): A reference to the recorder instance (for language access).
            loop (asyncio.AbstractEventLoop): The main asyncio event loop to schedule async tasks.
            verification_model (WhisperModel): MK3 - Whisper model for contextual consistency verification.
        """
        super().__init__(name="FullSentenceProcessorThread") # Give the thread a descriptive name
        self.full_sentence_queue = full_sentence_queue
        self.audio_queue = audio_queue
        self.translation_queue = translation_queue
        self.shared_executor = shared_executor
        self.global_args = global_args
        self.recorder = recorder_ref # Keep a reference to the recorder
        self.tts_model = tts_model  # <-- Store the TTS model
        self.verification_model = verification_model  # MK3: Whisper model for verification
        self.loop = loop
        self.stop_event = threading.Event() # Use an event to signal the thread to stop

        # --- ADD THIS BLOCK for Diarization State ---
        # This list will store a history of (text, speaker_embedding) tuples.
        self.full_sentences = []
        # This list will store the assigned speaker ID (e.g., 0, 1, 2) for each sentence.
        self.sentence_speakers = []
        # --- END Diarization State ---

        # --- MK3: Contextual Consistency Verification State ---
        # Store the previous audio buffer for combining with current audio
        self.previous_audio_buffer = None
        # Segment 0 double-verification: buffer segment 0 to re-verify when segment 1 arrives
        self.segment_0_buffer = None  # Will hold {'text': str, 'audio': ndarray, 'was_broadcast': bool}
        # --- END MK3 State ---

        # --- MK4: Combined Transcription State ---
        # Store the previous sentence text for text extraction from combined transcription
        self.previous_text = None
        # Store history of (index, text) for potential corrections
        self.sentence_history = []  # List of {'index': int, 'text': str, 'audio': ndarray}
        # --- END MK4 State ---
    
    def _get_speaker_embedding(self, audio_buffer):
        """
        Generates a speaker embedding from a given audio buffer.

        Args:
            audio_buffer (np.ndarray): The audio data as a NumPy array.

        Returns:
            np.ndarray or None: The calculated speaker embedding, or None if an error occurs.
        """
        # Return early if diarization isn't enabled or we have no audio/model
        if not self.global_args.enable_diarization or self.tts_model is None or audio_buffer is None:
            return None

        temp_filename = None
        try:
            # The TTS model requires a file path, so we create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name

            # The recorder provides audio as a float32 NumPy array.
            # We must convert it to 16-bit integers to save it as a standard WAV file.
            audio_int16 = (audio_buffer * 32767).astype(np.int16)

            # Write the audio data to the temporary file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                wf.setframerate(16000)  # The recorder's sample rate is 16kHz
                wf.writeframes(audio_int16.tobytes())

            # Use the TTS model to get the speaker embedding from the audio file
            _, speaker_embedding = self.tts_model.get_conditioning_latents(
                audio_path=temp_filename,
                gpt_cond_len=30,
                max_ref_length=60
            )

            # Convert the resulting tensor to a simple NumPy array and return it
            return speaker_embedding.view(-1).cpu().detach().numpy()

        except Exception as e:
            print(f"{bcolors.FAIL}Error generating speaker embedding: {e}{bcolors.ENDC}")
            return None
        finally:
            # Ensure the temporary file is deleted after we're done with it
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)

    def _determine_optimal_cluster_count(self, embeddings_scaled):
        """
        Determines the best number of clusters (speakers) for the given embeddings.
        """
        num_embeddings = len(embeddings_scaled)
        if num_embeddings <= 1:
            return 1

        # First, use K-Means with k=2 to quickly check if it's just one speaker or more.
        # This is a fast way to handle the common case of 1-2 speakers.
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings_scaled)
        distances = kmeans.transform(embeddings_scaled)
        avg_distance = np.mean(np.min(distances, axis=1))

        print(f"\n[DIARIZATION DEBUG] Calculated avg_distance: {avg_distance:.4f} (Threshold is {self.global_args.diarization_speaker_threshold})\n")
        
        # If the average distance is below our threshold, we assume it's a single speaker.
        if avg_distance < self.global_args.diarization_speaker_threshold:
            debug_print(f"Single speaker detected: low embedding distance {avg_distance:.2f}")
            return 1

        # If it's likely more than one speaker, use a more detailed method.
        # We test clustering with 2, 3, 4... up to 10 speakers.
        max_clusters = min(10, num_embeddings)
        range_clusters = range(2, max_clusters + 1)
        silhouette_scores = []

        for n_clusters in range_clusters:
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = hc.fit_predict(embeddings_scaled)

            # A silhouette score measures how similar an object is to its own cluster
            # compared to other clusters. A higher score is better.
            if 1 < len(set(cluster_labels)) < num_embeddings:
                score = silhouette_score(embeddings_scaled, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1) # Invalid clustering

        # We look for the point where adding another cluster (speaker) doesn't significantly
        # improve the silhouette score. This prevents creating too many speakers.
        optimal_cluster_count = 2
        for i in range(1, len(silhouette_scores)):
            improvement = silhouette_scores[i] - silhouette_scores[i-1]
            if improvement < self.global_args.diarization_silhouette_threshold:
                optimal_cluster_count = range_clusters[i-1]
                break
        
        debug_print(f"Optimal cluster count determined to be: {optimal_cluster_count}")
        return optimal_cluster_count

    def _process_speakers(self):
        """
        Main diarization logic. Extracts embeddings, scales them, finds the
        optimal number of speakers, and assigns a speaker ID to each sentence.
        """
        # Get all embeddings, filtering out any 'None' values from failed generations.
        embeddings = [emb for _, emb in self.full_sentences if emb is not None]
        # Check if we have embeddings to process
        if len(embeddings) == 0:
            self.sentence_speakers = []
            return
            
        # Standard scaling
        embeddings_array = np.array(embeddings)
        
        # Check if we have only one embedding
        if len(embeddings_array) == 1:
            self.sentence_speakers = [0]
            return

        # Scale the data so that all features have a mean of 0 and std deviation of 1.
        # This is crucial for distance-based algorithms like clustering.
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        optimal_n_clusters = self._determine_optimal_cluster_count(embeddings_scaled)

        # Perform the final clustering with the optimal number of speakers.
        hc = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
        clusters = hc.fit_predict(embeddings_scaled)
        
        # We need to map these cluster labels back to the original full_sentences list,
        # accounting for any sentences that had a 'None' embedding.
        speaker_map = []
        embedding_idx = 0
        for _, emb in self.full_sentences:
            if emb is not None:
                speaker_map.append(clusters[embedding_idx])
                embedding_idx += 1
            else:
                # Assign a default/unknown speaker ID if embedding failed.
                speaker_map.append(-1) 
        
        self.sentence_speakers = speaker_map

    def run(self):
        """
        Main loop of the thread. Consumes data packets and processes them.
        """
        debug_print(f"FullSentenceProcessorThread started.")
        try:
            while not self.stop_event.is_set():
                try:
                    # Block for a short time to allow checking stop_event
                    data_packet = self.full_sentence_queue.get(timeout=0.5)
                except queue.Empty:
                    # Timeout occurred, loop will check stop_event again
                    continue

                # Process the received data packet
                self._process_data_packet(data_packet)

                # Mark the task as done
                self.full_sentence_queue.task_done()

        except Exception as e:
            print(f"{bcolors.FAIL}Error in FullSentenceProcessorThread: {e}{bcolors.ENDC}")
        finally:
            debug_print(f"FullSentenceProcessorThread stopped.")

    # --- MK3: Contextual Consistency Verification Method ---
    def _verify_transcription_consistency(self, current_text, current_audio, previous_audio):
        """
        Verifies if the current transcription is consistent with a combined transcription
        of the previous and current audio segments.
        
        Args:
            current_text (str): The transcription of the current audio segment.
            current_audio (np.ndarray): Audio buffer of the current segment.
            previous_audio (np.ndarray): Audio buffer of the previous segment.
            
        Returns:
            tuple: (is_consistent: bool, combined_transcription: str, word_overlap: float)
        """
        if self.verification_model is None or current_audio is None:
            return True, "", 1.0  # Skip verification if model not available
        
        try:
            # Combine the previous and current audio
            if previous_audio is not None:
                combined_audio = np.concatenate([previous_audio, current_audio])
            else:
                combined_audio = current_audio
            
            # Transcribe the combined audio
            language = self.recorder.language if self.recorder and self.recorder.language else None
            segments, info = self.verification_model.transcribe(
                combined_audio,
                language=language,
                beam_size=3,
                vad_filter=True
            )
            combined_transcription = " ".join([seg.text.strip() for seg in segments])
            
            # Calculate word overlap
            current_words = set(current_text.lower().split())
            combined_words = set(combined_transcription.lower().split())
            
            if len(current_words) == 0:
                return True, combined_transcription, 1.0
            
            overlap = len(current_words.intersection(combined_words))
            word_overlap_ratio = overlap / len(current_words)
            
            # Check if current text content appears in combined transcription
            is_consistent = word_overlap_ratio >= self.global_args.verification_word_overlap_threshold
            
            return is_consistent, combined_transcription, word_overlap_ratio
            
        except Exception as e:
            print(f"{bcolors.WARNING}Verification error: {e}{bcolors.ENDC}")
            return True, "", 1.0  # On error, don't block the transcription
    # --- END MK3 Verification Method ---

    # --- MK4: Combined Transcription Text Extraction ---
    def _extract_corrected_text(self, combined_text, previous_text, current_text):
        """
        Extracts the corrected version of current_text from combined transcription.
        Uses fuzzy matching to find where the new content starts.
        
        Args:
            combined_text (str): Full transcription of combined (previous + current) audio.
            previous_text (str): The previously transcribed text.
            current_text (str): The original transcription of current segment.
            
        Returns:
            str: Corrected text extracted from combined transcription, 
                 or original current_text if extraction fails.
        """
        if not previous_text or not combined_text:
            return current_text
        
        combined_lower = combined_text.lower().strip()
        prev_lower = previous_text.lower().strip()
        
        # Safety check: if combined is essentially the same as previous, no new content
        # This means the "current" audio was noise/silence with no real speech
        if combined_lower == prev_lower:
            debug_print(f"Combined == previous, no new content")
            return ""  # Return empty to signal no valid content
        
        # Check word overlap between combined and previous
        combined_words = combined_lower.split()
        prev_words = prev_lower.split()
        
        # If combined has same or fewer words than previous, likely no new content
        if len(combined_words) <= len(prev_words):
            # Check if they're very similar
            overlap = len(set(combined_words) & set(prev_words))
            if overlap >= len(prev_words) * 0.8:
                debug_print(f"Combined has no new words beyond previous")
                return ""  # No new content
        
        # Strategy 1: Try to find where previous ends in combined and extract the rest
        # Look for the last few words of previous in combined
        min_match_words = min(3, len(prev_words))  # Match at least 3 words or all if fewer
        
        for match_len in range(min(5, len(prev_words)), min_match_words - 1, -1):
            # Get the last N words of previous
            prev_suffix_words = prev_words[-match_len:]
            prev_suffix = ' '.join(prev_suffix_words)
            
            # Find this suffix in combined
            pos = combined_lower.find(prev_suffix)
            if pos != -1:
                # Extract everything after this suffix
                end_pos = pos + len(prev_suffix)
                remaining = combined_text[end_pos:].strip()
                
                # Validate: must be at least 3 words and not a fragment
                remaining_words = remaining.split()
                if len(remaining_words) >= 2:
                    debug_print(f"Suffix match ('{prev_suffix}'), extracted: '{remaining[:50]}...'")
                    return remaining
        
        # Strategy 2: If combined is longer and starts differently, maybe previous was wrong
        # Just return the full combined if it looks substantially different
        if len(combined_words) > len(prev_words) + 2:
            # Check if the extra words are at the end
            extra_words = combined_words[len(prev_words):]
            if len(extra_words) >= 2:
                extracted = ' '.join(combined_text.split()[len(prev_words):])
                debug_print(f"Word count extraction: '{extracted[:50]}...'")
                return extracted
        
        # Fallback: if combined looks totally different, return it entirely
        # (this handles when previous_text was also wrong)
        common_words = len(set(combined_words) & set(prev_words))
        if common_words < len(combined_words) * 0.3:
            debug_print(f"Combined very different from prev, using full combined")
            return combined_text
        
        # Final fallback: return empty to signal extraction failed
        debug_print(f"No extraction match found, no valid content")
        return ""
    # --- END MK4 Text Extraction ---

    def _should_trigger_verification(self, sentence_text, sentence_index):
        """
        Determines if this sentence warrants contextual consistency verification.
        Uses simple trigger conditions instead of a suspicion score.
        
        Args:
            sentence_text (str): The transcribed text.
            sentence_index (int): The 0-based index of this sentence.
            
        Returns:
            bool: True if verification should be triggered, False otherwise.
        """
        # Trigger 1: Always verify first N sentences (hallucinations common at start)
        first_n = getattr(self.global_args, 'verification_first_n_sentences', 2)
        if sentence_index < first_n:
            debug_print(f"Trigger: sentence_index={sentence_index} < first_n={first_n}")
            return True
        
        # Trigger 2: Contains known hallucination keywords
        text_lower = sentence_text.lower()
        for keyword in HALLUCINATION_KEYWORDS:
            if keyword in text_lower:
                debug_print(f"Trigger: hallucination keyword '{keyword}' detected")
                return True
        
        # Trigger 3: Very short sentence (≤3 words) - may indicate hallucination
        word_count = len(sentence_text.split())
        if word_count <= 3:
            debug_print(f"Trigger: short sentence (word_count={word_count})")
            return True
        
        return False
    
    # In class FullSentenceProcessorThread:
    def _process_data_packet(self, data_packet):
        """
        Processes a single data packet, performs diarization, and sends a
        'diarization_update' message containing the new sentence and any corrections.
        """
        sentence_index = data_packet.get('index')
        sentence_text = data_packet.get('text', '')
        audio_buffer = data_packet.get('audio_buffer')

        debug_print(f"FSPT processing sentence index {sentence_index}")

        # --- Audio Saving Logic ---
        if self.global_args.audio_log_dir and audio_buffer is not None:
            try:
                # Ensure the directory exists
                os.makedirs(self.global_args.audio_log_dir, exist_ok=True)
                
                # Construct a filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(self.global_args.audio_log_dir, f"transcription_{sentence_index}_{timestamp}.wav")
                
                # Convert float32 buffer to int16 bytes
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # Submit the saving task to the executor
                self.shared_executor.submit(
                    _save_audio_file,
                    filename,
                    audio_bytes,
                    CHANNELS, 
                    2, # sample_width (2 bytes for 16-bit)
                    16000 # framerate
                )
                debug_print(f"Scheduled audio saving for sentence {sentence_index}")
            except Exception as e:
                print(f"{bcolors.FAIL}Error scheduling audio save: {e}{bcolors.ENDC}")
        # --- End Audio Saving Logic ---

        # --- MK3: CONTEXTUAL CONSISTENCY VERIFICATION (with simple triggers) ---
        # Only run verification if triggered by heuristics, rather than on every sentence
        # MK4: Initialize variables with defaults in case verification is skipped
        combined_text = ""
        word_overlap = 1.0  # Default to 1.0 (perfect overlap) when skipping verification
        
        # MK4: Get density_suspicious flag from data packet
        density_suspicious = data_packet.get('density_suspicious', False)
        
        if self.global_args.enable_verification and self.verification_model is not None:
            should_verify = self._should_trigger_verification(sentence_text, sentence_index)
            
            # MK4: Also trigger verification if density was suspicious
            if density_suspicious:
                debug_print(f"Trigger: density_suspicious flag set - forcing verification")
                should_verify = True
            
            if should_verify:
                # --- Handle Segment 0: Double Verification ---
                # When segment 0 arrives: verify with itself first, buffer it for re-verification with segment 1
                if sentence_index == 0:
                    # Verify segment 0 with only its own audio
                    is_consistent, combined_text, word_overlap = self._verify_transcription_consistency(
                        sentence_text,
                        audio_buffer,
                        None  # No previous audio for segment 0
                    )
                    
                    if not is_consistent:
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        print(f"\n[{timestamp}] {bcolors.WARNING}[HALLUCINATION DETECTED] Discarded segment 0: \"{sentence_text}\" "
                              f"(word overlap: {word_overlap:.1%}){bcolors.ENDC}")
                        self.previous_audio_buffer = audio_buffer
                        self.segment_0_buffer = None  # Clear buffer since we discarded
                        return  # Exit early
                    else:
                        debug_print(f"Segment 0 initial verification PASSED: word overlap {word_overlap:.1%}")
                        # Buffer segment 0 for re-verification when segment 1 arrives
                        self.segment_0_buffer = {
                            'text': sentence_text,
                            'audio': audio_buffer,
                            'was_broadcast': True  # Will be broadcast after this
                        }
                
                # --- Handle Segment 1: Re-verify Segment 0 with combined context ---
                elif sentence_index == 1 and self.segment_0_buffer is not None:
                    # First, re-verify buffered segment 0 with combined (segment 0 + segment 1) audio
                    segment_0_text = self.segment_0_buffer['text']
                    segment_0_audio = self.segment_0_buffer['audio']
                    combined_audio = np.concatenate([segment_0_audio, audio_buffer])
                    
                    # Re-transcribe combined audio and check if segment 0 appears in it
                    is_seg0_consistent, combined_text, seg0_overlap = self._verify_transcription_consistency(
                        segment_0_text,
                        combined_audio,
                        None  # Using combined audio directly
                    )
                    
                    if not is_seg0_consistent:
                        # Segment 0 was a false positive! Send correction to client
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        print(f"\n[{timestamp}] {bcolors.WARNING}[RETROACTIVE HALLUCINATION] Segment 0 failed re-verification: \"{segment_0_text}\" "
                              f"(overlap in combined: {seg0_overlap:.1%}){bcolors.ENDC}")
                        # TODO: Send correction message to client to remove segment 0
                        # For now, log it - correction mechanism can be added later
                    
                    # Clear the segment 0 buffer
                    self.segment_0_buffer = None
                    
                    # Now verify segment 1 normally
                    is_consistent, combined_text, word_overlap = self._verify_transcription_consistency(
                        sentence_text,
                        audio_buffer,
                        self.previous_audio_buffer
                    )
                    
                    if not is_consistent:
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        print(f"\n[{timestamp}] {bcolors.WARNING}[HALLUCINATION DETECTED] Original: \"{sentence_text}\" "
                              f"(word overlap: {word_overlap:.1%}){bcolors.ENDC}")
                        
                        # --- MK4 FIX: Try to extract valid speech from combined transcription ---
                        if combined_text and len(combined_text.strip()) > 3:
                            extracted_text = self._extract_corrected_text(
                                combined_text,
                                self.previous_text,
                                sentence_text
                            )
                            clean_extracted = ''.join(c for c in extracted_text if c.isalnum())
                            if len(clean_extracted) >= 3:
                                print(f"  Combined: \"{combined_text[:60]}...\"")
                                print(f"  {bcolors.OKGREEN}[MK4 RECOVERY] Using extracted: \"{extracted_text}\"{bcolors.ENDC}")
                                sentence_text = extracted_text
                            else:
                                if extracted_text == "":
                                    print(f"  {bcolors.WARNING}[MK4] No new content in combined (noise/silence), discarding{bcolors.ENDC}")
                                else:
                                    print(f"  {bcolors.WARNING}[MK4] Extraction too short ({clean_extracted}), discarding{bcolors.ENDC}")
                                self.previous_audio_buffer = audio_buffer
                                return
                        else:
                            self.previous_audio_buffer = audio_buffer
                            return
                    else:
                        debug_print(f"Verification PASSED: word overlap {word_overlap:.1%}")
                
                # --- Normal verification for other sentences ---
                else:
                    is_consistent, combined_text, word_overlap = self._verify_transcription_consistency(
                        sentence_text,
                        audio_buffer,
                        self.previous_audio_buffer
                    )
                    
                    if not is_consistent:
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        print(f"\n[{timestamp}] {bcolors.WARNING}[HALLUCINATION DETECTED] Original: \"{sentence_text}\" "
                              f"(word overlap: {word_overlap:.1%}){bcolors.ENDC}")
                        print(f"  Combined transcription: \"{combined_text[:100]}...\"" if len(combined_text) > 100 else f"  Combined transcription: \"{combined_text}\"")
                        
                        # --- MK4 FIX: Try to extract valid speech from combined transcription ---
                        if combined_text and len(combined_text.strip()) > 3:
                            # Extract the "new" portion from combined text
                            extracted_text = self._extract_corrected_text(
                                combined_text,
                                self.previous_text,
                                sentence_text  # fallback if extraction fails
                            )
                            
                            # Validate extracted text isn't just whitespace/punctuation
                            clean_extracted = ''.join(c for c in extracted_text if c.isalnum())
                            if len(clean_extracted) >= 3:
                                print(f"  {bcolors.OKGREEN}[MK4 RECOVERY] Using extracted: \"{extracted_text}\"{bcolors.ENDC}")
                                sentence_text = extracted_text
                                # Don't return - continue processing with extracted text
                            else:
                                if extracted_text == "":
                                    print(f"  {bcolors.WARNING}[MK4] No new content in combined (noise/silence), discarding{bcolors.ENDC}")
                                else:
                                    print(f"  {bcolors.WARNING}[MK4] Extraction too short ({clean_extracted}), discarding{bcolors.ENDC}")
                                self.previous_audio_buffer = audio_buffer
                                return
                        else:
                            # No combined text available, truly discard
                            self.previous_audio_buffer = audio_buffer
                            return
                    else:
                        debug_print(f"Verification PASSED: word overlap {word_overlap:.1%}")
            else:
                debug_print(f"Skipping verification for sentence {sentence_index} (no trigger)")
        
        # Update previous audio buffer for next verification
        self.previous_audio_buffer = audio_buffer
        # --- END MK3 VERIFICATION ---

        # --- MK4: COMBINED TRANSCRIPTION ENHANCEMENT ---
        # If verification ran and produced a combined_text, try to extract improved transcription
        original_sentence_text = sentence_text  # Keep original for comparison
        text_was_corrected = False
        
        # Check if verification ran and produced combined text
        if self.global_args.enable_verification and self.verification_model is not None:
            # Only use combined text if we have previous context and overlap isn't perfect
            if combined_text and self.previous_text and word_overlap < 0.95:
                corrected_text = self._extract_corrected_text(
                    combined_text,
                    self.previous_text,
                    sentence_text
                )
                # Only apply correction if it's meaningfully different
                if corrected_text != sentence_text and len(corrected_text) > 0:
                    # Check that corrected text isn't suspiciously short or long
                    original_len = len(sentence_text)
                    corrected_len = len(corrected_text)
                    if 0.3 * original_len <= corrected_len <= 2.0 * original_len:
                        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        print(f"\n[{timestamp}] {bcolors.OKCYAN}[MK4 TEXT CORRECTION] '{sentence_text[:40]}...' → '{corrected_text[:40]}...'{bcolors.ENDC}")
                        sentence_text = corrected_text
                        text_was_corrected = True
        
        # Update previous_text for next verification
        self.previous_text = sentence_text
        
        # Store in sentence history for potential future corrections
        self.sentence_history.append({
            'index': sentence_index,
            'text': sentence_text,
            'original_text': original_sentence_text,
            'audio': audio_buffer,
            'was_corrected': text_was_corrected
        })
        # --- END MK4 COMBINED TRANSCRIPTION ---

        # --- STEP 1: Store old assignments ---
        # We make a copy of the previous state. This is crucial for detecting changes later.
        old_assignments = list(self.sentence_speakers)

        # --- STEP 2: Process the new sentence and re-cluster ---
        # This block updates the thread's internal "source of truth".
        if self.global_args.enable_diarization:
            speaker_embedding = self._get_speaker_embedding(audio_buffer)
            self.full_sentences.append((sentence_text, speaker_embedding))
            self._process_speakers() # This updates self.sentence_speakers with the new, smarter results.
        
        # --- STEP 3: Find differences and build the update message ---
        # Compare the old state with the new source of truth to find what needs to be corrected on the client.
        updates = []
        for i in range(len(old_assignments)):
            # If a speaker ID for a past sentence has changed, log it as a required update.
            if old_assignments[i] != self.sentence_speakers[i]:
                updates.append({'index': i, 'speaker_id': int(self.sentence_speakers[i])})
                debug_print(f"Correction: Sentence index {i} changed from speaker {old_assignments[i]} to {self.sentence_speakers[i]}")

        # The speaker for the *new* sentence is the last one in our updated list.
        new_sentence_speaker_id = int(self.sentence_speakers[-1]) if self.sentence_speakers else -1

        # --- STEP 4: Construct the final message data ---
        # We now build the single, authoritative message for the client.
        # This message type, 'diarization_update', replaces the old 'fullSentence' type.
        message_data = {
            'type': 'diarization_update',
            'new_sentence': {
                'index': sentence_index,
                'text': sentence_text,
                'speaker_id': new_sentence_speaker_id
            },
            'updates': updates
        }

        # --- STEP 5: Route to translation or directly to broadcast queue ---
        if self.global_args.enable_translation:
            source_lang_code = map_lang_to_nllb(self.recorder.language)
            job = {
                'type': 'diarization_update', # Use the new type
                'source_lang': source_lang_code,
                'data': message_data # Nest the original data to preserve it
            }
            asyncio.run_coroutine_threadsafe(self.translation_queue.put(job), self.loop)
            debug_print(f"Scheduled translation job for diarization_update (index {sentence_index})")
        else:
            message_str = json.dumps(message_data)
            asyncio.run_coroutine_threadsafe(self.audio_queue.put(message_str), self.loop)
            debug_print(f"Scheduled message for diarization_update (index {sentence_index})")

        # --- Logging ---
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"\r[{timestamp}] {bcolors.BOLD}Speaker {new_sentence_speaker_id} | Sentence:{bcolors.ENDC} {bcolors.OKGREEN}{sentence_text}{bcolors.ENDC}", end="")
        if updates:
            print(f" {bcolors.WARNING}({len(updates)} corrections sent){bcolors.ENDC}\n")
        else:
            return
            # print("\n") # Just to ensure a newline after the sentence
    # def _audio_save_callback(self, future):
    #     """ Optional callback to handle the result of the audio saving task. """
    #     try:
    #         # This retrieves the result (None in _save_audio_file's case) or raises an exception
    #         result = future.result()
    #         # If _save_audio_file returned something or you wanted to confirm success
    #         # debug_print(f"Audio file saved successfully (callback). Result: {result}")
    #     except Exception as e:
    #         print(f"{bcolors.FAIL}Error in audio saving callback: {e}{bcolors.ENDC}")

    def stop(self):
        """
        Signals the thread to stop.
        """
        self.stop_event.set()

# --- END STEP 3 ---

def decode_and_resample(
        audio_data,
        original_sample_rate,
        target_sample_rate):

    # Decode 16-bit PCM data to numpy array
    if original_sample_rate == target_sample_rate:
        return audio_data

    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate the number of samples after resampling
    num_original_samples = len(audio_np)
    num_target_samples = int(num_original_samples * target_sample_rate /
                                original_sample_rate)

    # Resample the audio
    resampled_audio = resample(audio_np, num_target_samples)

    return resampled_audio.astype(np.int16).tobytes()

async def control_handler(request):
    # This is the new aiohttp way to handle WebSockets
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    debug_print(f"New control connection from {request.remote}")
    print(f"{bcolors.OKGREEN}Control client connected{bcolors.ENDC}")
    
    # --- MODIFICATION START ---
    # Declare all global variables you intend to modify at the top of the function
    global recorder
    global target_translation_language
    # --- MODIFICATION END ---

    control_connections.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                message = msg.data
                debug_print(f"Received control message: {message[:200]}...")
                if not recorder_ready.is_set():
                    print(f"{bcolors.WARNING}Recorder not ready{bcolors.ENDC}")
                    continue
                
                try:
                    command_data = json.loads(message)
                    command = command_data.get("command")
                    if command == "set_parameter":
                        parameter = command_data.get("parameter")
                        value = command_data.get("value")
                        if parameter in allowed_parameters and hasattr(recorder, parameter):
                            setattr(recorder, parameter, value)
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = value
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Set recorder.{parameter} to: {bcolors.OKBLUE}{value_formatted}{bcolors.ENDC}")
                            await ws.send_json({"status": "success", "message": f"Parameter {parameter} set to {value}"})
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (set_parameter){bcolors.ENDC}")
                                await ws.send_json({"status": "error", "message": f"Parameter {parameter} is not allowed (set_parameter)"})
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (set_parameter){bcolors.ENDC}")
                                await ws.send_json({"status": "error", "message": f"Parameter {parameter} does not exist (set_parameter)"})

                    elif command == "get_parameter":
                        parameter = command_data.get("parameter")
                        request_id = command_data.get("request_id")
                        if parameter in allowed_parameters and hasattr(recorder, parameter):
                            value = getattr(recorder, parameter)
                            if isinstance(value, float):
                                value_formatted = f"{value:.2f}"
                            else:
                                value_formatted = f"{value}"
                            value_truncated = value_formatted[:39] + "…" if len(value_formatted) > 40 else value_formatted
                            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            if extended_logging:
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Get recorder.{parameter}: {bcolors.OKBLUE}{value_truncated}{bcolors.ENDC}")
                            response = {"status": "success", "parameter": parameter, "value": value}
                            if request_id is not None:
                                response["request_id"] = request_id
                            await ws.send_json(response)
                        else:
                            if not parameter in allowed_parameters:
                                print(f"{bcolors.WARNING}Parameter {parameter} is not allowed (get_parameter){bcolors.ENDC}")
                                await ws.send_json({"status": "error", "message": f"Parameter {parameter} is not allowed (get_parameter)"})
                            else:
                                print(f"{bcolors.WARNING}Parameter {parameter} does not exist (get_parameter){bcolors.ENDC}")
                                await ws.send_json({"status": "error", "message": f"Parameter {parameter} does not exist (get_parameter)"})
                    elif command == "call_method":
                        method_name = command_data.get("method")
                        if method_name in allowed_methods:
                            method = getattr(recorder, method_name, None)
                            if method and callable(method):
                                args = command_data.get("args", [])
                                kwargs = command_data.get("kwargs", {})
                                method(*args, **kwargs)
                                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                print(f"  [{timestamp}] {bcolors.OKGREEN}Called method recorder.{bcolors.OKBLUE}{method_name}{bcolors.ENDC}")
                                await ws.send_json({"status": "success", "message": f"Method {method_name} called"})
                            else:
                                print(f"{bcolors.WARNING}Recorder does not have method {method_name}{bcolors.ENDC}")
                                await ws.send_json({"status": "error", "message": f"Recorder does not have method {method_name}"})
                        else:
                            print(f"{bcolors.WARNING}Method {method_name} is not allowed{bcolors.ENDC}")
                            await ws.send_json({"status": "error", "message": f"Method {method_name} is not allowed"})
                    elif command == "set_translation_language":
                        # The `global` keyword is no longer needed here
                        if not global_args.enable_translation:
                            await ws.send_json({"status": "error", "message": "Translation layer is not enabled on the server."})
                            continue
                        lang_code = command_data.get("language", "")
                        if lang_code == "":
                            target_translation_language = None
                            print(f"{bcolors.OKGREEN}Translation disabled by client.{bcolors.ENDC}")
                            await ws.send_json({"status": "success", "message": "Translation disabled"})
                        else:
                            # DMTS MK2: Normalize language code (supports ISO 639-1/2 and NLLB formats)
                            from language_codes import normalize_language_code
                            normalized_lang = normalize_language_code(lang_code)
                            target_translation_language = normalized_lang
                            print(f"{bcolors.OKGREEN}Translation target language set to: {normalized_lang} (input: {lang_code}){bcolors.ENDC}")
                            await ws.send_json({"status": "success", "message": f"Translation target set to {normalized_lang}"})
                    else:
                        print(f"{bcolors.WARNING}Unknown command: {command}{bcolors.ENDC}")
                        await ws.send_json({"status": "error", "message": f"Unknown command {command}"})
                except json.JSONDecodeError:
                    print(f"{bcolors.WARNING}Received invalid JSON command{bcolors.ENDC}")
                    await ws.send_json({"status": "error", "message": "Invalid JSON command"})
            elif msg.type == WSMsgType.ERROR:
                print(f"{bcolors.FAIL}Control WebSocket connection closed with exception {ws.exception()}{bcolors.ENDC}")

    except Exception as e:
        print(f"{bcolors.FAIL}Error in control handler: {e}{bcolors.ENDC}")
    finally:
        print(f"{bcolors.WARNING}Control client disconnected.{bcolors.ENDC}")
        control_connections.discard(ws)
        # Reset the language on disconnect
        # The `global` keyword is no longer needed here either
        target_translation_language = global_args.translation_target_language
        # Stop the recorder if no clients are connected
        if not data_connections and not control_connections:
            recorder.stop()
            recorder.clear_audio_queue()

    return ws

async def data_handler(request):
    # This is the new aiohttp way to handle WebSockets
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    global writechunks, wav_file, shared_executor
    print(f"{bcolors.OKGREEN}Data client connected{bcolors.ENDC}")
    data_connections.add(ws)
    try:
        loop = asyncio.get_running_loop()
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                message = msg.data
                if extended_logging:
                    debug_print(f"Received audio chunk (size: {len(message)} bytes)")
                elif log_incoming_chunks:
                    print(".", end='', flush=True)

                # The rest of your logic for handling audio data is correct
                metadata_length = int.from_bytes(message[:4], byteorder='little')
                metadata_json = message[4:4+metadata_length].decode('utf-8')
                metadata = json.loads(metadata_json)
                sample_rate = metadata['sampleRate']

                if 'server_sent_to_stt' in metadata:
                    stt_received_ns = time.time_ns()
                    metadata["stt_received"] = stt_received_ns
                    metadata["stt_received_formatted"] = format_timestamp_ns(stt_received_ns)
                    print(f"Server received audio chunk of length {len(message)} bytes, metadata: {metadata}")

                if extended_logging:
                    debug_print(f"Processing audio chunk with sample rate {sample_rate}")
                chunk = message[4+metadata_length:]

                if writechunks:
                    if not wav_file:
                        wav_file = wave.open(writechunks, 'wb')
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wav_file.setframerate(sample_rate)
                    wav_file.writeframes(chunk)

                if sample_rate != 16000:
                    if not shared_executor:
                        print(f"{bcolors.FAIL}Executor not available, cannot resample audio.{bcolors.ENDC}")
                        continue
                    resampled_chunk = await loop.run_in_executor( 
                        shared_executor,
                        decode_and_resample,
                        chunk,
                        sample_rate,
                        16000
                    )
                    if extended_logging:
                        debug_print(f"Resampled chunk size: {len(resampled_chunk)} bytes")
                    recorder.feed_audio(resampled_chunk)
                else:
                    recorder.feed_audio(chunk)
            elif msg.type == WSMsgType.ERROR:
                print(f"{bcolors.FAIL}Data WebSocket connection closed with exception {ws.exception()}{bcolors.ENDC}")
            else:
                print(f"{bcolors.WARNING}Received non-binary message on data connection: {msg.type}{bcolors.ENDC}")

    except Exception as e:
        print(f"{bcolors.FAIL}Error in data handler: {e}{bcolors.ENDC}")
    finally:
        print(f"{bcolors.WARNING}Data client disconnected.{bcolors.ENDC}")
        data_connections.discard(ws)
        # Add this check to stop the recorder if no clients are connected
        if not data_connections and not control_connections:
            recorder.stop()
            recorder.clear_audio_queue()

    
    return ws

# Replace the existing broadcast_audio_messages function
async def broadcast_audio_messages(log_filename=None):
    """
    Continuously gets messages from the audio_queue and broadcasts them to all
    connected data clients. If a log_filename is provided, it also appends
    each message as a JSON line to the specified file.
    """
    log_file = None
    try:
        # --- 1. Open the log file if a path is provided ---
        if log_filename:
            # Open in "append" mode with utf-8 encoding
            log_file = open(log_filename, "a", encoding="utf-8")
            print(f"{bcolors.OKGREEN}Logging all transcription events to: {log_filename}{bcolors.ENDC}")

        while True:
            # Get the message string (e.g., from translation_task or FSPT)
            message_str = await audio_queue.get()
            
            # --- 2. Enrich the message with a server timestamp ---
            # It's good practice to add a server-side timestamp for logging.
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Safely parse and re-serialize to add the timestamp
            try:
                message_data = json.loads(message_str)
                message_data['server_timestamp'] = timestamp
                final_message_str_for_client = json.dumps(message_data)
            except json.JSONDecodeError:
                # If for some reason the message isn't valid JSON, log it as-is
                final_message_str_for_client = message_str

            # --- 3. Write to the log file ---
            if log_file:
                log_file.write(final_message_str_for_client + "\n")
                log_file.flush() # Ensure data is written to disk immediately

            # --- 4. Broadcast to clients (existing logic) ---
            if data_connections:
                # Make a copy to avoid issues if the set changes during iteration
                connections_to_broadcast = list(data_connections)
                for conn in connections_to_broadcast:
                    try:
                        await conn.send_str(final_message_str_for_client)
                    except Exception as e:
                        print(f"{bcolors.WARNING}Could not send to a client, removing connection: {e}{bcolors.ENDC}")
                        data_connections.discard(conn)

    except asyncio.CancelledError:
        print(f"{bcolors.OKCYAN}Broadcast task cancelled.{bcolors.ENDC}")
    finally:
        # --- 5. Ensure the file is closed on shutdown ---
        if log_file:
            log_file.close()
            print(f"{bcolors.OKGREEN}Transcription log file closed.{bcolors.ENDC}")

# Helper function to create event loop bound closures for callbacks
def make_callback(loop, callback):
    def inner_callback(*args, **kwargs):
        callback(*args, **kwargs, loop=loop)
    return inner_callback
# In stt_server.py, add this new async function

async def translation_processor_task(executor, loop):
    """
    Processes translation jobs from a queue in a separate thread pool.
    Now handles both 'realtime' and 'diarization_update' message types.
    """
    global target_translation_language, translation_manager
    # In async def translation_processor_task(...):
    while True:
        # 1. Get a job from the translation queue
        job = await translation_queue.get()
        
        # If translation is globally disabled by the client, we have a problem.
        # The job should not have been put on this queue. We'll pass it through as-is.
        if not target_translation_language or not translation_manager:
            if job['type'] == 'diarization_update':
                await audio_queue.put(json.dumps(job['data']))
            else: # Handle older message types just in case
                await audio_queue.put(json.dumps(job))
            continue

        # --- NEW LOGIC FOR DIARIZATION UPDATE ---
        if job['type'] == 'diarization_update':
            original_data = job['data']
            new_sentence_text = original_data['new_sentence']['text']
            source_language = job['source_lang']
            
            try:
                # Translate only the text from the new sentence
                translated_text = await loop.run_in_executor(
                    executor,
                    translation_manager.translate,
                    new_sentence_text,
                    source_language,
                    target_translation_language,
                    'full'  # Final sentences should use the high-quality model
                )
            except Exception as e:
                print(f"{bcolors.FAIL}Error during translation: {e}{bcolors.ENDC}")
                translated_text = "[Translation Error]"

            # --- Reconstruct the message for the client ---
            # Start with the original data structure
            final_message_data = original_data
            
            # Inject the translation into the new_sentence object
            final_message_data['new_sentence']['translation'] = {'text': translated_text}
            
            # --- MK4: Translate text corrections in updates array ---
            # If any update contains a 'text' field (text correction), translate it too
            if 'updates' in final_message_data:
                for update in final_message_data['updates']:
                    if 'text' in update and update['text']:
                        try:
                            update_translation = await loop.run_in_executor(
                                executor,
                                translation_manager.translate,
                                update['text'],
                                source_language,
                                target_translation_language,
                                'full'
                            )
                            update['translation'] = {'text': update_translation}
                            debug_print(f"Translated correction for index {update.get('index')}: {update_translation[:40]}...")
                        except Exception as e:
                            print(f"{bcolors.WARNING}Error translating correction: {e}{bcolors.ENDC}")
                            update['translation'] = {'text': '[Translation Error]'}
            # --- END MK4 ---
            
            # Put the final, enriched message on the audio_queue for broadcasting
            await audio_queue.put(json.dumps(final_message_data))
            
            # --- Logging ---
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            speaker_id = final_message_data['new_sentence']['speaker_id']
            print(f"[{timestamp}] {bcolors.BOLD}Speaker {speaker_id} | Translation:{bcolors.ENDC} {bcolors.OKBLUE}{translated_text}{bcolors.ENDC}")
            
            # Continue to the next job in the queue
            continue

        # --- EXISTING LOGIC for 'realtime' messages (should remain) ---
        if job['type'] == 'realtime':
            # Skip realtime translation if flag is set (for slow LLM backends)
            if global_args.skip_realtime_translation:
                # Broadcast original text without translation
                message_data = {
                    'type': 'realtime',
                    'text': job['text']
                }
                await audio_queue.put(json.dumps(message_data))
                continue
            
            text_to_translate = job['text']
            source_language = job['source_lang']
            
            try:
                translated_text = await loop.run_in_executor(
                    executor,
                    translation_manager.translate,
                    text_to_translate,
                    source_language,
                    target_translation_language,
                    'realtime' # Use the faster model for real-time updates
                )
            except Exception as e:
                print(f"{bcolors.FAIL}Error during translation: {e}{bcolors.ENDC}")
                translated_text = "[Translation Error]"

            # --- MK3: Translation Consistency Check ---
            # Only broadcast if source text is a valid continuation of previous (not a loop/jump)
            global previous_realtime_translation, previous_realtime_source
            source_text = text_to_translate
            
            if previous_realtime_source is not None:
                # Check if current source is a continuation of previous
                # A valid continuation: starts with previous prefix, OR is similar (refinement)
                is_continuation = False
                
                # Check prefix match (current extends previous)
                prefix_len = min(len(previous_realtime_source), 15)
                if source_text.startswith(previous_realtime_source[:prefix_len]):
                    is_continuation = True
                
                # Check if previous is contained in current (extension)
                if not is_continuation and len(previous_realtime_source) > 5:
                    if previous_realtime_source in source_text:
                        is_continuation = True
                
                # Check similarity for refinements (using SequenceMatcher)
                if not is_continuation:
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, previous_realtime_source, source_text).ratio()
                    threshold = getattr(global_args, 'translation_consistency_threshold', 0.3)
                    if similarity >= threshold:
                        is_continuation = True
                
                if not is_continuation:
                    # Source jumped - likely hallucination loop, skip broadcast
                    debug_print(f"Skipping realtime broadcast: source not continuation (prev='{previous_realtime_source[:30]}...' curr='{source_text[:30]}...')")
                    continue
            
            # Update state for next comparison
            previous_realtime_source = source_text
            previous_realtime_translation = translated_text
            # --- END MK3 Translation Consistency Check ---

            # Logging for real-time
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            original_text = job['text']
            output = (f"\r[{timestamp}] rt-sentence: {bcolors.OKCYAN}{original_text}{bcolors.ENDC}\n"
                    f"[{timestamp}] rt-translation: {bcolors.OKBLUE}{translated_text}{bcolors.ENDC}")
            print(output, flush=True, end="")

            # Construct message for client
            message_data = {
                'type': 'realtime',
                'text': original_text,
                'translation': {
                    'language': target_translation_language,
                    'text': translated_text
                }
            }
            await audio_queue.put(json.dumps(message_data))

def initialize_tts_model(args):
    """Initializes and returns the TTS model for diarization."""
    if not args.diarization_model_path:
        print(f"{bcolors.FAIL}Error: --diarization_model_path must be specified.{bcolors.ENDC}")
        sys.exit(1)
    
    print(f"{bcolors.OKCYAN}Initializing TTS model from: {args.diarization_model_path}{bcolors.ENDC}")
    try:
        device = torch.device("cpu")
        config = load_config(os.path.join(args.diarization_model_path, "config.json"))
        model = setup_tts_model(config)
        # --- ADD THIS BLOCK TO FIX THE LOADING ERROR ---
        # Add safe globals for PyTorch >= 2.6 compatibility
        try:
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        except AttributeError:
            # Older PyTorch versions don't have this function, so it's safe to ignore.
            pass
        # --- END OF ADDED BLOCK ---
        model.load_checkpoint(config, checkpoint_dir=args.diarization_model_path, eval=True)
        model.to(device)
        print(f"{bcolors.OKGREEN}TTS model for diarization initialized successfully on {device}.{bcolors.ENDC}")
        return model
    except Exception as e:
        print(f"{bcolors.FAIL}Failed to initialize TTS model: {e}{bcolors.ENDC}")
        sys.exit(1)

async def simple_data_handler(request):
    """
    A simplified WebSocket handler for external testing (e.g., Wang Sheng).
    It accepts RAW audio bytes (no metadata header required).
    Assumes 16000Hz, 1 Channel, Int16 audio.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    global writechunks, wav_file, shared_executor
    print(f"{bcolors.OKGREEN}External client connected to /stream{bcolors.ENDC}")
    data_connections.add(ws)
    stream_connections.add(ws)  # MK4: Also track for debug logging
    
    try:
        loop = asyncio.get_running_loop()
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                chunk = msg.data
                
                # Visual feedback that data is flowing
                if log_incoming_chunks:
                    print(".", end='', flush=True)

                # Direct feed to recorder (Assuming 16kHz source)
                # If Wang Sheng sends 44.1kHz or 48kHz, we might need resampling logic here.
                # For now, we assume he sends standard 16kHz voice data.
                recorder.feed_audio(chunk)

            elif msg.type == WSMsgType.ERROR:
                print(f"{bcolors.FAIL}Simple Data WS connection closed with exception {ws.exception()}{bcolors.ENDC}")

    except Exception as e:
        print(f"{bcolors.FAIL}Error in simple_data_handler: {e}{bcolors.ENDC}")
    finally:
        print(f"{bcolors.WARNING}External client disconnected.{bcolors.ENDC}")
        
        # --- FIX: Stop the "Zombie Loop" immediately ---
        if recorder:
            # 1. Force the VAD to trigger "End of Sentence" by feeding silence
            # We send 1.0 second of silence (16000Hz * 2 bytes * 1.0s = 32000 bytes)
            silence_chunk = b'\x00' * 32000 
            recorder.feed_audio(silence_chunk)
            
            # 2. Optional: Wait a tiny bit for the VAD to process it
            # (In async, we can't easily sleep here without 'await', 
            # but feeding the audio is usually enough to queue the stop).

        data_connections.discard(ws)
        stream_connections.discard(ws)  # MK4: Also remove from stream tracking
        
        # Stop recorder if no one is left (Standard Cleanup)
        if not data_connections and not control_connections:
            recorder.stop()
            recorder.clear_audio_queue()

    return ws

async def main_async():            
    global stop_recorder, recorder_config, global_args, translation_manager, target_translation_language, shared_executor, full_sentence_processor_thread, tts_model, verification_model # global args
    args = parse_arguments() # parse line arguments
    global_args = args 
    
    # Get the event loop here and pass it to the recorder thread
    loop = asyncio.get_event_loop() # event loop
    shared_executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix='JobExecutor') # for cpu intensive tasks like resampling and saving audio files

    # --- START DIARIZATION INITIALIZATION (Inside main_async, after args parsing) ---
    if args.enable_diarization:
        tts_model = initialize_tts_model(args)

    # --- MK3: VERIFICATION MODEL INITIALIZATION ---
    if args.enable_verification:
        print(f"{bcolors.OKCYAN}Initializing verification model from: {args.verification_model_path}{bcolors.ENDC}")
        try:
            from faster_whisper import WhisperModel
            verification_model = WhisperModel(
                args.verification_model_path,
                device=args.device,
                compute_type=args.verification_compute_type,
                device_index=args.gpu_device_index
            )
            print(f"{bcolors.OKGREEN}Verification model initialized successfully.{bcolors.ENDC}")
        except Exception as e:
            print(f"{bcolors.FAIL}Failed to initialize verification model: {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Contextual consistency verification will be disabled.{bcolors.ENDC}")
            verification_model = None
    # --- END MK3 VERIFICATION MODEL INITIALIZATION ---

    if args.enable_translation: # If translation is enabled, initialize the translation manager
        print(f"{bcolors.OKCYAN}Translation layer enabled.{bcolors.ENDC}")
        
        # Dynamically load the appropriate translation manager based on backend
        if args.translation_backend == 'hunyuan':
            from translation.manager_hunyuan import HunyuanTranslationManager
            translation_manager = HunyuanTranslationManager(
                model_path=args.translation_model,
                device=args.device,
                load_in_8bit=args.translation_load_in_8bit,
                gpu_device_index=args.translation_gpu_device
            )
        elif args.translation_backend == 'hybrid':
            from translation.manager_hybrid import HybridTranslationManager
            translation_manager = HybridTranslationManager(
                nllb_realtime_model_path=args.translation_model_realtime,
                nllb_full_model_path=args.translation_model_full,
                hunyuan_model_path=args.translation_model,
                device=args.device,
                hunyuan_load_in_8bit=args.translation_load_in_8bit,
                hunyuan_gpu_device=args.translation_gpu_device
            )
        else:  # Default to NLLB
            from translation.manager_nllb import TranslationManager
            translation_manager = TranslationManager(
                args.translation_model_realtime,
                args.translation_model_full,
                args.device
            )
        
        target_translation_language = args.translation_target_language

        # Start the task that processes the translation queue, using the new shared executor.
        asyncio.create_task(translation_processor_task(shared_executor, loop))

    # --- START OF MODIFIED CODE ---
    # Create the aiohttp web application to serve the HTML file and handle WebSockets
    app = web.Application()

    # Add the WebSocket handlers to the application routes
    app.router.add_get('/', handle_index)
    app.router.add_get('/control', control_handler) # Use existing control_handler
    app.router.add_get('/data', data_handler)     # Use existing data_handler
    app.router.add_get('/stream', simple_data_handler)
    
    # Configure CORS for all routes
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    runner = web.AppRunner(app)
    await runner.setup()
    web_server_port = args.port # Use the port from arguments
    site = web.TCPSite(runner, '0.0.0.0', web_server_port)
    # --- END OF MODIFIED CODE ---

    # recorder config
    recorder_config = {
        'model': args.model,
        'download_root': args.root,
        'realtime_model_type': args.rt_model,
        'language': args.lang,
        'batch_size': args.batch,
        'init_realtime_after_seconds': args.init_realtime_after_seconds,
        'realtime_batch_size': args.realtime_batch_size,
        'initial_prompt_realtime': args.initial_prompt_realtime,
        'input_device_index': args.input_device,
        'silero_sensitivity': args.silero_sensitivity,
        'silero_use_onnx': args.silero_use_onnx,
        'webrtc_sensitivity': args.webrtc_sensitivity,
        'post_speech_silence_duration': args.unknown_sentence_detection_pause,
        'min_length_of_recording': args.min_length_of_recording,
        'min_gap_between_recordings': args.min_gap_between_recordings,
        'pre_recording_buffer_duration': args.pre_recording_buffer_duration,
        'enable_realtime_transcription': args.enable_realtime_transcription,
        'realtime_processing_pause': args.realtime_processing_pause,
        'silero_deactivity_detection': args.silero_deactivity_detection,
        'early_transcription_on_silence': args.early_transcription_on_silence,
        'beam_size': args.beam_size,
        'beam_size_realtime': args.beam_size_realtime,
        'initial_prompt': args.initial_prompt,
        'wake_words': args.wake_words,
        'wake_words_sensitivity': args.wake_words_sensitivity,
        'wake_word_timeout': args.wake_word_timeout,
        'wake_word_activation_delay': args.wake_word_activation_delay,
        'wakeword_backend': args.wakeword_backend,
        'openwakeword_model_paths': args.openwakeword_model_paths,
        'openwakeword_inference_framework': args.openwakeword_inference_framework,
        'wake_word_buffer_duration': args.wake_word_buffer_duration,
        'use_main_model_for_realtime': args.use_main_model_for_realtime,
        'spinner': False,
        'use_microphone': False,

        'on_realtime_transcription_update': make_callback(loop, text_detected),
        'on_recording_start': make_callback(loop, on_recording_start),
        'on_recording_stop': make_callback(loop, on_recording_stop),
        'on_vad_detect_start': make_callback(loop, on_vad_detect_start),
        'on_vad_detect_stop': make_callback(loop, on_vad_detect_stop),
        'on_wakeword_detected': make_callback(loop, on_wakeword_detected),
        'on_wakeword_detection_start': make_callback(loop, on_wakeword_detection_start),
        'on_wakeword_detection_end': make_callback(loop, on_wakeword_detection_end),
        'on_transcription_start': make_callback(loop, on_transcription_start),
        'on_turn_detection_start': make_callback(loop, on_turn_detection_start),
        'on_turn_detection_stop': make_callback(loop, on_turn_detection_stop),

        # 'on_recorded_chunk': make_callback(loop, on_recorded_chunk),
        'no_log_file': True,  # Disable logging to file
        'use_extended_logging': args.use_extended_logging,
        'level': loglevel,
        'compute_type': args.compute_type,
        'gpu_device_index': args.gpu_device_index,
        'device': args.device,
        'handle_buffer_overflow': args.handle_buffer_overflow,
        'suppress_tokens': args.suppress_tokens,
        'allowed_latency_limit': args.allowed_latency_limit,
        'faster_whisper_vad_filter': args.faster_whisper_vad_filter,
    }

    try:
        await site.start()
        print(f"{bcolors.OKGREEN}Web server started on {bcolors.OKBLUE}http://localhost:{web_server_port}{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Control and Data WebSockets available on the same port at /control and /data paths.{bcolors.ENDC}")
        broadcast_task = asyncio.create_task(broadcast_audio_messages(args.transcription_log))

             # --- STEP 4: Instantiate FullSentenceProcessorThread ---
        # Create the instance *before* starting _recorder_thread
        # Pass all required dependencies
        # Note: recorder_ref is initially None, it will be set after recorder is ready
        global full_sentence_processor_thread # Declare global usage here as well for clarity
        full_sentence_processor_thread = FullSentenceProcessorThread(
            full_sentence_queue=full_sentence_queue,       # The queue we created in Step 1
            audio_queue=audio_queue,                       # For non-translated messages
            translation_queue=translation_queue,           # For translation jobs
            shared_executor=shared_executor,               # For offloading tasks like audio saving
            global_args=args,                              # Command line arguments
            recorder_ref=None,                             # Initially None, will be set later
            loop=loop,                                      # Main asyncio event loop
            tts_model=tts_model,                           # Pass the loaded TTS model instance
            verification_model=verification_model          # MK3: Pass the verification model
        )
        # --- END STEP 4 Instantiation ---

        # Start the original recorder thread
        recorder_thread = threading.Thread(target=_recorder_thread, args=(loop,))
        recorder_thread.start()
        recorder_ready.wait() # Wait for recorder to be ready

        # --- STEP 4: Set Recorder Reference & Start FullSentenceProcessorThread ---
        # Now that recorder is initialized, pass its reference to the processor thread
        full_sentence_processor_thread.recorder = recorder
        # Start the new thread
        full_sentence_processor_thread.start()
        print(f"{bcolors.OKGREEN}FullSentenceProcessorThread started.{bcolors.ENDC}")
        # --- END STEP 4 Set Recorder & Start ---
        await broadcast_task

    except OSError as e:
        print(f"{bcolors.FAIL}Error: Could not start server on port {web_server_port}. It’s possible another instance is already running or the port is in use.{bcolors.ENDC}")
    except KeyboardInterrupt:
        print(f"{bcolors.WARNING}Server interrupted by user, shutting down...{bcolors.ENDC}")
    finally:
        await runner.cleanup()
        await shutdown_procedure() # This needs modification
        print("...")

# --- STEP 4: Modify shutdown_procedure ---
async def shutdown_procedure():
    global stop_recorder, recorder_thread, shared_executor, full_sentence_processor_thread # Add full_sentence_processor_thread
    # --- Signal and Join FullSentenceProcessorThread ---
    if 'full_sentence_processor_thread' in globals() and full_sentence_processor_thread and full_sentence_processor_thread.is_alive():
        print(f"{bcolors.OKGREEN}Stopping FullSentenceProcessorThread...{bcolors.ENDC}")
        full_sentence_processor_thread.stop() # Signal the thread to stop
        full_sentence_processor_thread.join() # Wait for it to finish
        print(f"{bcolors.OKGREEN}FullSentenceProcessorThread finished.{bcolors.ENDC}")
    # --- END FullSentenceProcessorThread Shutdown ---

    # --- Existing Recorder Shutdown (should remain largely the same) ---
    if recorder:
        stop_recorder = True
        recorder.abort()
        recorder.stop()
        recorder.shutdown()
        print(f"{bcolors.OKGREEN}Recorder shut down{bcolors.ENDC}")
        if recorder_thread:
            recorder_thread.join()
            print(f"{bcolors.OKGREEN}Recorder thread finished{bcolors.ENDC}")
    # --- END Recorder Shutdown ---

    # --- Existing Executor and Task Shutdown (should remain the same) ---
    if shared_executor:
        shared_executor.shutdown(wait=True)
        print(f"{bcolors.OKGREEN}Shared job executor shut down{bcolors.ENDC}")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"{bcolors.OKGREEN}All tasks cancelled, closing event loop now.{bcolors.ENDC}")
    # --- END Existing Shutdown ---
# --- END STEP 4 shutdown_procedure ---

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        # Capture any final KeyboardInterrupt to prevent it from showing up in logs
        print(f"{bcolors.WARNING}Server interrupted by user.{bcolors.ENDC}")
        exit(0)

if __name__ == '__main__':
    main()

'''
python /home/zhouzhencheng/realtime_mt/Real-time_STT/RealtimeSTT_calvin/stt_server_mt_public_5_diarization_mk3_v3.py --diarization_model_path /home/zhouzhencheng/realtime_mt/Real-time_STT/WhoSpeaks/XTTS-v2/v2.0.2 --audio-log-dir="./saved_audio_mk3_v3" --transcription-log="transcript_mk3_v3.log"
'''