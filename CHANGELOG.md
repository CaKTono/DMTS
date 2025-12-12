# DMTS Changelog

## Version 4.0.0 (2025-12-11)

Initial open-source release of DMTS MK4.

### Features

#### Intelligent Transcription Recovery

DMTS MK4 introduces **intelligent transcription recovery** that uses verified combined audio to extract real speech even when hallucinations are detected.

**How It Works:**

1. **Hallucination Detected:** Original transcription doesn't match combined audio
2. **Extract New Portion:** Use `_extract_corrected_text` to find actual new content
3. **Validate Extraction:** Ensure extracted text is ≥2 words (not fragments)
4. **MK4 Recovery:** Use extracted text as the real transcription

**Example:**
```
[HALLUCINATION DETECTED] Original: "Thank you." (word overlap: 0.0%)
  Combined transcription: "We'll have to do another evacuation."
  [MK4 RECOVERY] Using extracted: "We'll have to do another evacuation."
Speaker 0 | Sentence: We'll have to do another evacuation.
```

---

#### Speech Density Filtering

Suspicious sentences are **flagged** for verification instead of immediate discard, allowing recovery of valid speech.

**Console Output Examples:**

```
[DENSITY SUSPICIOUS] "Thank you." (density: 29.9 chars/sec) - flagged for MK4 verification
[HALLUCINATION DETECTED] Original: "Thank you." (word overlap: 0.0%)
  Combined transcription: "We'll have to do another evacuation."
  [MK4 RECOVERY] Using extracted: "We'll have to do another evacuation."
Speaker 0 | Sentence: We'll have to do another evacuation.
```

---

#### Text Correction Protocol

The `updates` array in `diarization_update` messages supports **text and translation corrections**, not just speaker ID changes.

**Extended Updates Array:**

```json
{
    "type": "diarization_update",
    "new_sentence": {
        "index": 5,
        "text": "...",
        "speaker_id": 1,
        "translation": {"text": "..."}
    },
    "updates": [
        {
            "index": 3,
            "speaker_id": 0,
            "text": "corrected text here",
            "translation": {"text": "corrected translation"}
        }
    ]
}
```

Clients can receive retroactive text corrections when combined transcription provides better accuracy.

---

#### Debug Stream Endpoint

The `/stream` endpoint receives all transcription broadcasts for debugging:

- Diarization updates (new sentences + corrections)
- Translation updates
- Text corrections

Connect to `/stream` to monitor all server output in real-time.

---

### Core Features

- **Real-time Transcription**: Live speech-to-text using Whisper models
- **Speaker Diarization**: Identifies speakers using Coqui TTS embeddings + agglomerative clustering
- **Multilingual Translation**: NLLB (200+ languages), Hunyuan (38 languages), or Hybrid backend
- **ISO Language Code Support**: Use simple codes like `zh`, `ja`, `ko` instead of full NLLB codes
- **Contextual Consistency Verification**: Detects Whisper hallucinations
- **Heuristic-Triggered Verification**: Only verifies suspicious sentences to reduce latency
- **Translation Consistency Check**: Prevents "jumpy" real-time translations

---

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_verification` | `True` | Enable contextual verification + MK4 recovery |
| `--verification_model_path` | `...` | Path to verification Whisper model |
| `--verification_word_overlap_threshold` | `0.3` | Minimum word overlap for passing |
| `--translation_backend` | `nllb` | Translation backend: `nllb`, `hunyuan`, or `hybrid` |
| `--enable_diarization` | `False` | Enable speaker diarization |
| `--enable_translation` | `False` | Enable translation layer |

---

*Initial Release: December 11, 2025*
