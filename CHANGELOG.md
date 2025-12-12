# DMTS MK4 Changelog

## Version 4.0.0 (2025-12-11)

### New Feature: Combined Transcription Enhancement & MK4 Recovery

DMTS MK4 introduces **intelligent transcription recovery** that uses verified combined audio to extract real speech even when hallucinations are detected.

---

### Combined Transcription Output

When contextual verification detects a potential hallucination, MK4 now attempts to **recover valid speech** from the combined transcription instead of simply discarding the sentence.

#### How It Works

1. **Hallucination Detected:** Original transcription doesn't match combined audio
2. **Extract New Portion:** Use `_extract_corrected_text` to find actual new content
3. **Validate Extraction:** Ensure extracted text is ≥2 words (not fragments)
4. **MK4 Recovery:** Use extracted text as the real transcription

#### Example

```
[HALLUCINATION DETECTED] Original: "Thank you." (word overlap: 0.0%)
  Combined transcription: "We'll have to do another evacuation."
  [MK4 RECOVERY] Using extracted: "We'll have to do another evacuation."
Speaker 0 | Sentence: We'll have to do another evacuation.
```

---

### Speech Density Integration

Previously, the speech density filter would **discard** suspicious sentences immediately. MK4 now **flags** them for verification, allowing recovery of valid speech.

| Before (MK3) | After (MK4) |
|-------------|-------------|
| `[HALLUCINATION FILTER] Discarded` | `[DENSITY SUSPICIOUS] flagged for MK4 verification` |
| Sentence lost forever | Sentence goes through MK4 recovery |

#### Console Output Examples

**Successful MK4 Recovery:**
```
[DENSITY SUSPICIOUS] "Thank you." (density: 29.9 chars/sec) - flagged for MK4 verification
[HALLUCINATION DETECTED] Original: "Thank you." (word overlap: 0.0%)
  Combined transcription: "We'll have to do another evacuation."
  [MK4 RECOVERY] Using extracted: "We'll have to do another evacuation."
Speaker 0 | Sentence: We'll have to do another evacuation.
```

**No New Content (Noise/Silence):**
```
[HALLUCINATION DETECTED] Original: "Um..." (word overlap: 0.0%)
  Combined transcription: "halo selamat malam"
  [MK4] No new content in combined (noise/silence), discarding
```

---

### Text Correction Protocol

MK4 extends the `updates` array in `diarization_update` messages to support **text and translation corrections**, not just speaker ID changes.

#### Extended Updates Array

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
            "text": "corrected text here",           // NEW in MK4
            "translation": {"text": "corrected 翻译"}  // NEW in MK4
        }
    ]
}
```

Clients can now receive retroactive text corrections when combined transcription provides better accuracy.

---

### /stream Debug Logging

The `/stream` endpoint now receives all transcription broadcasts for debugging:

- Diarization updates (new sentences + corrections)
- Translation updates
- Text corrections

Connect to `/stream` to monitor all server output in real-time.

---

### New CLI Arguments

No new CLI arguments. MK4 uses the same verification arguments as MK3:

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_verification` | `True` | Enable contextual verification + MK4 recovery |
| `--verification_model_path` | `...` | Path to verification Whisper model |
| `--verification_word_overlap_threshold` | `0.3` | Minimum word overlap for passing |

---

### Inherited from MK3

- Contextual Consistency Verification
- Heuristic-Triggered Verification (keywords, short sentences, first N)
- Double-Verification for Segment 0
- Translation Consistency Check
- Speech Density Heuristic

### Inherited from MK2

- ISO language code support (zh, ja, ko, etc.)
- Hybrid translation backend (NLLB + Hunyuan)
- Speaker diarization

---

## Files Changed

| File | Changes |
|------|---------|
| `dmts_mk4.py` | Added `_extract_corrected_text`, MK4 recovery logic, `density_suspicious` flag, `stream_connections` |
| `index_4.html` | Updated `handleDiarizationUpdate` to use text/translation from updates |
| `run_server*.sh` | All scripts point to `dmts_mk4.py` |

---

## Migration from MK3

MK4 is fully backward compatible with MK3 clients. The extended `updates` array fields are optional - clients that don't handle them will continue to work.

**To upgrade:**
1. Replace `dmts_mk3.py` with `dmts_mk4.py`
2. Update run scripts to reference `dmts_mk4.py`
3. (Optional) Update client to handle `text` and `translation` in `updates` array

---

*Created: December 11, 2025 | Version: DMTS MK4*
