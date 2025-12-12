"""
NLLB Translation Manager

This module provides NLLB-based translation with support for 200+ languages.
Uses separate models for real-time (fast) and final (accurate) translation.
"""

from transformers import pipeline
import torch


class TranslationManager:
    """
    NLLB-based translation manager.

    Uses two models:
    - Real-time: NLLB-600M (fast, ~200ms)
    - Final: NLLB-3.3B (more accurate)
    """

    def __init__(self, realtime_model_name, full_model_name, device):
        """
        Initialize NLLB translation manager.

        Args:
            realtime_model_name: Path to NLLB-600M model
            full_model_name: Path to NLLB-3.3B model
            device: 'cuda' or 'cpu'
        """
        print("Loading NLLB translation models...")
        self.device = 0 if device == 'cuda' and torch.cuda.is_available() else -1

        print(f"Loading real-time translation model: {realtime_model_name}")
        self.realtime_translator = pipeline(
            'translation',
            model=realtime_model_name,
            device=self.device
        )

        print(f"Loading full-sentence translation model: {full_model_name}")
        self.full_translator = pipeline(
            'translation',
            model=full_model_name,
            device=self.device
        )
        print("NLLB translation models loaded.")

    def translate(self, text, src_lang, tgt_lang, model_type='realtime'):
        """
        Translate text using NLLB models.

        Args:
            text: Source text to translate
            src_lang: Source language code (NLLB format, e.g., 'eng_Latn')
            tgt_lang: Target language code (NLLB format, e.g., 'zho_Hans')
            model_type: 'realtime' for fast or 'full' for accurate

        Returns:
            Translated text string
        """
        if not text or not text.strip():
            return ""

        if model_type == 'realtime':
            translator = self.realtime_translator
        else:
            translator = self.full_translator

        result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        return result[0]['translation_text']
