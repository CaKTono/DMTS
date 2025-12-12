"""
Hybrid Translation Manager with Smart Language Fallback

This module provides a hybrid translation approach:
- Real-time: Always NLLB-600M (fast ~200ms)
- Final: Hunyuan-MT-7B if target language is supported, otherwise NLLB-3.3B

Hunyuan-MT-7B supports 38 languages (see HUNYUAN_SUPPORTED_LANGUAGES).
NLLB supports ~200 languages.
"""

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Import centralized language mappings
from language_codes import (
    NLLB_TO_LANGUAGE_NAME,
    CHINESE_CODES,
    HUNYUAN_SUPPORTED_LANGUAGES,
    get_language_name,
    is_hunyuan_supported,
    is_chinese
)


class HybridTranslationManager:
    """
    Hybrid translation manager with smart language fallback:

    - Real-time: Always NLLB-600M (fast)
    - Final:
        - Hunyuan-MT-7B if target language is in HUNYUAN_SUPPORTED_LANGUAGES
        - NLLB-3.3B otherwise (fallback for unsupported languages)
    """

    def __init__(self,
                 nllb_realtime_model_path,
                 nllb_full_model_path,
                 hunyuan_model_path,
                 device,
                 hunyuan_load_in_8bit=False,
                 hunyuan_gpu_device=0):
        """
        Initialize hybrid translation manager.

        Args:
            nllb_realtime_model_path: Path to NLLB-600M for real-time translation
            nllb_full_model_path: Path to NLLB-3.3B for fallback final translation
            hunyuan_model_path: Path to Hunyuan-MT-7B for final translation (supported languages)
            device: 'cuda' or 'cpu'
            hunyuan_load_in_8bit: Load Hunyuan in 8-bit quantization
            hunyuan_gpu_device: GPU index for Hunyuan model
        """
        print("=" * 60)
        print("Initializing Hybrid Translation Manager (Smart Fallback)")
        print("=" * 60)
        self.device = device
        nllb_device = 0 if device == 'cuda' and torch.cuda.is_available() else -1

        # --- Initialize NLLB-600M for real-time translation ---
        print(f"\n[1/3] Loading NLLB-600M for real-time: {nllb_realtime_model_path}")
        self.nllb_realtime = pipeline(
            'translation',
            model=nllb_realtime_model_path,
            device=nllb_device
        )
        print("      NLLB-600M loaded")

        # --- Initialize NLLB-3.3B for fallback final translation ---
        print(f"\n[2/3] Loading NLLB-3.3B for fallback: {nllb_full_model_path}")
        self.nllb_full = pipeline(
            'translation',
            model=nllb_full_model_path,
            device=nllb_device
        )
        print("      NLLB-3.3B loaded")

        # --- Initialize Hunyuan for final translation (supported languages) ---
        print(f"\n[3/3] Loading Hunyuan-MT-7B for final: {hunyuan_model_path}")

        is_local_path = hunyuan_model_path.startswith('/') or hunyuan_model_path.startswith('./')

        self.hunyuan_tokenizer = AutoTokenizer.from_pretrained(
            hunyuan_model_path,
            trust_remote_code=True,
            local_files_only=is_local_path
        )

        # Determine dtype
        if device == 'cuda' and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch_dtype,
            'local_files_only': is_local_path,
        }

        if hunyuan_load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = {'': hunyuan_gpu_device}
        elif device == 'cuda':
            model_kwargs['device_map'] = {'': hunyuan_gpu_device}

        self.hunyuan_model = AutoModelForCausalLM.from_pretrained(
            hunyuan_model_path,
            **model_kwargs
        )

        # Hunyuan generation parameters
        self.hunyuan_gen_kwargs = {
            'top_k': 20,
            'top_p': 0.6,
            'repetition_penalty': 1.05,
            'temperature': 0.7,
            'max_new_tokens': 512,
            'do_sample': True,
            'pad_token_id': self.hunyuan_tokenizer.eos_token_id,
        }
        print("      Hunyuan-MT-7B loaded")

        print("\n" + "=" * 60)
        print("Hybrid Translation Manager initialized successfully!")
        print(f"  - Real-time: NLLB-600M (always)")
        print(f"  - Final (38 langs): Hunyuan-MT-7B")
        print(f"  - Final (fallback): NLLB-3.3B")
        print("=" * 60 + "\n")

    def _is_hunyuan_supported(self, lang_code):
        """Check if the language is supported by Hunyuan-MT-7B."""
        return lang_code in HUNYUAN_SUPPORTED_LANGUAGES

    def _get_hunyuan_language_name(self, lang_code):
        """Get the language name for Hunyuan prompts."""
        return HUNYUAN_SUPPORTED_LANGUAGES.get(lang_code, lang_code)

    def _is_chinese(self, lang_code):
        """Check if the language code represents Chinese."""
        return lang_code in CHINESE_CODES or 'chinese' in lang_code.lower()

    def _translate_nllb_realtime(self, text, src_lang, tgt_lang):
        """Translate using NLLB-600M (fast, for real-time)."""
        result = self.nllb_realtime(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        return result[0]['translation_text']

    def _translate_nllb_full(self, text, src_lang, tgt_lang):
        """Translate using NLLB-3.3B (fallback for unsupported languages)."""
        result = self.nllb_full(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1024)
        return result[0]['translation_text']

    def _translate_hunyuan(self, text, src_lang, tgt_lang):
        """Translate using Hunyuan-MT-7B (high quality, for supported languages)."""
        tgt_name = self._get_hunyuan_language_name(tgt_lang)

        if self._is_chinese(src_lang) or self._is_chinese(tgt_lang):
            prompt = f"把下面的文本翻译成{tgt_name}，不要额外解释。 {text}"
        else:
            prompt = f"Translate the following segment into {tgt_name}, without additional explanation. {text}"

        messages = [{"role": "user", "content": prompt}]
        tokenized = self.hunyuan_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_length = tokenized.shape[1]

        with torch.no_grad():
            outputs = self.hunyuan_model.generate(
                tokenized.to(self.hunyuan_model.device),
                **self.hunyuan_gen_kwargs
            )

        result = self.hunyuan_tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return result.strip()

    def translate(self, text, src_lang, tgt_lang, model_type='full'):
        """
        Translate text using the appropriate model.

        Args:
            text: Source text to translate
            src_lang: Source language code (NLLB format)
            tgt_lang: Target language code (NLLB format)
            model_type: 'realtime' for NLLB-600M, 'full' for Hunyuan/NLLB-3.3B

        Returns:
            Translated text string
        """
        if not text or not text.strip():
            return ""

        if model_type == 'realtime':
            # Real-time: Always use NLLB-600M (fast)
            return self._translate_nllb_realtime(text, src_lang, tgt_lang)
        else:
            # Final translation: Check if Hunyuan supports the target language
            if self._is_hunyuan_supported(tgt_lang):
                # Target language is supported by Hunyuan - use it for high quality
                return self._translate_hunyuan(text, src_lang, tgt_lang)
            else:
                # Target language not supported by Hunyuan - fallback to NLLB-3.3B
                return self._translate_nllb_full(text, src_lang, tgt_lang)
