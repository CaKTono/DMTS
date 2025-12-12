"""
Hunyuan-MT-7B Translation Manager

This module provides LLM-based translation using Tencent's Hunyuan-MT-7B model.
It replaces the NLLB-based translation with a more powerful 7B parameter LLM.

Model: https://huggingface.co/tencent/Hunyuan-MT-7B

Requirements:
- transformers >= 4.56.0
- ~14GB VRAM (float16) or ~7GB (int8)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Import centralized language mappings
from language_codes import NLLB_TO_LANGUAGE_NAME, CHINESE_CODES, get_language_name, is_chinese


class HunyuanTranslationManager:
    """
    Translation manager using Tencent's Hunyuan-MT-7B model.

    This uses a single 7B LLM for both real-time and full translations,
    as loading two 7B models would require excessive VRAM (~28GB+).

    The model uses chat templates with specific prompts:
    - For Chinese <-> X: Uses Chinese prompt
    - For X <-> Y (non-Chinese): Uses English prompt
    """

    def __init__(self, model_path, device, torch_dtype=None, load_in_8bit=False, gpu_device_index=0):
        """
        Initialize Hunyuan-MT-7B translation model.

        Args:
            model_path: Path to local model or 'tencent/Hunyuan-MT-7B'
            device: 'cuda' or 'cpu'
            torch_dtype: torch.float16, torch.bfloat16, or None (auto)
            load_in_8bit: If True, load model in 8-bit quantization (saves VRAM)
            gpu_device_index: GPU index to use (default: 0)
        """
        print(f"Loading Hunyuan-MT-7B translation model from: {model_path}")

        self.device = device

        # Determine dtype
        if torch_dtype is None:
            if device == 'cuda' and torch.cuda.is_available():
                # Use bfloat16 if available, otherwise float16
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        print(f"Loading tokenizer...")

        # Detect if using local path or HuggingFace model ID
        is_local_path = model_path.startswith('/') or model_path.startswith('./')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local_path
        )

        print(f"Loading model with dtype={torch_dtype}, load_in_8bit={load_in_8bit}...")

        # Model loading configuration
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch_dtype,
            'local_files_only': is_local_path,
        }

        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = {'': gpu_device_index}
        elif device == 'cuda':
            model_kwargs['device_map'] = {'': gpu_device_index}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Move to CPU if explicitly requested and not using quantization
        if device == 'cpu' and not load_in_8bit:
            self.model = self.model.to('cpu')

        # Recommended generation parameters from HuggingFace
        self.gen_kwargs = {
            'top_k': 20,
            'top_p': 0.6,
            'repetition_penalty': 1.05,
            'temperature': 0.7,
            'max_new_tokens': 512,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
        }

        print(f"Hunyuan-MT-7B translation model loaded successfully.")

    def _get_language_name(self, lang_code):
        """
        Convert NLLB language code to human-readable language name.
        Falls back to the code itself if not found in mapping.
        """
        # If already a human-readable name (contains letters only), return as-is
        if lang_code.replace(' ', '').isalpha() and '_' not in lang_code:
            return lang_code

        return NLLB_TO_LANGUAGE_NAME.get(lang_code, lang_code)

    def _is_chinese(self, lang_code):
        """Check if the language code represents Chinese."""
        return lang_code in CHINESE_CODES or 'chinese' in lang_code.lower()

    def _build_prompt(self, text, src_lang, tgt_lang):
        """
        Build the translation prompt based on Hunyuan-MT documentation.

        For ZH <-> X: Uses Chinese prompt
        For X <-> Y (non-Chinese): Uses English prompt
        """
        tgt_name = self._get_language_name(tgt_lang)

        if self._is_chinese(src_lang) or self._is_chinese(tgt_lang):
            # Chinese prompt template
            return f"把下面的文本翻译成{tgt_name}，不要额外解释。 {text}"
        else:
            # English prompt template (for non-Chinese pairs)
            return f"Translate the following segment into {tgt_name}, without additional explanation. {text}"

    def translate(self, text, src_lang, tgt_lang, model_type='full'):
        """
        Translate text using Hunyuan-MT-7B.

        Args:
            text: Source text to translate
            src_lang: Source language code (NLLB format like 'eng_Latn' or name like 'English')
            tgt_lang: Target language code (NLLB format like 'zho_Hans' or name like 'Chinese')
            model_type: Ignored (single model architecture). Kept for API compatibility.

        Returns:
            Translated text string
        """
        if not text or not text.strip():
            return ""

        # Build the prompt
        prompt = self._build_prompt(text.strip(), src_lang, tgt_lang)

        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Generate translation
        input_length = tokenized.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                tokenized.to(self.model.device),
                **self.gen_kwargs
            )

        # Decode only the new tokens (skip input)
        result = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return result.strip()


# Factory function for easy instantiation
def create_translation_manager(model_path, device, load_in_8bit=False):
    """
    Factory function to create a HunyuanTranslationManager.

    Args:
        model_path: Path to model or 'tencent/Hunyuan-MT-7B'
        device: 'cuda' or 'cpu'
        load_in_8bit: Use 8-bit quantization to save VRAM

    Returns:
        HunyuanTranslationManager instance
    """
    return HunyuanTranslationManager(
        model_path=model_path,
        device=device,
        load_in_8bit=load_in_8bit
    )
