"""
Language Code Conversion Module for DMTS MK2

Provides centralized mappings for:
- ISO 639-1/2 codes → NLLB codes (client convenience)
- NLLB codes → Human-readable language names (for Hunyuan prompts)
- Hunyuan-supported language set (for fallback logic)

Usage:
    from language_codes import normalize_language_code, get_language_name, is_hunyuan_supported
    
    # Convert ISO to NLLB
    nllb_code = normalize_language_code("zh")  # → "zho_Hans"
    
    # Get language name for Hunyuan prompts
    name = get_language_name("zho_Hans")  # → "Chinese"
    
    # Check Hunyuan support
    supported = is_hunyuan_supported("zho_Hans")  # → True
"""

# =============================================================================
# ISO 639-1/2 → NLLB Code Mapping
# =============================================================================
# Source: language_support.txt
# Allows clients to send simplified codes like "zh" instead of "zho_Hans"

ISO_TO_NLLB = {
    # ----- Hunyuan-MT-7B Supported Languages (38) -----
    "zh": "zho_Hans",
    "zh-Hans": "zho_Hans",
    "zh-Hant": "zho_Hant",
    "zh-TW": "zho_Hant",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "ara_Arab",
    "hi": "hin_Deva",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "msa_Latn",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "cs": "ces_Latn",
    "uk": "ukr_Cyrl",
    "tl": "tgl_Latn",
    "fil": "tgl_Latn",  # Filipino alias
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "ur": "urd_Arab",
    "fa": "pes_Arab",
    "he": "heb_Hebr",
    "km": "khm_Khmr",
    "my": "mya_Mymr",
    "bo": "bod_Tibt",
    "kk": "kaz_Cyrl",
    "mn": "mon_Cyrl",
    "ug": "uig_Arab",
    "yue": "yue_Hant",
    
    # ----- NLLB Extended Languages (162 additional) -----
    "af": "afr_Latn",       # Afrikaans
    "am": "amh_Ethi",       # Amharic
    "as": "asm_Beng",       # Assamese
    "az": "azj_Latn",       # Azerbaijani (North)
    "be": "bel_Cyrl",       # Belarusian
    "bg": "bul_Cyrl",       # Bulgarian
    "bm": "bam_Latn",       # Bambara
    "bs": "bos_Latn",       # Bosnian
    "ca": "cat_Latn",       # Catalan
    "ceb": "ceb_Latn",      # Cebuano
    "cy": "cym_Latn",       # Welsh
    "da": "dan_Latn",       # Danish
    "el": "ell_Grek",       # Greek
    "eo": "epo_Latn",       # Esperanto
    "et": "est_Latn",       # Estonian
    "eu": "eus_Latn",       # Basque
    "fi": "fin_Latn",       # Finnish
    "fo": "fao_Latn",       # Faroese
    "ga": "gle_Latn",       # Irish
    "gd": "gla_Latn",       # Scottish Gaelic
    "gl": "glg_Latn",       # Galician
    "gn": "grn_Latn",       # Guarani
    "ha": "hau_Latn",       # Hausa
    "hr": "hrv_Latn",       # Croatian
    "ht": "hat_Latn",       # Haitian Creole
    "hu": "hun_Latn",       # Hungarian
    "hy": "hye_Armn",       # Armenian
    "ig": "ibo_Latn",       # Igbo
    "is": "isl_Latn",       # Icelandic
    "jv": "jav_Latn",       # Javanese
    "ka": "kat_Geor",       # Georgian
    "kn": "kan_Knda",       # Kannada
    "ky": "kir_Cyrl",       # Kyrgyz
    "la": "lat_Latn",       # Latin (mapped to closest)
    "lb": "ltz_Latn",       # Luxembourgish
    "lg": "lug_Latn",       # Ganda
    "ln": "lin_Latn",       # Lingala
    "lo": "lao_Laoo",       # Lao
    "lt": "lit_Latn",       # Lithuanian
    "lv": "lvs_Latn",       # Latvian
    "mg": "plt_Latn",       # Malagasy (Plateau)
    "mi": "mri_Latn",       # Maori
    "mk": "mkd_Cyrl",       # Macedonian
    "ml": "mal_Mlym",       # Malayalam
    "mt": "mlt_Latn",       # Maltese
    "ne": "npi_Deva",       # Nepali
    "no": "nob_Latn",       # Norwegian Bokmål
    "nb": "nob_Latn",       # Norwegian Bokmål
    "nn": "nno_Latn",       # Norwegian Nynorsk
    "ny": "nya_Latn",       # Nyanja/Chichewa
    "om": "gaz_Latn",       # Oromo
    "or": "ory_Orya",       # Odia
    "pa": "pan_Guru",       # Punjabi
    "ps": "pbt_Arab",       # Pashto
    "ro": "ron_Latn",       # Romanian
    "rw": "kin_Latn",       # Kinyarwanda
    "sd": "snd_Arab",       # Sindhi
    "si": "sin_Sinh",       # Sinhala
    "sk": "slk_Latn",       # Slovak
    "sl": "slv_Latn",       # Slovenian
    "sm": "smo_Latn",       # Samoan
    "sn": "sna_Latn",       # Shona
    "so": "som_Latn",       # Somali
    "sq": "als_Latn",       # Albanian (Tosk)
    "sr": "srp_Cyrl",       # Serbian
    "st": "sot_Latn",       # Southern Sotho
    "su": "sun_Latn",       # Sundanese
    "sv": "swe_Latn",       # Swedish
    "sw": "swh_Latn",       # Swahili
    "tg": "tgk_Cyrl",       # Tajik
    "ti": "tir_Ethi",       # Tigrinya
    "tk": "tuk_Latn",       # Turkmen
    "tn": "tsn_Latn",       # Tswana
    "to": "ton_Latn",       # Tongan (mapped to closest)
    "ts": "tso_Latn",       # Tsonga
    "tt": "tat_Cyrl",       # Tatar
    "tw": "twi_Latn",       # Twi
    "uz": "uzn_Latn",       # Uzbek (Northern)
    "wo": "wol_Latn",       # Wolof
    "xh": "xho_Latn",       # Xhosa
    "yi": "ydd_Hebr",       # Yiddish
    "yo": "yor_Latn",       # Yoruba
    "zu": "zul_Latn",       # Zulu
    
    # Common regional/script variants
    "zh-CN": "zho_Hans",
    "pt-BR": "por_Latn",
    "pt-PT": "por_Latn",
    "en-US": "eng_Latn",
    "en-GB": "eng_Latn",
    "es-ES": "spa_Latn",
    "es-MX": "spa_Latn",
    "fr-FR": "fra_Latn",
    "fr-CA": "fra_Latn",
    "de-DE": "deu_Latn",
    "ar-SA": "ara_Arab",
    "ja-JP": "jpn_Jpan",
    "ko-KR": "kor_Hang",
}


# =============================================================================
# NLLB Code → Human-Readable Language Name Mapping
# =============================================================================
# Used by Hunyuan translation prompts

NLLB_TO_LANGUAGE_NAME = {
    # ----- Hunyuan-MT-7B Supported Languages (38) -----
    'zho_Hans': 'Chinese',
    'zho_Hant': 'Traditional Chinese',
    'eng_Latn': 'English',
    'fra_Latn': 'French',
    'spa_Latn': 'Spanish',
    'por_Latn': 'Portuguese',
    'deu_Latn': 'German',
    'ita_Latn': 'Italian',
    'rus_Cyrl': 'Russian',
    'jpn_Jpan': 'Japanese',
    'kor_Hang': 'Korean',
    'ara_Arab': 'Arabic',
    'hin_Deva': 'Hindi',
    'vie_Latn': 'Vietnamese',
    'tha_Thai': 'Thai',
    'ind_Latn': 'Indonesian',
    'msa_Latn': 'Malay',
    'tur_Latn': 'Turkish',
    'pol_Latn': 'Polish',
    'nld_Latn': 'Dutch',
    'ces_Latn': 'Czech',
    'ukr_Cyrl': 'Ukrainian',
    'tgl_Latn': 'Filipino',
    'ben_Beng': 'Bengali',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'mar_Deva': 'Marathi',
    'guj_Gujr': 'Gujarati',
    'urd_Arab': 'Urdu',
    'pes_Arab': 'Persian',
    'fas_Arab': 'Persian',  # Alternative code
    'heb_Hebr': 'Hebrew',
    'khm_Khmr': 'Khmer',
    'mya_Mymr': 'Burmese',
    'bod_Tibt': 'Tibetan',
    'kaz_Cyrl': 'Kazakh',
    'mon_Cyrl': 'Mongolian',
    'uig_Arab': 'Uyghur',
    'yue_Hant': 'Cantonese',
    
    # ----- Extended Languages for NLLB -----
    'afr_Latn': 'Afrikaans',
    'amh_Ethi': 'Amharic',
    'asm_Beng': 'Assamese',
    'azj_Latn': 'Azerbaijani',
    'bel_Cyrl': 'Belarusian',
    'bul_Cyrl': 'Bulgarian',
    'bos_Latn': 'Bosnian',
    'cat_Latn': 'Catalan',
    'ceb_Latn': 'Cebuano',
    'cym_Latn': 'Welsh',
    'dan_Latn': 'Danish',
    'ell_Grek': 'Greek',
    'est_Latn': 'Estonian',
    'eus_Latn': 'Basque',
    'fin_Latn': 'Finnish',
    'gle_Latn': 'Irish',
    'gla_Latn': 'Scottish Gaelic',
    'glg_Latn': 'Galician',
    'hau_Latn': 'Hausa',
    'hrv_Latn': 'Croatian',
    'hat_Latn': 'Haitian Creole',
    'hun_Latn': 'Hungarian',
    'hye_Armn': 'Armenian',
    'ibo_Latn': 'Igbo',
    'isl_Latn': 'Icelandic',
    'jav_Latn': 'Javanese',
    'kat_Geor': 'Georgian',
    'kan_Knda': 'Kannada',
    'kir_Cyrl': 'Kyrgyz',
    'ltz_Latn': 'Luxembourgish',
    'lao_Laoo': 'Lao',
    'lit_Latn': 'Lithuanian',
    'lvs_Latn': 'Latvian',
    'mkd_Cyrl': 'Macedonian',
    'mal_Mlym': 'Malayalam',
    'mlt_Latn': 'Maltese',
    'npi_Deva': 'Nepali',
    'nob_Latn': 'Norwegian',
    'nya_Latn': 'Nyanja',
    'ory_Orya': 'Odia',
    'pan_Guru': 'Punjabi',
    'pbt_Arab': 'Pashto',
    'ron_Latn': 'Romanian',
    'snd_Arab': 'Sindhi',
    'sin_Sinh': 'Sinhala',
    'slk_Latn': 'Slovak',
    'slv_Latn': 'Slovenian',
    'sna_Latn': 'Shona',
    'som_Latn': 'Somali',
    'als_Latn': 'Albanian',
    'srp_Cyrl': 'Serbian',
    'sun_Latn': 'Sundanese',
    'swe_Latn': 'Swedish',
    'swh_Latn': 'Swahili',
    'tgk_Cyrl': 'Tajik',
    'tir_Ethi': 'Tigrinya',
    'uzn_Latn': 'Uzbek',
    'xho_Latn': 'Xhosa',
    'yor_Latn': 'Yoruba',
    'zul_Latn': 'Zulu',
}


# =============================================================================
# Hunyuan-MT-7B Supported Languages Set
# =============================================================================
# 38 languages supported by the high-quality Hunyuan model

HUNYUAN_SUPPORTED_LANGUAGES = {
    'zho_Hans': 'Chinese',
    'zho_Hant': 'Traditional Chinese',
    'eng_Latn': 'English',
    'fra_Latn': 'French',
    'spa_Latn': 'Spanish',
    'por_Latn': 'Portuguese',
    'deu_Latn': 'German',
    'ita_Latn': 'Italian',
    'rus_Cyrl': 'Russian',
    'jpn_Jpan': 'Japanese',
    'kor_Hang': 'Korean',
    'ara_Arab': 'Arabic',
    'hin_Deva': 'Hindi',
    'vie_Latn': 'Vietnamese',
    'tha_Thai': 'Thai',
    'ind_Latn': 'Indonesian',
    'msa_Latn': 'Malay',
    'tur_Latn': 'Turkish',
    'pol_Latn': 'Polish',
    'nld_Latn': 'Dutch',
    'ces_Latn': 'Czech',
    'ukr_Cyrl': 'Ukrainian',
    'tgl_Latn': 'Filipino',
    'ben_Beng': 'Bengali',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'mar_Deva': 'Marathi',
    'guj_Gujr': 'Gujarati',
    'urd_Arab': 'Urdu',
    'pes_Arab': 'Persian',
    'fas_Arab': 'Persian',
    'heb_Hebr': 'Hebrew',
    'khm_Khmr': 'Khmer',
    'mya_Mymr': 'Burmese',
    'bod_Tibt': 'Tibetan',
    'kaz_Cyrl': 'Kazakh',
    'mon_Cyrl': 'Mongolian',
    'uig_Arab': 'Uyghur',
    'yue_Hant': 'Cantonese',
}

# Chinese language codes for prompt selection
CHINESE_CODES = {'zho_Hans', 'zho_Hant', 'cmn_Hans', 'cmn_Hant', 'yue_Hant'}


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_language_code(code: str) -> str:
    """
    Normalize any language code input to NLLB format.
    
    Accepts:
        - ISO 639-1 codes: "zh", "en", "ja"
        - ISO 639-1 with region: "zh-CN", "en-US"
        - NLLB codes: "zho_Hans", "eng_Latn" (passthrough)
    
    Returns:
        NLLB format code (e.g., "zho_Hans")
        
    Fallback Behavior:
        - If code is not found in ISO_TO_NLLB mapping:
          1. If it looks like NLLB format (xxx_Xxxx), return as-is
          2. Otherwise, return as-is (let downstream NLLB model handle it)
        
    Example:
        normalize_language_code("zh") → "zho_Hans"
        normalize_language_code("zho_Hans") → "zho_Hans"
        normalize_language_code("ace_Arab") → "ace_Arab" (passthrough, valid NLLB)
        normalize_language_code("unknown") → "unknown" (passthrough)
    """
    if not code:
        return code
    
    # Clean the input
    code = code.strip()
    
    # If already NLLB format (contains underscore), return as-is
    # This handles both known and unknown NLLB codes like "ace_Arab"
    if '_' in code:
        return code
    
    # Try direct lookup in ISO_TO_NLLB mapping
    if code in ISO_TO_NLLB:
        return ISO_TO_NLLB[code]
    
    # Try case-insensitive lookup
    code_lower = code.lower()
    if code_lower in ISO_TO_NLLB:
        return ISO_TO_NLLB[code_lower]
    
    # Unknown code - return as-is
    # This allows clients to send raw NLLB codes not in ISO mapping
    # Example: "ace_Arab" (Acehnese) has no ISO 639-1 code
    return code


def get_language_name(nllb_code: str) -> str:
    """
    Get human-readable language name from NLLB code.
    
    Used for Hunyuan translation prompts.
    Falls back to the code itself if not found.
    
    Example:
        get_language_name("zho_Hans") → "Chinese"
        get_language_name("unknown") → "unknown"
    """
    if not nllb_code:
        return nllb_code
    
    # If already a plain language name, return as-is
    if nllb_code.replace(' ', '').isalpha() and '_' not in nllb_code:
        return nllb_code
    
    return NLLB_TO_LANGUAGE_NAME.get(nllb_code, nllb_code)


def is_hunyuan_supported(nllb_code: str) -> bool:
    """
    Check if a language is supported by Hunyuan-MT-7B.
    
    Example:
        is_hunyuan_supported("zho_Hans") → True
        is_hunyuan_supported("ace_Arab") → False
    """
    return nllb_code in HUNYUAN_SUPPORTED_LANGUAGES


def is_chinese(lang_code: str) -> bool:
    """
    Check if a language code represents Chinese (for prompt selection).
    
    Example:
        is_chinese("zho_Hans") → True
        is_chinese("eng_Latn") → False
    """
    return lang_code in CHINESE_CODES or 'chinese' in lang_code.lower()


# =============================================================================
# Quick Self-Test
# =============================================================================

if __name__ == "__main__":
    print("Testing language_codes.py...")
    
    # Test ISO → NLLB
    assert normalize_language_code("zh") == "zho_Hans", "zh → zho_Hans"
    assert normalize_language_code("en") == "eng_Latn", "en → eng_Latn"
    assert normalize_language_code("ja") == "jpn_Jpan", "ja → jpn_Jpan"
    assert normalize_language_code("zh-Hant") == "zho_Hant", "zh-Hant → zho_Hant"
    
    # Test NLLB passthrough
    assert normalize_language_code("zho_Hans") == "zho_Hans", "NLLB passthrough"
    assert normalize_language_code("eng_Latn") == "eng_Latn", "NLLB passthrough"
    
    # Test unknown passthrough
    assert normalize_language_code("xyz") == "xyz", "Unknown passthrough"
    
    # Test language name lookup
    assert get_language_name("zho_Hans") == "Chinese", "Name lookup"
    assert get_language_name("eng_Latn") == "English", "Name lookup"
    assert get_language_name("jpn_Jpan") == "Japanese", "Name lookup"
    
    # Test Hunyuan support
    assert is_hunyuan_supported("zho_Hans") == True, "Hunyuan support"
    assert is_hunyuan_supported("eng_Latn") == True, "Hunyuan support"
    assert is_hunyuan_supported("afr_Latn") == False, "Hunyuan not supported"
    
    # Test Chinese detection
    assert is_chinese("zho_Hans") == True, "Chinese detection"
    assert is_chinese("yue_Hant") == True, "Cantonese detection"
    assert is_chinese("eng_Latn") == False, "Non-Chinese"
    
    print(f"✅ All tests passed!")
    print(f"   ISO_TO_NLLB: {len(ISO_TO_NLLB)} mappings")
    print(f"   NLLB_TO_LANGUAGE_NAME: {len(NLLB_TO_LANGUAGE_NAME)} mappings")
    print(f"   HUNYUAN_SUPPORTED_LANGUAGES: {len(HUNYUAN_SUPPORTED_LANGUAGES)} languages")
