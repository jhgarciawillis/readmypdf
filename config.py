"""
config.py
=========
Single source of truth for all configuration constants.
No business logic. No imports beyond os and pathlib.
"""

import os
from pathlib import Path


class Config:

    APP_NAME    = "ReadMyPDF"
    VERSION     = "2.0.0"
    DESCRIPTION = "Intelligent PDF to Audio Converter with translation, language detection, and content verification"

    # ------------------------------------------------------------------ #
    # SUPPORTED LANGUAGES                                                  #
    # ------------------------------------------------------------------ #

    SUPPORTED_LANGUAGES = {
        "en":    {"gtts": "en",    "tesseract": "eng",     "label": "English"},
        "es":    {"gtts": "es",    "tesseract": "spa",     "label": "Spanish"},
        "fr":    {"gtts": "fr",    "tesseract": "fra",     "label": "French"},
        "de":    {"gtts": "de",    "tesseract": "deu",     "label": "German"},
        "it":    {"gtts": "it",    "tesseract": "ita",     "label": "Italian"},
        "ru":    {"gtts": "ru",    "tesseract": "rus",     "label": "Russian"},
        "zh-CN": {"gtts": "zh-CN", "tesseract": "chi_sim", "label": "Chinese (Simplified)"},
        "ja":    {"gtts": "ja",    "tesseract": "jpn",     "label": "Japanese"},
        "ar":    {"gtts": "ar",    "tesseract": "ara",     "label": "Arabic"},
        "hi":    {"gtts": "hi",    "tesseract": "hin",     "label": "Hindi"},
        "pt":    {"gtts": "pt",    "tesseract": "por",     "label": "Portuguese"},
        "nl":    {"gtts": "nl",    "tesseract": "nld",     "label": "Dutch"},
    }

    # ------------------------------------------------------------------ #
    # TTS ENGINE                                                           #
    # ------------------------------------------------------------------ #

    TTS_ENGINE:      str = os.getenv("TTS_ENGINE", "gtts").lower()
    EXTRACTION_MODE: str = os.getenv("EXTRACTION_MODE", "auto").lower()

    MIN_CHARS_FOR_TEXT_MODE: int = 100

    # ------------------------------------------------------------------ #
    # HEADER / FOOTER DETECTION                                            #
    # ------------------------------------------------------------------ #

    HEADER_ZONE:      float = 0.08
    FOOTER_ZONE:      float = 0.08
    REPEAT_THRESHOLD: int   = 3
    Y_BAND_PRECISION: int   = 2

    # ------------------------------------------------------------------ #
    # HEADING DETECTION                                                    #
    # ------------------------------------------------------------------ #

    HEADING_SIZE_RATIO: float = 1.2
    MIN_HEADING_CHARS:  int   = 3
    MAX_HEADING_CHARS:  int   = 200

    # ------------------------------------------------------------------ #
    # TEXT CLEANING                                                        #
    # ------------------------------------------------------------------ #

    CLEAN_CITATIONS:    bool = True
    CLEAN_FIGURE_REFS:  bool = True
    CLEAN_URLS:         bool = True
    CLEAN_PAGE_NUMBERS: bool = True
    FIX_HYPHENATION:    bool = True
    NORMALIZE_ABBREVS:  bool = True
    NORMALIZE_NUMBERS:  bool = True

    # ------------------------------------------------------------------ #
    # TTS CHUNKING                                                         #
    # ------------------------------------------------------------------ #

    MAX_TTS_CHUNK_CHARS: int = 500
    CHUNK_SILENCE_MS:    int = 300

    # ------------------------------------------------------------------ #
    # AUDIO DEFAULTS                                                       #
    # ------------------------------------------------------------------ #

    DEFAULT_RATE:         float = 1.0
    DEFAULT_PITCH:        float = 1.0
    DEFAULT_AUDIO_FORMAT: str   = "mp3"
    AUDIO_READING_WPM:    int   = 150

    OPENAI_TTS_VOICE: str = "onyx"
    OPENAI_TTS_MODEL: str = "tts-1"

    # ------------------------------------------------------------------ #
    # PDF CONSTRAINTS                                                      #
    # ------------------------------------------------------------------ #

    MAX_PDF_SIZE_MB:    int = 50
    MAX_PDF_SIZE_BYTES: int = 50 * 1024 * 1024
    MIN_CHAPTER_CHARS:  int = 200

    # ------------------------------------------------------------------ #
    # PAGE / CHAPTER COMPLETENESS VERIFICATION                             #
    #                                                                      #
    # LAST_WORDS_FINGERPRINT_COUNT                                         #
    #   Number of words taken from the last content block on a page and    #
    #   used as a fingerprint. If these words appear in the assembled      #
    #   chapter text, the page was fully captured. 5 words is enough to    #
    #   be unique while staying robust against minor OCR variation.        #
    #                                                                      #
    # FINGERPRINT_MIN_WORDS                                                #
    #   Minimum words a fingerprint must contain to be considered          #
    #   reliable. Protects against false positives when a last line        #
    #   is a single word, a number, or a symbol.                           #
    #                                                                      #
    # AUTO_EXTEND_PAGE_RANGE                                               #
    #   When True: if the user's selected page range cuts through a        #
    #   chapter mid-way, silently extend end_page to the natural end of    #
    #   that chapter. A banner informs the user what happened.             #
    #   When False: show a choice dialog before processing.                #
    # ------------------------------------------------------------------ #

    LAST_WORDS_FINGERPRINT_COUNT: int  = 5
    FINGERPRINT_MIN_WORDS:        int  = 3
    AUTO_EXTEND_PAGE_RANGE:       bool = True

    # ------------------------------------------------------------------ #
    # TRANSLATION                                                          #
    # ------------------------------------------------------------------ #

    TRANSLATION_ENABLED: bool = os.getenv("TRANSLATION_ENABLED", "true").lower() == "true"
    AUTO_DETECT_LANGUAGE: bool = os.getenv("AUTO_DETECT_LANGUAGE", "true").lower() == "true"

    TRANSLATE_ENGINE_PRIMARY:  str = os.getenv("TRANSLATE_ENGINE_PRIMARY",  "google").lower()
    TRANSLATE_ENGINE_FALLBACK: str = os.getenv("TRANSLATE_ENGINE_FALLBACK", "libretranslate").lower()

    LIBRETRANSLATE_URL:     str = os.getenv("LIBRETRANSLATE_URL",     "https://libretranslate.com")
    LIBRETRANSLATE_API_KEY: str = os.getenv("LIBRETRANSLATE_API_KEY", "")

    TRANSLATE_MAX_CHUNK_CHARS: int = 4500

    TRANSLATION_SUPPORTED_LANGUAGES: dict = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh-CN": "Chinese (Simplified)",
        "ja": "Japanese",
        "ar": "Arabic",
    }

    # ------------------------------------------------------------------ #
    # API KEYS                                                             #
    # ------------------------------------------------------------------ #

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ------------------------------------------------------------------ #
    # FILE PATHS                                                           #
    # ------------------------------------------------------------------ #

    BASE_DIR:   Path = Path(__file__).resolve().parent
    CACHE_DIR:  Path = BASE_DIR / ".cache"
    LOG_DIR:    Path = BASE_DIR / "logs"
    TEMP_DIR:   Path = BASE_DIR / "temp"
    OUTPUT_DIR: Path = BASE_DIR / "output"

    for _d in (CACHE_DIR, LOG_DIR, TEMP_DIR, OUTPUT_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # LOGGING                                                              #
    # ------------------------------------------------------------------ #

    LOG_FILE:   Path = LOG_DIR / "app.log"
    LOG_FORMAT: str  = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL:  str  = os.getenv("LOG_LEVEL", "INFO").upper()

    # ================================================================== #
    # LANGUAGE HELPERS                                                     #
    # ================================================================== #

    @classmethod
    def get_language_config(cls, lang_code: str) -> dict:
        if lang_code not in cls.SUPPORTED_LANGUAGES:
            raise KeyError(f"Unsupported language code: '{lang_code}'.")
        return cls.SUPPORTED_LANGUAGES[lang_code]

    @classmethod
    def get_gtts_code(cls, lang_code: str) -> str:
        return cls.get_language_config(lang_code)["gtts"]

    @classmethod
    def get_tesseract_code(cls, lang_code: str) -> str:
        return cls.get_language_config(lang_code)["tesseract"]

    @classmethod
    def get_spacy_model(cls, lang_code: str):
        return None

    @classmethod
    def get_language_label(cls, lang_code: str) -> str:
        return cls.get_language_config(lang_code)["label"]

    @classmethod
    def get_all_language_labels(cls) -> dict:
        return {code: cfg["label"] for code, cfg in cls.SUPPORTED_LANGUAGES.items()}

    @classmethod
    def get_lang_code_from_label(cls, label: str) -> str:
        for code, cfg in cls.SUPPORTED_LANGUAGES.items():
            if cfg["label"].lower() == label.lower():
                return code
        raise ValueError(f"No language found with label '{label}'.")

    @classmethod
    def is_openai_available(cls) -> bool:
        key = cls.OPENAI_API_KEY
        return bool(key) and key.startswith("sk-") and len(key) > 20

    @classmethod
    def get_active_tts_engine(cls) -> str:
        if cls.TTS_ENGINE == "openai" and cls.is_openai_available():
            return "openai"
        return "gtts"

    @classmethod
    def has_spacy_support(cls, lang_code: str) -> bool:
        return False

    @classmethod
    def validate_extraction_mode(cls) -> str:
        return cls.EXTRACTION_MODE if cls.EXTRACTION_MODE in {"auto", "text", "ocr"} else "auto"

    # ================================================================== #
    # TRANSLATION HELPERS                                                  #
    # ================================================================== #

    @classmethod
    def is_translation_available(cls) -> bool:
        if not cls.TRANSLATION_ENABLED:
            return False
        try:
            import googletrans  # noqa
            return True
        except ImportError:
            pass
        if cls.LIBRETRANSLATE_URL:
            return True
        return False

    @classmethod
    def is_google_translate_available(cls) -> bool:
        try:
            import googletrans  # noqa
            return True
        except ImportError:
            return False

    @classmethod
    def is_libretranslate_available(cls) -> bool:
        return bool(cls.LIBRETRANSLATE_URL)

    @classmethod
    def get_translation_config(cls) -> dict:
        return {
            "primary_engine":           cls.TRANSLATE_ENGINE_PRIMARY,
            "fallback_engine":          cls.TRANSLATE_ENGINE_FALLBACK,
            "libretranslate_url":       cls.LIBRETRANSLATE_URL,
            "libretranslate_api_key":   cls.LIBRETRANSLATE_API_KEY,
            "max_chunk_chars":          cls.TRANSLATE_MAX_CHUNK_CHARS,
            "google_available":         cls.is_google_translate_available(),
            "libretranslate_available": cls.is_libretranslate_available(),
        }

    @classmethod
    def map_detected_lang_to_supported(cls, detected_code: str) -> str:
        if detected_code in cls.SUPPORTED_LANGUAGES:
            return detected_code
        for code in cls.SUPPORTED_LANGUAGES:
            if code.lower() == detected_code.lower():
                return code
        if detected_code in ("zh-cn", "zh_CN", "zh"):
            return "zh-CN"
        prefix = detected_code.split("-")[0].split("_")[0]
        if prefix in cls.SUPPORTED_LANGUAGES:
            return prefix
        return "en"