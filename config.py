"""
config.py
=========
Single source of truth for all configuration constants.
spaCy removed — incompatible with Python 3.14.
"""

import os
from pathlib import Path


class Config:

    APP_NAME    = "ReadMyPDF"
    VERSION     = "2.0.0"
    DESCRIPTION = "Intelligent PDF to Audio Converter with header/footer removal"

    # Supported languages
    # spacy key removed — no longer used
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

    TTS_ENGINE:      str = os.getenv("TTS_ENGINE", "gtts").lower()
    EXTRACTION_MODE: str = os.getenv("EXTRACTION_MODE", "auto").lower()

    MIN_CHARS_FOR_TEXT_MODE: int = 100

    HEADER_ZONE:      float = 0.08
    FOOTER_ZONE:      float = 0.08
    REPEAT_THRESHOLD: int   = 3
    Y_BAND_PRECISION: int   = 2

    HEADING_SIZE_RATIO: float = 1.2
    MIN_HEADING_CHARS:  int   = 3
    MAX_HEADING_CHARS:  int   = 200

    CLEAN_CITATIONS:    bool = True
    CLEAN_FIGURE_REFS:  bool = True
    CLEAN_URLS:         bool = True
    CLEAN_PAGE_NUMBERS: bool = True
    FIX_HYPHENATION:    bool = True
    NORMALIZE_ABBREVS:  bool = True
    NORMALIZE_NUMBERS:  bool = True

    MAX_TTS_CHUNK_CHARS: int = 500
    CHUNK_SILENCE_MS:    int = 300

    DEFAULT_RATE:         float = 1.0
    DEFAULT_PITCH:        float = 1.0
    DEFAULT_AUDIO_FORMAT: str   = "mp3"
    AUDIO_READING_WPM:    int   = 150

    OPENAI_TTS_VOICE: str = "onyx"
    OPENAI_TTS_MODEL: str = "tts-1"

    MAX_PDF_SIZE_MB:    int = 50
    MAX_PDF_SIZE_BYTES: int = 50 * 1024 * 1024

    MIN_CHAPTER_CHARS: int = 200

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    BASE_DIR:   Path = Path(__file__).resolve().parent
    CACHE_DIR:  Path = BASE_DIR / ".cache"
    LOG_DIR:    Path = BASE_DIR / "logs"
    TEMP_DIR:   Path = BASE_DIR / "temp"
    OUTPUT_DIR: Path = BASE_DIR / "output"

    for _d in (CACHE_DIR, LOG_DIR, TEMP_DIR, OUTPUT_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    LOG_FILE:   Path = LOG_DIR / "app.log"
    LOG_FORMAT: str  = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL:  str  = os.getenv("LOG_LEVEL", "INFO").upper()

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
        """Stub — spaCy removed. Always returns None."""
        return None

    @classmethod
    def get_language_label(cls, lang_code: str) -> str:
        return cls.get_language_config(lang_code)["label"]

    @classmethod
    def get_all_language_labels(cls) -> dict[str, str]:
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
        """Always False — spaCy removed."""
        return False

    @classmethod
    def validate_extraction_mode(cls) -> str:
        return cls.EXTRACTION_MODE if cls.EXTRACTION_MODE in {"auto", "text", "ocr"} else "auto"