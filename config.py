"""
config.py
=========
Single source of truth for all configuration constants and settings.
No business logic. No imports beyond os and pathlib.

All tuneable parameters live here. No other file should hardcode
thresholds, paths, language codes, or feature flags.
"""

import os
from pathlib import Path


class Config:

    # ------------------------------------------------------------------ #
    # APPLICATION                                                          #
    # ------------------------------------------------------------------ #

    APP_NAME    = "ReadMyPDF"
    VERSION     = "2.0.0"
    DESCRIPTION = "Intelligent PDF to Audio Converter with header/footer removal"

    # ------------------------------------------------------------------ #
    # SUPPORTED LANGUAGES                                                  #
    #                                                                      #
    # Each entry is a dict with three keys:                                #
    #   gtts       — language code for gTTS / OpenAI TTS                  #
    #   tesseract  — language code for pytesseract / Tesseract OCR         #
    #   spacy      — spaCy model package name                              #
    #   label      — human-readable name shown in the UI                  #
    #                                                                      #
    # spaCy models must be downloaded separately after pip install:        #
    #   python -m spacy download en_core_web_sm                            #
    #   python -m spacy download es_core_news_sm                           #
    #   python -m spacy download fr_core_news_sm                           #
    #                                                                      #
    # For languages without a spaCy model (de, it, ru, zh-CN, ja, ar,     #
    # hi, pt, nl), NLP analysis features are disabled gracefully —         #
    # PDF extraction and TTS still work normally.                          #
    # ------------------------------------------------------------------ #

    SUPPORTED_LANGUAGES = {
        "en": {
            "gtts":      "en",
            "tesseract": "eng",
            "spacy":     "en_core_web_sm",
            "label":     "English",
        },
        "es": {
            "gtts":      "es",
            "tesseract": "spa",
            "spacy":     "es_core_news_sm",
            "label":     "Spanish",
        },
        "fr": {
            "gtts":      "fr",
            "tesseract": "fra",
            "spacy":     "fr_core_news_sm",
            "label":     "French",
        },
        "de": {
            "gtts":      "de",
            "tesseract": "deu",
            "spacy":     None,            # no small model bundled by default
            "label":     "German",
        },
        "it": {
            "gtts":      "it",
            "tesseract": "ita",
            "spacy":     None,
            "label":     "Italian",
        },
        "ru": {
            "gtts":      "ru",
            "tesseract": "rus",
            "spacy":     None,
            "label":     "Russian",
        },
        "zh-CN": {
            "gtts":      "zh-CN",
            "tesseract": "chi_sim",
            "spacy":     None,
            "label":     "Chinese (Simplified)",
        },
        "ja": {
            "gtts":      "ja",
            "tesseract": "jpn",
            "spacy":     None,
            "label":     "Japanese",
        },
        "ar": {
            "gtts":      "ar",
            "tesseract": "ara",
            "spacy":     None,
            "label":     "Arabic",
        },
        "hi": {
            "gtts":      "hi",
            "tesseract": "hin",
            "spacy":     None,
            "label":     "Hindi",
        },
        "pt": {
            "gtts":      "pt",
            "tesseract": "por",
            "spacy":     None,
            "label":     "Portuguese",
        },
        "nl": {
            "gtts":      "nl",
            "tesseract": "nld",
            "spacy":     None,
            "label":     "Dutch",
        },
    }

    # ------------------------------------------------------------------ #
    # TTS ENGINE                                                           #
    #                                                                      #
    # "gtts"   — free, uses Google Translate endpoint (no key needed).    #
    #            Rate-limited under heavy use. Robotic voice quality.      #
    # "openai" — premium, requires OPENAI_API_KEY env var.                #
    #            Uses tts-1 model, "onyx" voice. ~$0.015/1k chars.        #
    #            Sounds significantly more natural.                        #
    #                                                                      #
    # Override at runtime by setting env var:                              #
    #   export TTS_ENGINE=openai                                           #
    # ------------------------------------------------------------------ #

    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "gtts").lower()

    # ------------------------------------------------------------------ #
    # PDF EXTRACTION MODE                                                  #
    #                                                                      #
    # "auto"    — try native text extraction first; fall back to OCR      #
    #             if fewer than MIN_CHARS_FOR_TEXT_MODE chars extracted.  #
    # "text"    — force native PyMuPDF text extraction (fast, accurate).  #
    #             Will silently return empty strings on scanned PDFs.      #
    # "ocr"     — force Tesseract OCR on every page (slow, for scans).   #
    #                                                                      #
    # Override at runtime:                                                 #
    #   export EXTRACTION_MODE=text                                        #
    # ------------------------------------------------------------------ #

    EXTRACTION_MODE: str = os.getenv("EXTRACTION_MODE", "auto").lower()

    # Minimum average characters per page to consider a PDF "text-based"
    # in auto mode. PDFs below this fall back to OCR.
    MIN_CHARS_FOR_TEXT_MODE: int = 100

    # ------------------------------------------------------------------ #
    # HEADER / FOOTER DETECTION                                            #
    #                                                                      #
    # HEADER_ZONE  — top fraction of page height considered a header zone #
    # FOOTER_ZONE  — bottom fraction of page height considered footer zone#
    # REPEAT_THRESHOLD — a text block must appear on this many pages at   #
    #                    the same Y-band to be flagged as header/footer.   #
    #                    Lower = more aggressive removal.                  #
    # Y_BAND_PRECISION — rounding precision for Y-coordinate grouping.    #
    #                    2 decimal places = groups blocks within ~1% of    #
    #                    page height together.                             #
    # ------------------------------------------------------------------ #

    HEADER_ZONE:      float = 0.08   # top 8% of page
    FOOTER_ZONE:      float = 0.08   # bottom 8% of page
    REPEAT_THRESHOLD: int   = 3      # appears on 3+ pages → flagged
    Y_BAND_PRECISION: int   = 2      # decimal places for Y grouping

    # ------------------------------------------------------------------ #
    # HEADING DETECTION                                                    #
    #                                                                      #
    # HEADING_SIZE_RATIO — a text span is considered a heading if its     #
    #                      font size is >= median body font size * ratio.  #
    #                      1.2 = 20% larger than body text.               #
    # MIN_HEADING_CHARS  — headings shorter than this are ignored         #
    #                      (catches lone letters or page numbers that      #
    #                      happen to be large).                            #
    # MAX_HEADING_CHARS  — headings longer than this are treated as body  #
    #                      text (catches large pull-quotes).               #
    # ------------------------------------------------------------------ #

    HEADING_SIZE_RATIO: float = 1.2
    MIN_HEADING_CHARS:  int   = 3
    MAX_HEADING_CHARS:  int   = 200

    # ------------------------------------------------------------------ #
    # TEXT CLEANING FLAGS                                                  #
    # All default True — can be overridden in the UI settings panel.      #
    # ------------------------------------------------------------------ #

    CLEAN_CITATIONS:    bool = True   # strip [1], [14], [1-3] etc.
    CLEAN_FIGURE_REFS:  bool = True   # "Fig. 3" → "Figure 3"
    CLEAN_URLS:         bool = True   # remove http/www links
    CLEAN_PAGE_NUMBERS: bool = True   # strip standalone integers
    FIX_HYPHENATION:    bool = True   # rejoin "impor-\ntant" → "important"
    NORMALIZE_ABBREVS:  bool = True   # "e.g." → "for example"
    NORMALIZE_NUMBERS:  bool = True   # "12%" → "12 percent"

    # ------------------------------------------------------------------ #
    # TTS CHUNKING                                                         #
    #                                                                      #
    # gTTS and OpenAI TTS both have per-request character limits.         #
    # We split text into sentence-boundary chunks before sending.         #
    #                                                                      #
    # MAX_TTS_CHUNK_CHARS — max characters per TTS API call.              #
    #   gTTS:   ~500 chars is safe (it handles longer but gets unstable). #
    #   OpenAI: 4096 chars max per request.                               #
    #   We use 500 for both so the same chunking works for either engine. #
    #                                                                      #
    # CHUNK_SILENCE_MS — milliseconds of silence inserted between         #
    #                    stitched audio chunks. Prevents choppy joins.    #
    # ------------------------------------------------------------------ #

    MAX_TTS_CHUNK_CHARS: int = 500
    CHUNK_SILENCE_MS:    int = 300

    # ------------------------------------------------------------------ #
    # AUDIO DEFAULTS                                                       #
    # ------------------------------------------------------------------ #

    DEFAULT_RATE:         float = 1.0    # speech rate multiplier
    DEFAULT_PITCH:        float = 1.0    # pitch multiplier
    DEFAULT_AUDIO_FORMAT: str   = "mp3"
    AUDIO_READING_WPM:    int   = 150    # words per minute for time estimate

    # OpenAI TTS voice — one of: alloy, echo, fable, onyx, nova, shimmer
    OPENAI_TTS_VOICE: str = "onyx"
    OPENAI_TTS_MODEL: str = "tts-1"

    # ------------------------------------------------------------------ #
    # PDF CONSTRAINTS                                                      #
    # ------------------------------------------------------------------ #

    MAX_PDF_SIZE_MB:   int  = 50                          # upload size cap
    MAX_PDF_SIZE_BYTES: int = MAX_PDF_SIZE_MB * 1024 * 1024

    # ------------------------------------------------------------------ #
    # CHAPTER MERGING                                                      #
    #                                                                      #
    # Chapters with fewer than MIN_CHAPTER_CHARS characters are merged    #
    # into the previous chapter. Prevents 1-sentence audio files.         #
    # ------------------------------------------------------------------ #

    MIN_CHAPTER_CHARS: int = 200

    # ------------------------------------------------------------------ #
    # API KEYS                                                             #
    # Loaded from environment variables only. Never hardcoded.            #
    # OPENAI_API_KEY is optional — app works without it using gTTS.       #
    # ------------------------------------------------------------------ #

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ------------------------------------------------------------------ #
    # FILE PATHS                                                           #
    # BASE_DIR is the project root (parent of this file).                 #
    # ------------------------------------------------------------------ #

    BASE_DIR:   Path = Path(__file__).resolve().parent
    CACHE_DIR:  Path = BASE_DIR / ".cache"
    LOG_DIR:    Path = BASE_DIR / "logs"
    TEMP_DIR:   Path = BASE_DIR / "temp"
    OUTPUT_DIR: Path = BASE_DIR / "output"

    # Ensure directories exist at import time
    for _d in (CACHE_DIR, LOG_DIR, TEMP_DIR, OUTPUT_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # LOGGING                                                              #
    # ------------------------------------------------------------------ #

    LOG_FILE:   Path = LOG_DIR / "app.log"
    LOG_FORMAT: str  = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    LOG_LEVEL:  str  = os.getenv("LOG_LEVEL", "INFO").upper()

    # ------------------------------------------------------------------ #
    # CLASS METHODS — language helpers                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def get_language_config(cls, lang_code: str) -> dict:
        """
        Return the full config dict for a language code.
        Raises KeyError if lang_code is not supported.
        """
        if lang_code not in cls.SUPPORTED_LANGUAGES:
            raise KeyError(
                f"Unsupported language code: '{lang_code}'. "
                f"Supported: {list(cls.SUPPORTED_LANGUAGES.keys())}"
            )
        return cls.SUPPORTED_LANGUAGES[lang_code]

    @classmethod
    def get_gtts_code(cls, lang_code: str) -> str:
        """Return the gTTS language code for a given language code."""
        return cls.get_language_config(lang_code)["gtts"]

    @classmethod
    def get_tesseract_code(cls, lang_code: str) -> str:
        """Return the Tesseract OCR language code for a given language code."""
        return cls.get_language_config(lang_code)["tesseract"]

    @classmethod
    def get_spacy_model(cls, lang_code: str) -> str | None:
        """
        Return the spaCy model name for a given language code.
        Returns None if no spaCy model is available for that language.
        Callers must handle None gracefully (disable NLP features).
        """
        return cls.get_language_config(lang_code)["spacy"]

    @classmethod
    def get_language_label(cls, lang_code: str) -> str:
        """Return the human-readable label for a given language code."""
        return cls.get_language_config(lang_code)["label"]

    @classmethod
    def get_all_language_labels(cls) -> dict[str, str]:
        """
        Return a dict mapping lang_code → human-readable label.
        Used to populate the UI language selector.
        """
        return {
            code: cfg["label"]
            for code, cfg in cls.SUPPORTED_LANGUAGES.items()
        }

    @classmethod
    def get_lang_code_from_label(cls, label: str) -> str:
        """
        Reverse lookup: given a human-readable label, return the lang_code.
        Raises ValueError if label is not found.
        """
        for code, cfg in cls.SUPPORTED_LANGUAGES.items():
            if cfg["label"].lower() == label.lower():
                return code
        raise ValueError(
            f"No language found with label '{label}'. "
            f"Available: {[c['label'] for c in cls.SUPPORTED_LANGUAGES.values()]}"
        )

    @classmethod
    def is_openai_available(cls) -> bool:
        """
        Return True if an OpenAI API key is configured and looks valid.
        Does NOT make a network call — only checks format.
        A key starting with 'sk-' and longer than 20 chars is considered valid format.
        """
        key = cls.OPENAI_API_KEY
        return bool(key) and key.startswith("sk-") and len(key) > 20

    @classmethod
    def get_active_tts_engine(cls) -> str:
        """
        Return the TTS engine that will actually be used.
        If TTS_ENGINE is set to 'openai' but no valid key is configured,
        falls back to 'gtts' and logs a warning.
        """
        if cls.TTS_ENGINE == "openai":
            if cls.is_openai_available():
                return "openai"
            # silently fall back — UI will show a warning banner
            return "gtts"
        return "gtts"

    @classmethod
    def has_spacy_support(cls, lang_code: str) -> bool:
        """Return True if a spaCy model is available for this language."""
        return cls.get_spacy_model(lang_code) is not None

    @classmethod
    def validate_extraction_mode(cls) -> str:
        """
        Return EXTRACTION_MODE if valid, else default to 'auto' with a warning.
        """
        valid = {"auto", "text", "ocr"}
        if cls.EXTRACTION_MODE in valid:
            return cls.EXTRACTION_MODE
        return "auto"


# ------------------------------------------------------------------ #
# Standalone test — run `python config.py` to verify setup           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print(f"App:              {Config.APP_NAME} v{Config.VERSION}")
    print(f"TTS Engine:       {Config.get_active_tts_engine()}")
    print(f"OpenAI Available: {Config.is_openai_available()}")
    print(f"Extraction Mode:  {Config.validate_extraction_mode()}")
    print(f"Base Dir:         {Config.BASE_DIR}")
    print(f"Log File:         {Config.LOG_FILE}")
    print(f"Max PDF Size:     {Config.MAX_PDF_SIZE_MB} MB")
    print(f"Languages:        {list(Config.SUPPORTED_LANGUAGES.keys())}")
    print()
    print("Language configs:")
    for code, cfg in Config.SUPPORTED_LANGUAGES.items():
        spacy = cfg['spacy'] or '(none)'
        print(f"  {code:8s} | gTTS: {cfg['gtts']:8s} | "
              f"Tesseract: {cfg['tesseract']:8s} | spaCy: {spacy}")