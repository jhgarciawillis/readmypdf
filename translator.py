"""
translator.py
=============
Language detection and translation. All translation logic lives here.
No other module needs to know which engine is being used.

Responsibilities:
  1. Detect language of extracted text (langdetect)
  2. Translate text via Google Translate unofficial API (googletrans)
  3. Translate text via LibreTranslate REST API (requests)
  4. Fallback chain: primary engine → fallback engine → original text
  5. Chunk long texts for APIs with character limits

This module knows about:
  - langdetect
  - googletrans
  - requests
  - config.Config (for engine settings)

This module knows NOTHING about:
  - Streamlit
  - PDF processing
  - Audio generation
  - UI components

All functions are module-level (not class methods) — translation is
stateless and doesn't benefit from the class pattern.
"""

import asyncio
import logging
import re
import time
from typing import Optional

import requests

from config import Config

logger = logging.getLogger(__name__)


# ================================================================== #
# SECTION 1 — LANGUAGE DETECTION                                      #
# ================================================================== #

def detect_language(text: str) -> str:
    """
    Detect the language of a text string using langdetect.

    Uses the first 800 characters for speed — enough for reliable detection.
    Sets DetectorFactory.seed = 0 for deterministic results (langdetect is
    non-deterministic by default).

    Args:
      text: any text string (raw extracted PDF text works fine)

    Returns:
      ISO 639-1 language code string (e.g. "en", "es", "fr", "zh-cn").
      Returns "en" on any failure — safe fallback.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # make results deterministic

        # Use a clean sample — strip very short words and digits
        # which can confuse the detector
        sample = text[:800].strip()
        if len(sample) < 20:
            logger.info("Text too short for reliable language detection — defaulting to 'en'")
            return "en"

        detected = detect(sample)
        mapped   = Config.map_detected_lang_to_supported(detected)

        logger.info(
            f"Language detected: '{detected}' → mapped to supported code '{mapped}'"
        )
        return mapped

    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
        return "en"


# ================================================================== #
# SECTION 2 — GOOGLE TRANSLATE ENGINE                                 #
# ================================================================== #

def _translate_google(
    text:        str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    Translate text using the unofficial Google Translate API via googletrans.

    googletrans 4.x uses an async API. We run it synchronously using
    asyncio.run() which creates a new event loop for each call — safe
    in a Streamlit context where there is no running event loop.

    Args:
      text:        text to translate (should be ≤ 4500 chars)
      source_lang: source language code (e.g. "en", "es")
      target_lang: target language code

    Returns:
      Translated text string.

    Raises:
      RuntimeError on failure — caller handles fallback.
    """
    try:
        from googletrans import Translator

        # Map our lang codes to googletrans format
        # googletrans uses "zh-cn" not "zh-CN"
        src = source_lang.lower().replace("zh-cn", "zh-cn")
        dst = target_lang.lower().replace("zh-cn", "zh-cn")

        async def _do_translate():
            async with Translator() as translator:
                result = await translator.translate(text, src=src, dest=dst)
                return result.text

        translated = asyncio.run(_do_translate())

        if not translated or not translated.strip():
            raise RuntimeError("Google Translate returned empty result")

        # Detect silent failure: if result is identical to input AND
        # source and target languages differ, translation likely failed
        # (googletrans sometimes returns original text when network is unreliable)
        if (translated.strip() == text.strip()
                and source_lang != target_lang
                and source_lang not in ("auto", "")
                and len(text) > 20):
            raise RuntimeError(
                f"Google Translate returned original text unchanged "
                f"({source_lang}→{target_lang}) — likely a network detection failure. "
                "This often happens when Google misidentifies the source language."
            )

        return translated

    except ImportError:
        raise RuntimeError("googletrans package not installed")
    except Exception as e:
        raise RuntimeError(f"Google Translate failed: {e}") from e


# ================================================================== #
# SECTION 3 — LIBRETRANSLATE ENGINE                                   #
# ================================================================== #

def _translate_libretranslate(
    text:        str,
    source_lang: str,
    target_lang: str,
    api_url:     str,
    api_key:     str = "",
) -> str:
    """
    Translate text using the LibreTranslate REST API.

    LibreTranslate is open-source and can be:
      - Self-hosted (no rate limits, no API key required)
      - Used via public instance at libretranslate.com
        (rate-limited ~80 req/hour, API key optional)

    Maps our lang codes to LibreTranslate format (lowercase ISO 639-1).

    Args:
      text:        text to translate
      source_lang: source language code
      target_lang: target language code
      api_url:     LibreTranslate base URL (e.g. "https://libretranslate.com")
      api_key:     API key (empty string for public instance)

    Returns:
      Translated text string.

    Raises:
      RuntimeError on failure.
    """
    # Normalize lang codes for LibreTranslate (lowercase, no variants)
    src = source_lang.lower().replace("zh-cn", "zh").replace("zh-tw", "zh")
    dst = target_lang.lower().replace("zh-cn", "zh").replace("zh-tw", "zh")

    endpoint = api_url.rstrip("/") + "/translate"

    payload = {
        "q":      text,
        "source": src,
        "target": dst,
        "format": "text",
    }
    if api_key:
        payload["api_key"] = api_key

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 429:
            raise RuntimeError(
                "LibreTranslate rate limit exceeded. "
                "Try again in a few minutes or configure a self-hosted instance."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "LibreTranslate returned 403 — API key may be required. "
                "Set LIBRETRANSLATE_API_KEY environment variable."
            )

        response.raise_for_status()

        data = response.json()
        translated = data.get("translatedText", "")

        if not translated or not translated.strip():
            raise RuntimeError("LibreTranslate returned empty result")

        return translated

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to LibreTranslate at {api_url}. "
            "Check the URL or try again."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"LibreTranslate request timed out at {api_url}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"LibreTranslate request failed: {e}") from e


# ================================================================== #
# SECTION 4 — CHUNKING                                                #
# ================================================================== #

def _split_for_translation(text: str, max_chars: int = 4500) -> list[str]:
    """
    Split text into chunks suitable for translation API calls.

    Splits on paragraph boundaries (double newlines) first, then on
    sentence boundaries if a paragraph exceeds max_chars.
    Preserves paragraph structure so translated text reassembles cleanly.

    Args:
      text:      full text to split
      max_chars: maximum characters per chunk

    Returns:
      List of text chunk strings.
    """
    # Split on paragraph boundaries first
    paragraphs = re.split(r"\n\n+", text.strip())

    chunks:       list[str] = []
    current:      list[str] = []
    current_len:  int       = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds max_chars, split on sentences
        if len(para) > max_chars:
            # Flush current buffer first
            if current:
                chunks.append("\n\n".join(current))
                current     = []
                current_len = 0
            # Split paragraph into sentences
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
            sub_chunk: list[str] = []
            sub_len:   int       = 0
            for sent in sentences:
                if sub_len + len(sent) > max_chars and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_len   = 0
                sub_chunk.append(sent)
                sub_len += len(sent) + 1
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        # Normal paragraph — accumulate
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current     = []
            current_len = 0

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


# ================================================================== #
# SECTION 5 — FALLBACK ORCHESTRATION                                  #
# ================================================================== #

def translate_with_fallback(
    text:        str,
    source_lang: str,
    target_lang: str,
    config:      dict,
) -> tuple[str, str]:
    """
    Translate text using the configured engine chain with automatic fallback.

    Engine chain:
      1. Try primary engine (default: google)
      2. If primary fails, try fallback engine (default: libretranslate)
      3. If both fail, return original text with a warning

    Args:
      text:        text to translate
      source_lang: source language code (e.g. "en")
      target_lang: target language code (e.g. "es")
      config:      dict from Config.get_translation_config()

    Returns:
      Tuple of (translated_text, engine_used)
      engine_used is "google", "libretranslate", or "original" (if both failed)
    """
    if source_lang == target_lang:
        return text, "original (same language)"

    if not text or not text.strip():
        return text, "original (empty)"

    primary  = config.get("primary_engine",  "google")
    fallback = config.get("fallback_engine", "libretranslate")

    engines_to_try = []
    if primary in ("google", "libretranslate"):
        engines_to_try.append(primary)
    if fallback in ("google", "libretranslate") and fallback != primary:
        engines_to_try.append(fallback)

    for engine in engines_to_try:
        try:
            translated = chunk_and_translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                engine=engine,
                config=config,
            )
            logger.info(
                f"Translation succeeded via {engine}: "
                f"{len(text)} → {len(translated)} chars"
            )
            return translated, engine

        except Exception as e:
            logger.warning(f"Translation engine '{engine}' failed: {e}")
            # Small delay before trying fallback
            time.sleep(1)
            continue

    # Both engines failed — return original text
    logger.error(
        f"All translation engines failed for {source_lang}→{target_lang}. "
        f"Returning original text."
    )
    return text, "original (translation failed)"


def chunk_and_translate(
    text:        str,
    source_lang: str,
    target_lang: str,
    engine:      str,
    config:      dict,
) -> str:
    """
    Split text into API-safe chunks, translate each, and rejoin.

    Args:
      text:        full chapter or document text
      source_lang: source language code
      target_lang: target language code
      engine:      "google" or "libretranslate"
      config:      dict from Config.get_translation_config()

    Returns:
      Full translated text string with paragraphs preserved.
    """
    max_chars = config.get("max_chunk_chars", 4500)
    chunks    = _split_for_translation(text, max_chars)

    if not chunks:
        return text

    translated_chunks: list[str] = []

    for i, chunk in enumerate(chunks):
        logger.debug(
            f"Translating chunk {i+1}/{len(chunks)} "
            f"({len(chunk)} chars) via {engine}"
        )

        if engine == "google":
            translated_chunk = _translate_google(chunk, source_lang, target_lang)
        elif engine == "libretranslate":
            translated_chunk = _translate_libretranslate(
                text=chunk,
                source_lang=source_lang,
                target_lang=target_lang,
                api_url=config.get("libretranslate_url", "https://libretranslate.com"),
                api_key=config.get("libretranslate_api_key", ""),
            )
        else:
            raise ValueError(f"Unknown translation engine: '{engine}'")

        translated_chunks.append(translated_chunk)

        # Small delay between chunks to be respectful to free API endpoints
        if i < len(chunks) - 1:
            time.sleep(0.3)

    return "\n\n".join(translated_chunks)


# ================================================================== #
# SECTION 6 — ENGINE AVAILABILITY CHECK                               #
# ================================================================== #

def check_libretranslate_connection(api_url: str, api_key: str = "") -> dict:
    """
    Ping the LibreTranslate /languages endpoint to verify connectivity
    and get the list of supported language pairs.

    Returns dict: {
      "available": bool,
      "languages": list of {"code": str, "name": str} dicts,
      "error": str or None
    }
    Called by the UI to show engine status indicators.
    """
    try:
        endpoint = api_url.rstrip("/") + "/languages"
        params   = {}
        if api_key:
            params["api_key"] = api_key

        response = requests.get(endpoint, params=params, timeout=5)
        response.raise_for_status()

        languages = response.json()
        return {
            "available": True,
            "languages": languages,
            "error":     None,
        }

    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "languages": [],
            "error":     f"Cannot connect to {api_url}",
        }
    except requests.exceptions.Timeout:
        return {
            "available": False,
            "languages": [],
            "error":     "Connection timed out",
        }
    except Exception as e:
        return {
            "available": False,
            "languages": [],
            "error":     str(e),
        }


def check_google_translate_connection() -> dict:
    """
    Attempt a minimal translation to verify Google Translate is working.

    Returns dict: { "available": bool, "error": str or None }
    """
    try:
        result, engine = translate_with_fallback(
            text="hello",
            source_lang="en",
            target_lang="es",
            config={
                "primary_engine":         "google",
                "fallback_engine":        "none",
                "max_chunk_chars":        100,
                "google_available":       True,
                "libretranslate_available": False,
            },
        )
        if engine == "original (translation failed)":
            return {"available": False, "error": "Translation returned original"}
        return {"available": True, "error": None}
    except Exception as e:
        return {"available": False, "error": str(e)}


# ================================================================== #
# Standalone test                                                      #
# ================================================================== #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Language Detection Tests ===\n")
    samples = [
        ("English",  "Every executive I've met believes they know how to build great teams."),
        ("Spanish",  "En 2023, las fintechs latinoamericanas recaudaron 2.1 mil millones de dólares."),
        ("French",   "La recherche montre que la sécurité psychologique est le facteur le plus important."),
        ("German",   "Die Forschung zeigt, dass psychologische Sicherheit der wichtigste Faktor ist."),
        ("Portuguese", "A pesquisa mostra que a segurança psicológica é o fator mais importante."),
    ]
    for label, text in samples:
        detected = detect_language(text)
        print(f"  {label}: detected='{detected}'")

    print("\n=== Translation Chunk Splitting ===\n")
    long_text = "This is paragraph one.\n\nThis is paragraph two with more content.\n\nThird paragraph here."
    chunks = _split_for_translation(long_text, max_chars=50)
    print(f"  Split '{long_text[:40]}...' into {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i+1}: {repr(c[:50])}")

    print("\n=== Google Translate Test (requires internet) ===\n")
    try:
        result = _translate_google(
            "High-performing teams require psychological safety above all else.",
            "en", "es"
        )
        print(f"  EN→ES: {result}")
    except Exception as e:
        print(f"  SKIP: {e}")