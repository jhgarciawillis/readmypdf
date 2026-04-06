"""
audio_generator.py
==================
Converts clean text to audio. No pydub dependency — works on Python 3.14+.

Audio stitching is done at the raw MP3 byte level. For gTTS, each chunk
is a valid MP3 and concatenating them produces a valid VBR MP3 stream
that all modern players handle correctly.

Duration reading uses mutagen (pure Python, no native deps).

Rate/pitch adjustment is deferred to the OpenAI engine (native speed param)
or skipped for gTTS (acceptable tradeoff vs. broken pydub on Python 3.14).
"""

import io
import logging
import time
from typing import Optional

import streamlit as st
from gtts import gTTS

from config import Config
from text_cleaner import TextCleaner
from text_analyzer import TextAnalyzer

logger = logging.getLogger(__name__)


class AudioGenerator:

    # ================================================================== #
    # SECTION 1 - MASTER GENERATION METHOD                               #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_audio(
        text:      str,
        lang_code: str   = "en",
        rate:      float = 1.0,
        pitch:     float = 1.0,
        engine:    str   = None,
    ) -> bytes:
        """
        Master method. Clean text -> chunk -> TTS -> stitch -> return MP3 bytes.
        """
        if engine is None:
            engine = Config.get_active_tts_engine()

        clean_text = TextCleaner.clean_for_tts(text)
        if not clean_text.strip():
            logger.warning("generate_audio called with empty text after cleaning.")
            return AudioGenerator._generate_silence_bytes()

        chunks = TextAnalyzer.split_text_on_sentences(
            clean_text,
            lang_code=lang_code,
            max_chars=Config.MAX_TTS_CHUNK_CHARS,
        )

        if not chunks:
            return AudioGenerator._generate_silence_bytes()

        logger.info(
            f"Generating audio: {len(chunks)} chunks, engine={engine}, "
            f"lang={lang_code}, rate={rate}"
        )

        chunk_audio_list: list[bytes] = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                if engine == "openai":
                    chunk_bytes = AudioGenerator._generate_openai(chunk, lang_code, rate)
                else:
                    chunk_bytes = AudioGenerator._generate_gtts(chunk, lang_code)
                chunk_audio_list.append(chunk_bytes)
            except Exception as e:
                logger.error(f"TTS failed on chunk {i+1}/{len(chunks)}: {e}")
                chunk_audio_list.append(AudioGenerator._generate_silence_bytes())

        if not chunk_audio_list:
            raise RuntimeError("All TTS chunks failed.")

        return AudioGenerator.merge_audio_bytes(chunk_audio_list)

    # ================================================================== #
    # SECTION 2 - TTS ENGINE IMPLEMENTATIONS                             #
    # ================================================================== #

    @staticmethod
    def _generate_gtts(text: str, lang_code: str) -> bytes:
        """Generate MP3 bytes using gTTS with retry logic."""
        gtts_lang   = Config.get_gtts_code(lang_code)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                tts = gTTS(text=text, lang=gtts_lang, slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                return buf.read()
            except Exception as e:
                logger.warning(f"gTTS attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    raise RuntimeError(
                        f"gTTS failed after {max_retries} attempts: {e}"
                    ) from e

    @staticmethod
    def _generate_openai(
        text:      str,
        lang_code: str,
        rate:      float = 1.0,
    ) -> bytes:
        """Generate MP3 bytes using OpenAI TTS API (tts-1 model)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed.")

        if not Config.is_openai_available():
            raise RuntimeError("OPENAI_API_KEY not configured.")

        clamped_rate = max(0.25, min(4.0, rate))
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        try:
            response = client.audio.speech.create(
                model=Config.OPENAI_TTS_MODEL,
                voice=Config.OPENAI_TTS_VOICE,
                input=text,
                response_format="mp3",
                speed=clamped_rate,
            )
            return response.read()
        except Exception as e:
            raise RuntimeError(f"OpenAI TTS API call failed: {e}") from e

    # ================================================================== #
    # SECTION 3 - AUDIO STITCHING                                        #
    # ================================================================== #

    @staticmethod
    def merge_audio_bytes(audio_bytes_list: list[bytes]) -> bytes:
        """
        Concatenate MP3 byte strings into a single MP3 stream.

        Raw MP3 concatenation is valid because MP3 is a frame-based format.
        Each frame is self-contained and decoders resync at frame boundaries.
        A short silent MP3 segment is inserted between chunks for pacing.
        """
        if not audio_bytes_list:
            raise ValueError("merge_audio_bytes received empty list.")

        if len(audio_bytes_list) == 1:
            return audio_bytes_list[0]

        # 8 silent MP3 frames (~200ms silence between chunks)
        silent_frame = bytes([
            0xFF, 0xFB, 0x90, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]) * 8

        buf = io.BytesIO()
        for i, chunk_bytes in enumerate(audio_bytes_list):
            buf.write(chunk_bytes)
            if i < len(audio_bytes_list) - 1:
                buf.write(silent_frame)

        return buf.getvalue()

    @staticmethod
    def split_into_chunks(
        audio_bytes:   bytes,
        chunk_seconds: int = 60,
    ) -> list[bytes]:
        """Split MP3 bytes into chunks by estimated byte size."""
        bytes_per_sec = 16_000
        chunk_size    = bytes_per_sec * chunk_seconds
        chunks        = []

        for start in range(0, len(audio_bytes), chunk_size):
            chunks.append(audio_bytes[start: start + chunk_size])

        return chunks if chunks else [audio_bytes]

    # ================================================================== #
    # SECTION 4 - UTILITIES                                              #
    # ================================================================== #

    @staticmethod
    def get_duration(audio_bytes: bytes) -> float:
        """Return MP3 duration in seconds using mutagen. Falls back to byte estimate."""
        try:
            from mutagen.mp3 import MP3
            audio = MP3(io.BytesIO(audio_bytes))
            return audio.info.length
        except Exception as e:
            logger.warning(f"get_duration failed: {e}")
            return len(audio_bytes) / 16_000.0

    @staticmethod
    def get_file_size_mb(audio_bytes: bytes) -> float:
        """Return size of audio bytes in megabytes."""
        return len(audio_bytes) / (1024 * 1024)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format seconds as human-readable duration string."""
        total_sec = int(seconds)
        hours     = total_sec // 3600
        minutes   = (total_sec % 3600) // 60
        secs      = total_sec % 60

        if hours > 0:
            return f"{hours}h {minutes}min {secs}sec"
        elif minutes > 0:
            return f"{minutes}min {secs}sec"
        else:
            return f"{secs}sec"

    @staticmethod
    def estimate_cost_openai(text: str) -> float:
        """Estimate OpenAI TTS cost at $0.015 per 1,000 characters."""
        clean      = TextCleaner.clean_for_tts(text)
        char_count = len(clean)
        return (char_count / 1000) * 0.015

    @staticmethod
    def _generate_silence_bytes() -> bytes:
        """Return minimal valid silent MP3 frames as a placeholder."""
        return bytes([
            0xFF, 0xFB, 0x90, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]) * 8

    @staticmethod
    def apply_rate_pitch(
        audio_bytes: bytes,
        rate:        float = 1.0,
        pitch:       float = 1.0,
    ) -> bytes:
        """
        Rate/pitch placeholder. pydub removed (broken on Python 3.14).
        Audio returned unchanged. Use OpenAI engine for native speed control.
        """
        if rate != 1.0 or pitch != 1.0:
            logger.info(
                "Rate/pitch requested but unavailable without pydub. "
                "Use OpenAI TTS engine for speed control."
            )
        return audio_bytes

    @staticmethod
    def apply_effects(
        audio_bytes:  bytes,
        volume_db:    float = 0.0,
        fade_in_sec:  float = 0.0,
        fade_out_sec: float = 0.0,
    ) -> bytes:
        """Audio effects placeholder. Returns audio unchanged."""
        return audio_bytes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== AudioGenerator Tests ===\n")

    print("1. Silence generation...")
    silence = AudioGenerator._generate_silence_bytes()
    print(f"   {len(silence)} bytes")

    print("\n2. Duration formatting...")
    for d in [45.3, 125.0, 3661.5]:
        print(f"   {d}s -> '{AudioGenerator.format_duration(d)}'")

    print("\n3. Merge test...")
    s1     = AudioGenerator._generate_silence_bytes()
    s2     = AudioGenerator._generate_silence_bytes()
    merged = AudioGenerator.merge_audio_bytes([s1, s2])
    print(f"   {len(s1)}+{len(s2)} bytes -> {len(merged)} bytes merged")

    print("\n4. gTTS test (requires internet)...")
    try:
        audio = AudioGenerator._generate_gtts("Hello, this is a test.", "en")
        dur   = AudioGenerator.get_duration(audio)
        size  = AudioGenerator.get_file_size_mb(audio)
        print(f"   {size:.3f} MB, {dur:.1f}s - PASS")
    except Exception as e:
        print(f"   SKIP ({e})")

    print("\nDone.")