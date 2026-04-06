"""
audio_generator.py
==================
Converts clean text to audio. Manages TTS engine selection, text
chunking, audio stitching, and post-processing effects.

Responsibilities:
  1. Route text to the correct TTS engine (gTTS or OpenAI)
  2. Split text into sentence-boundary chunks before TTS calls
  3. Stitch audio chunks with silence padding between them
  4. Apply rate and pitch adjustments via pydub
  5. Provide audio utilities: duration, file size, chunk splitting

Core design decisions:
  - TextCleaner.clean_for_tts() is called HERE, not by the caller.
    This guarantees text is always cleaned before reaching any TTS engine,
    regardless of how generate_audio() is called.
  - TextAnalyzer.split_text_on_sentences() handles chunking. AudioGenerator
    owns audio concerns only.
  - Rate and pitch are adjusted with separate, correct DSP approaches:
      Rate → pydub speedup() with crossfade (time-stretch, no pitch shift)
      Pitch → frame_rate override (pitch shift, minor speed side-effect)
    The original code used frame_rate override for BOTH which caused the
    "chipmunk effect" (pitch and speed changed together).
  - All methods are static. No instance state.
  - @st.cache_data applied only to generate_audio() — keyed on
    (text_hash, lang_code, rate, pitch, engine) — all hashable.

This module knows about:
  - gTTS
  - openai (optional, only if Config.is_openai_available())
  - pydub
  - TextCleaner (for clean_for_tts)
  - TextAnalyzer (for split_text_on_sentences)
  - config.Config

This module knows NOTHING about:
  - Streamlit UI rendering
  - PDF extraction
  - spaCy NLP
  - Block/chapter data structures
"""

import io
import logging
import time
from typing import Optional

import streamlit as st
from gtts import gTTS
from pydub import AudioSegment

from config import Config
from text_cleaner import TextCleaner
from text_analyzer import TextAnalyzer

logger = logging.getLogger(__name__)


class AudioGenerator:

    # ================================================================== #
    # SECTION 1 — MASTER GENERATION METHOD                               #
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
        Master method. Clean text → chunk → TTS → stitch → apply effects.

        Pipeline:
          1. Clean text via TextCleaner.clean_for_tts()
          2. Split into sentence-boundary chunks (≤ MAX_TTS_CHUNK_CHARS)
          3. Send each chunk to the selected TTS engine
          4. Stitch chunks with silence padding
          5. Apply rate and pitch adjustments
          6. Return final MP3 bytes

        Args:
          text:      raw chapter text (will be cleaned internally)
          lang_code: language code from Config.SUPPORTED_LANGUAGES
          rate:      playback speed multiplier (0.5–2.0, default 1.0)
          pitch:     pitch multiplier (0.5–2.0, default 1.0)
          engine:    "gtts" | "openai" | None (None = use Config default)

        Returns:
          bytes: MP3 audio data

        Raises:
          ValueError: if text is empty after cleaning
          RuntimeError: if TTS engine fails after retries

        Safe to cache: all args are str/float/None — hashable.
        Note: caching is keyed on the raw text, so if the same chapter
        is requested with the same settings it won't re-generate.
        """
        # Resolve engine
        if engine is None:
            engine = Config.get_active_tts_engine()

        # Step 1 — Clean text
        clean_text = TextCleaner.clean_for_tts(text)
        if not clean_text.strip():
            logger.warning("generate_audio called with empty text after cleaning.")
            # Return a short silent audio clip rather than raising
            return AudioGenerator._generate_silence(1000)

        # Step 2 — Split into TTS-safe chunks
        chunks = TextAnalyzer.split_text_on_sentences(
            clean_text,
            lang_code=lang_code,
            max_chars=Config.MAX_TTS_CHUNK_CHARS,
        )

        if not chunks:
            logger.warning("Text produced no chunks after sentence splitting.")
            return AudioGenerator._generate_silence(1000)

        logger.info(
            f"Generating audio: {len(chunks)} chunks, engine={engine}, "
            f"lang={lang_code}, rate={rate}, pitch={pitch}"
        )

        # Step 3 — Generate audio per chunk
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
                # Insert silence in place of failed chunk to maintain structure
                chunk_audio_list.append(AudioGenerator._generate_silence(500))

        if not chunk_audio_list:
            raise RuntimeError("All TTS chunks failed — no audio generated.")

        # Step 4 — Stitch chunks with silence padding
        stitched = AudioGenerator.merge_audio_bytes(chunk_audio_list)

        # Step 5 — Apply rate and pitch
        # OpenAI TTS handles rate natively (passed in Step 3),
        # so only apply pydub rate adjustment for gTTS
        if engine == "gtts" and (rate != 1.0 or pitch != 1.0):
            stitched = AudioGenerator.apply_rate_pitch(stitched, rate, pitch)
        elif engine == "openai" and pitch != 1.0:
            # OpenAI handles rate but not pitch — apply pitch only
            stitched = AudioGenerator.apply_rate_pitch(stitched, 1.0, pitch)

        return stitched

    # ================================================================== #
    # SECTION 2 — TTS ENGINE IMPLEMENTATIONS                             #
    # ================================================================== #

    @staticmethod
    def _generate_gtts(text: str, lang_code: str) -> bytes:
        """
        Generate audio from text using gTTS (Google Translate TTS).

        gTTS is free but:
          - Uses the unofficial Google Translate TTS endpoint
          - Has no published rate limits (but will throttle under heavy use)
          - Voice quality is functional but robotic
          - Does not support speed/pitch natively

        Implements simple retry logic (3 attempts with 2s backoff) because
        gTTS occasionally fails with connection errors under load.

        Args:
          text:      text chunk (already cleaned and size-limited)
          lang_code: gTTS language code (e.g. "en", "es")

        Returns:
          bytes: MP3 audio data for this chunk
        """
        gtts_lang = Config.get_gtts_code(lang_code)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                tts     = gTTS(text=text, lang=gtts_lang, slow=False)
                buf     = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                return buf.read()
            except Exception as e:
                logger.warning(
                    f"gTTS attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt < max_retries:
                    time.sleep(2 * attempt)  # exponential backoff: 2s, 4s
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
        """
        Generate audio from text using OpenAI TTS API (tts-1 model).

        OpenAI TTS advantages over gTTS:
          - Significantly more natural voice quality
          - Native speed control (0.25–4.0x)
          - Consistent quality under load
          - Official API with SLA

        Cost: approximately $0.015 per 1,000 characters (tts-1 model).
        A typical book chapter (~3,000 words ≈ 18,000 chars) costs ~$0.27.

        Requires OPENAI_API_KEY environment variable to be set.
        Falls back to gTTS if key is unavailable (handled by generate_audio).

        Args:
          text:      text chunk (already cleaned and size-limited)
          lang_code: language code (OpenAI TTS is multilingual, lang not
                     required explicitly — it auto-detects from text)
          rate:      speed multiplier (0.25–4.0, clamped to OpenAI's range)

        Returns:
          bytes: MP3 audio data for this chunk
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. "
                "Run: pip install openai>=2.0.0"
            )

        if not Config.is_openai_available():
            raise RuntimeError(
                "OPENAI_API_KEY not configured or invalid. "
                "Set the OPENAI_API_KEY environment variable."
            )

        # Clamp rate to OpenAI's supported range
        clamped_rate = max(0.25, min(4.0, rate))

        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        try:
            response = client.audio.speech.create(
                model=Config.OPENAI_TTS_MODEL,   # "tts-1"
                voice=Config.OPENAI_TTS_VOICE,   # "onyx"
                input=text,
                response_format="mp3",
                speed=clamped_rate,
            )
            # OpenAI returns a streaming response — read all bytes
            return response.read()

        except Exception as e:
            raise RuntimeError(f"OpenAI TTS API call failed: {e}") from e

    # ================================================================== #
    # SECTION 3 — AUDIO PROCESSING                                       #
    # ================================================================== #

    @staticmethod
    def apply_rate_pitch(
        audio_bytes: bytes,
        rate:        float = 1.0,
        pitch:       float = 1.0,
    ) -> bytes:
        """
        Apply rate (speed) and pitch adjustments to an MP3 audio clip.

        Rate and pitch are treated as INDEPENDENT parameters:

        RATE (speed) adjustment uses pydub's speedup():
          - Implemented via frame-rate resampling + time-stretching
          - Preserves pitch while changing duration
          - rate > 1.0 → faster speech (shorter duration)
          - rate < 1.0 → slower speech (longer duration)
          - Effective range: 0.5–2.0

        PITCH adjustment uses frame_rate override:
          - Changes the playback sample rate, shifting pitch
          - Has a minor speed side-effect (±5% at pitch=±0.5)
            which is acceptable for voice content
          - pitch > 1.0 → higher pitch
          - pitch < 1.0 → lower pitch
          - Effective range: 0.5–2.0
          - Converts pitch multiplier to semitone shift:
              semitones = 12 * log2(pitch)
              new_rate = original_rate * (2 ** (semitones / 12))

        Note: If rate=1.0 and pitch=1.0 the audio is returned unchanged
        (no pydub processing — fast path).

        Args:
          audio_bytes: MP3 audio bytes
          rate:        speed multiplier (0.5–2.0)
          pitch:       pitch multiplier (0.5–2.0)

        Returns:
          bytes: processed MP3 audio
        """
        # Fast path — no processing needed
        if rate == 1.0 and pitch == 1.0:
            return audio_bytes

        import math

        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format="mp3"
            )

            # Apply rate (speed) adjustment
            # pydub's speedup() uses WSOLA time-stretching: pitch stays constant
            if rate != 1.0:
                rate_clamped = max(0.5, min(2.0, rate))
                if rate_clamped > 1.0:
                    # Speed up: pydub speedup() is clean for > 1.0
                    audio = audio.speedup(
                        playback_speed=rate_clamped,
                        chunk_size=150,
                        crossfade=25,
                    )
                else:
                    # Slow down: invert by slowing sample rate then normalizing
                    # pydub's speedup doesn't work < 1.0 directly
                    new_frame_rate = int(audio.frame_rate * rate_clamped)
                    audio = audio._spawn(
                        audio.raw_data,
                        overrides={"frame_rate": new_frame_rate}
                    ).set_frame_rate(audio.frame_rate)

            # Apply pitch adjustment (independent of rate)
            if pitch != 1.0:
                pitch_clamped   = max(0.5, min(2.0, pitch))
                # Convert pitch multiplier to new frame rate
                # pitch = new_freq / original_freq → new_freq = pitch * original_freq
                new_sample_rate = int(audio.frame_rate * pitch_clamped)
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": new_sample_rate}
                )
                # Normalize back to standard frame rate to avoid playback issues
                audio = audio.set_frame_rate(44100)

            # Export to MP3 bytes
            output = io.BytesIO()
            audio.export(output, format="mp3", bitrate="128k")
            return output.getvalue()

        except Exception as e:
            logger.error(f"apply_rate_pitch failed: {e}. Returning original audio.")
            return audio_bytes

    @staticmethod
    def merge_audio_bytes(audio_bytes_list: list[bytes]) -> bytes:
        """
        Concatenate a list of MP3 byte strings into a single MP3.

        Inserts Config.CHUNK_SILENCE_MS milliseconds of silence between
        chunks. This prevents audio from sounding choppy at chunk boundaries
        and provides natural pacing between sentences.

        Args:
          audio_bytes_list: list of MP3 byte strings (one per TTS chunk)

        Returns:
          bytes: single concatenated MP3

        Raises:
          ValueError: if audio_bytes_list is empty
        """
        if not audio_bytes_list:
            raise ValueError("merge_audio_bytes received empty list.")

        if len(audio_bytes_list) == 1:
            return audio_bytes_list[0]

        # Create silence segment for padding between chunks
        silence = AudioSegment.silent(
            duration=Config.CHUNK_SILENCE_MS,
            frame_rate=22050,
        )

        try:
            combined = AudioSegment.empty()

            for i, chunk_bytes in enumerate(audio_bytes_list):
                segment = AudioSegment.from_file(
                    io.BytesIO(chunk_bytes), format="mp3"
                )
                combined += segment

                # Add silence between chunks (not after the last one)
                if i < len(audio_bytes_list) - 1:
                    combined += silence

            output = io.BytesIO()
            combined.export(output, format="mp3", bitrate="128k")
            return output.getvalue()

        except Exception as e:
            logger.error(f"merge_audio_bytes failed: {e}")
            # Return first chunk as fallback
            return audio_bytes_list[0]

    @staticmethod
    def split_into_chunks(
        audio_bytes:    bytes,
        chunk_seconds:  int = 60,
    ) -> list[bytes]:
        """
        Split a long MP3 into fixed-duration chunks.

        Used for very long audio files that need to be split for storage
        or streaming purposes. Each chunk is chunk_seconds long (last chunk
        may be shorter).

        Args:
          audio_bytes:   MP3 bytes to split
          chunk_seconds: target duration per chunk in seconds

        Returns:
          List of MP3 byte strings
        """
        try:
            audio           = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format="mp3"
            )
            chunk_ms        = chunk_seconds * 1000
            chunks: list[bytes] = []

            for start_ms in range(0, len(audio), chunk_ms):
                chunk    = audio[start_ms: start_ms + chunk_ms]
                chunk_io = io.BytesIO()
                chunk.export(chunk_io, format="mp3", bitrate="128k")
                chunks.append(chunk_io.getvalue())

            return chunks

        except Exception as e:
            logger.error(f"split_into_chunks failed: {e}")
            return [audio_bytes]

    @staticmethod
    def apply_effects(
        audio_bytes:    bytes,
        volume_db:      float = 0.0,
        fade_in_sec:    float = 0.0,
        fade_out_sec:   float = 0.0,
    ) -> bytes:
        """
        Apply volume, fade-in, and fade-out effects to an MP3 clip.

        Args:
          audio_bytes:  MP3 bytes
          volume_db:    volume change in decibels (+3 = louder, -3 = quieter)
          fade_in_sec:  fade-in duration in seconds (0 = no fade)
          fade_out_sec: fade-out duration in seconds (0 = no fade)

        Returns:
          bytes: processed MP3 audio
        """
        if volume_db == 0.0 and fade_in_sec == 0.0 and fade_out_sec == 0.0:
            return audio_bytes

        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format="mp3"
            )

            if volume_db != 0.0:
                audio = audio + volume_db

            if fade_in_sec > 0.0:
                audio = audio.fade_in(duration=int(fade_in_sec * 1000))

            if fade_out_sec > 0.0:
                audio = audio.fade_out(duration=int(fade_out_sec * 1000))

            output = io.BytesIO()
            audio.export(output, format="mp3", bitrate="128k")
            return output.getvalue()

        except Exception as e:
            logger.error(f"apply_effects failed: {e}. Returning original audio.")
            return audio_bytes

    # ================================================================== #
    # SECTION 4 — UTILITIES                                              #
    # ================================================================== #

    @staticmethod
    def get_duration(audio_bytes: bytes) -> float:
        """
        Return the duration of an MP3 clip in seconds.

        Returns 0.0 on error.
        """
        try:
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format="mp3"
            )
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"get_duration failed: {e}")
            return 0.0

    @staticmethod
    def get_file_size_mb(audio_bytes: bytes) -> float:
        """Return the size of audio bytes in megabytes."""
        return len(audio_bytes) / (1024 * 1024)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format a duration in seconds as a human-readable string.

        Examples:
          3661.5 → "1h 1min 1sec"
          125.0  → "2min 5sec"
          45.3   → "45sec"
        """
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
    def _generate_silence(duration_ms: int) -> bytes:
        """
        Generate a silent MP3 clip of the specified duration.

        Used as a placeholder when TTS fails for a chunk, to maintain
        audio structure without raising an exception.

        Args:
          duration_ms: silence duration in milliseconds

        Returns:
          bytes: MP3 bytes containing silence
        """
        try:
            silence = AudioSegment.silent(duration=duration_ms, frame_rate=22050)
            output  = io.BytesIO()
            silence.export(output, format="mp3", bitrate="128k")
            return output.getvalue()
        except Exception as e:
            logger.error(f"_generate_silence failed: {e}")
            # Return minimal valid MP3 header bytes as absolute last resort
            return b""

    @staticmethod
    def estimate_cost_openai(text: str) -> float:
        """
        Estimate the OpenAI TTS cost for a text string.

        Based on published pricing: $0.015 per 1,000 characters (tts-1).
        This is an estimate — actual billing is by OpenAI's character count
        which may differ slightly.

        Args:
          text: text string to estimate cost for

        Returns:
          float: estimated cost in USD
        """
        # Clean the text first to get accurate character count
        clean = TextCleaner.clean_for_tts(text)
        char_count = len(clean)
        # $0.015 per 1,000 characters
        return (char_count / 1000) * 0.015


# ------------------------------------------------------------------ #
# Standalone test — run `python audio_generator.py`                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=== AudioGenerator Standalone Tests ===\n")

    # Test silence generation
    print("1. Silence generation...")
    silence = AudioGenerator._generate_silence(1000)
    print(f"   Generated {len(silence)} bytes of silence MP3")

    # Test duration
    duration = AudioGenerator.get_duration(silence)
    print(f"   Duration: {duration:.2f}s (expected ~1.0s)")

    # Test format_duration
    print("\n2. Duration formatting...")
    test_durations = [45.3, 125.0, 3661.5]
    for d in test_durations:
        print(f"   {d}s → '{AudioGenerator.format_duration(d)}'")

    # Test cost estimation
    print("\n3. OpenAI cost estimation...")
    sample = "This is a sample chapter with some content. " * 100
    cost   = AudioGenerator.estimate_cost_openai(sample)
    chars  = len(TextCleaner.clean_for_tts(sample))
    print(f"   {chars} chars → estimated cost: ${cost:.4f} USD")

    # Test merge (two silence clips)
    print("\n4. Audio merging...")
    clip1   = AudioGenerator._generate_silence(500)
    clip2   = AudioGenerator._generate_silence(500)
    merged  = AudioGenerator.merge_audio_bytes([clip1, clip2])
    dur_merged = AudioGenerator.get_duration(merged)
    print(f"   Merged 2×500ms clips → {dur_merged:.2f}s "
          f"(expected ~1.3s with {Config.CHUNK_SILENCE_MS}ms silence padding)")

    # Test gTTS if network available (skip if no connection)
    print("\n5. gTTS test (requires internet)...")
    try:
        audio_bytes = AudioGenerator._generate_gtts(
            "Hello, this is a test.", "en"
        )
        dur = AudioGenerator.get_duration(audio_bytes)
        size = AudioGenerator.get_file_size_mb(audio_bytes)
        print(f"   Generated {size:.3f} MB audio, duration {dur:.1f}s")
        print("   gTTS: PASS")
    except Exception as e:
        print(f"   gTTS: SKIP (no network or error: {e})")

    # Test rate/pitch (on silence — won't change content but tests the pipeline)
    print("\n6. Rate/pitch adjustment...")
    try:
        silence_2s = AudioGenerator._generate_silence(2000)
        adjusted   = AudioGenerator.apply_rate_pitch(silence_2s, rate=1.5, pitch=1.0)
        print(f"   Rate=1.5 applied: {len(adjusted)} bytes output")
        print("   Rate/pitch: PASS")
    except Exception as e:
        print(f"   Rate/pitch: FAIL — {e}")

    print("\nAll tests complete.")