"""
streamlit_app.py
================
Entry point and orchestrator. Wires all modules together in the correct
order. Contains zero business logic — all computation is delegated.

Responsibilities:
  1. Configure the Streamlit page
  2. Initialize session state
  3. Accept PDF input (upload or local file)
  4. Run the processing pipeline when triggered
  5. Display results using UIComponents
  6. Handle errors gracefully with user-facing messages

Pipeline (triggered by "Convert to Audio" button):
  1. Read pdf_bytes ONCE — passed as bytes everywhere
  2. Validate PDF
  3. Detect extraction mode (text vs OCR)
  4. Extract structured blocks (PDFProcessor)
  5. Build document structure (TextAnalyzer)
     — header/footer removal, heading detection, chapter splitting
  6. Generate audio per chapter (AudioGenerator)
     — text cleaning and TTS chunking happen inside AudioGenerator
  7. Store results in session state
  8. Display results

Session state keys:
  pdf_bytes         bytes        — raw PDF, read once
  audio_data        dict         — { chapter_title: mp3_bytes }
  document_structure dict        — from TextAnalyzer.build_document_structure()
  extraction_result  dict        — from PDFProcessor.extract()
  current_chapter   str          — title of active chapter in player
  bookmarks         list[str]    — bookmarked chapter titles
  processing_done   bool         — True after successful pipeline run
  settings          dict         — last-used settings dict from sidebar
"""

import logging
import traceback
from collections import OrderedDict

import streamlit as st

from config import Config
from text_cleaner import TextCleaner
from pdf_processor import PDFProcessor
from text_analyzer import TextAnalyzer
from audio_generator import AudioGenerator
from ui_components import UIComponents

# ------------------------------------------------------------------ #
# Logging setup                                                        #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format=Config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


# ================================================================== #
# SESSION STATE INITIALISATION                                         #
# ================================================================== #

def initialize_session_state() -> None:
    """
    Set default values for all session state keys.
    Only sets a key if it does not already exist — safe to call on
    every rerun without resetting user state.
    """
    defaults = {
        "pdf_bytes":          None,
        "audio_data":         {},
        "document_structure": None,
        "extraction_result":  None,
        "current_chapter":    None,
        "bookmarks":          [],
        "processing_done":    False,
        "settings":           {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ================================================================== #
# PROCESSING PIPELINE                                                  #
# ================================================================== #

def run_pipeline(
    pdf_bytes: bytes,
    settings:  dict,
) -> None:
    """
    Execute the full PDF → Audio pipeline.

    Called when the user clicks "Convert to Audio". All results are
    stored in st.session_state so the display section can access them
    on subsequent reruns.

    Args:
      pdf_bytes: raw PDF bytes (read once by caller before this call)
      settings:  dict from UIComponents.render_sidebar_settings()

    On success: sets st.session_state.processing_done = True
    On failure: displays error message, leaves processing_done = False
    """
    lang_code    = settings["lang_code"]
    tts_engine   = settings["tts_engine"]
    rate         = settings["rate"]
    pitch        = settings["pitch"]
    start_page   = settings["start_page"]
    end_page     = settings["end_page"]
    remove_hf    = settings["remove_headers_footers"]

    # ---- Step 1: Validate ----
    with st.spinner("Validating PDF…"):
        valid, err = PDFProcessor.validate_pdf(pdf_bytes)
        if not valid:
            UIComponents.render_error(err)
            return

    # ---- Step 2: Detect extraction mode ----
    with st.spinner("Analysing PDF structure…"):
        configured_mode = Config.validate_extraction_mode()

        if configured_mode == "auto":
            detected_mode = TextCleaner.detect_pdf_mode(pdf_bytes)
        else:
            detected_mode = configured_mode

        logger.info(f"Extraction mode: {detected_mode}")

    # ---- Step 3: Extract blocks ----
    page_count = PDFProcessor.get_page_count(pdf_bytes)

    # Resolve end_page: if user left it at 1 (no PDF was loaded when
    # sidebar rendered), default to all pages
    resolved_end = end_page if end_page > 1 else page_count

    with st.spinner(
        f"Extracting text from pages {start_page}–{resolved_end} "
        f"({'text mode' if detected_mode == 'text' else 'OCR mode'})…"
    ):
        try:
            extraction_result = PDFProcessor.extract(
                pdf_bytes=pdf_bytes,
                lang_code=lang_code,
                start_page=start_page,
                end_page=resolved_end,
                mode=detected_mode,
            )
        except Exception as e:
            UIComponents.render_error(
                f"Text extraction failed: {e}\n\n"
                f"If this is a scanned PDF, try switching extraction mode "
                f"to OCR in the environment settings."
            )
            logger.error(traceback.format_exc())
            return

    blocks     = extraction_result["blocks"]
    mode_used  = extraction_result["mode_used"]

    if not blocks:
        UIComponents.render_error(
            "No text could be extracted from this PDF. "
            "If it is a scanned document, ensure Tesseract OCR is installed "
            "and the correct language is selected."
        )
        return

    # ---- Step 4: Build document structure ----
    with st.spinner("Detecting headings and building chapter structure…"):
        try:
            document_structure = TextAnalyzer.build_document_structure(
                blocks=blocks,
                lang_code=lang_code,
                remove_headers=remove_hf,
            )
        except Exception as e:
            UIComponents.render_error(f"Chapter detection failed: {e}")
            logger.error(traceback.format_exc())
            return

    chapters = document_structure["chapters"]

    if not chapters:
        UIComponents.render_error(
            "Document structure could not be determined. "
            "The PDF may contain only images or non-standard encoding."
        )
        return

    logger.info(f"Chapters detected: {list(chapters.keys())}")

    # ---- Step 5: Generate audio per chapter ----
    audio_data: dict[str, bytes] = {}
    chapter_titles = list(chapters.keys())
    total_chapters = len(chapter_titles)

    progress_bar = st.progress(0, text="Generating audio…")

    for i, chapter_title in enumerate(chapter_titles):
        chapter_text = chapters[chapter_title]

        if not chapter_text.strip():
            logger.warning(f"Skipping empty chapter: '{chapter_title}'")
            continue

        progress_text = (
            f"Generating audio: '{chapter_title}' "
            f"({i + 1}/{total_chapters})…"
        )
        progress_bar.progress((i) / total_chapters, text=progress_text)

        try:
            audio_bytes = AudioGenerator.generate_audio(
                text=chapter_text,
                lang_code=lang_code,
                rate=rate,
                pitch=pitch,
                engine=tts_engine,
            )
            audio_data[chapter_title] = audio_bytes
            logger.info(
                f"Audio generated for '{chapter_title}': "
                f"{AudioGenerator.get_file_size_mb(audio_bytes):.2f} MB, "
                f"{AudioGenerator.format_duration(AudioGenerator.get_duration(audio_bytes))}"
            )
        except Exception as e:
            logger.error(
                f"Audio generation failed for '{chapter_title}': {e}"
            )
            # Insert silence so the chapter still appears in the player
            audio_data[chapter_title] = AudioGenerator._generate_silence_bytes()

    progress_bar.progress(1.0, text="Audio generation complete.")

    if not audio_data:
        UIComponents.render_error(
            "Audio generation failed for all chapters. "
            "Check your internet connection (gTTS requires network access) "
            "or switch to a different TTS engine."
        )
        return

    # ---- Step 6: Store results in session state ----
    st.session_state.audio_data          = audio_data
    st.session_state.document_structure  = document_structure
    st.session_state.extraction_result   = extraction_result
    st.session_state.pdf_bytes           = pdf_bytes
    st.session_state.settings            = settings
    st.session_state.processing_done     = True

    # Set initial chapter to the first one
    if st.session_state.current_chapter not in audio_data:
        st.session_state.current_chapter = chapter_titles[0]

    logger.info(
        f"Pipeline complete. {len(audio_data)} chapters, "
        f"mode={mode_used}, engine={tts_engine}."
    )


# ================================================================== #
# DISPLAY SECTION                                                      #
# ================================================================== #

def render_results() -> None:
    """
    Render all results after a successful pipeline run.
    Reads from st.session_state. Called on every rerun when
    st.session_state.processing_done is True.
    """
    audio_data          = st.session_state.audio_data
    document_structure  = st.session_state.document_structure
    extraction_result   = st.session_state.extraction_result
    pdf_bytes           = st.session_state.pdf_bytes
    settings            = st.session_state.settings
    current_chapter     = st.session_state.current_chapter

    chapters   = document_structure["chapters"]
    mode_used  = extraction_result["mode_used"]
    page_count = extraction_result["page_count"]
    metadata   = extraction_result["metadata"]

    # Validate current_chapter — could be stale after reprocessing
    if current_chapter not in audio_data:
        current_chapter = list(audio_data.keys())[0]
        st.session_state.current_chapter = current_chapter

    # ---- Processing status banner ----
    UIComponents.render_processing_status(mode_used, page_count)

    # ---- Metadata ----
    UIComponents.render_metadata(metadata)

    # ---- PDF preview (lazy) ----
    UIComponents.render_pdf_preview(pdf_bytes)

    st.divider()

    # ---- Cost estimate (OpenAI only) ----
    UIComponents.render_cost_estimate(chapters, settings.get("tts_engine", "gtts"))

    # ---- Main layout: TOC + Audio player ----
    col_toc, col_player = st.columns([1, 3])

    with col_toc:
        UIComponents.render_table_of_contents(chapters, current_chapter)

        st.divider()

        UIComponents.render_bookmark_button(current_chapter)
        UIComponents.render_bookmarks_list(chapters)

    with col_player:
        current_audio = audio_data.get(current_chapter, b"")
        chapter_index = list(chapters.keys()).index(current_chapter)

        UIComponents.render_audio_player(
            audio_bytes=current_audio,
            chapter_title=current_chapter,
            chapter_index=chapter_index,
            total_chapters=len(chapters),
        )

        UIComponents.render_chapter_navigation(chapters, current_chapter)

        st.divider()

        UIComponents.render_progress(current_chapter, chapters)

        st.divider()

        UIComponents.render_download_buttons(
            audio_data=audio_data,
            current_chapter=current_chapter,
            chapters=chapters,
        )

    # ---- Analysis panel ----
    if settings.get("show_analysis", False):
        st.divider()
        _render_analysis_section(document_structure, settings["lang_code"])


def _render_analysis_section(
    document_structure: dict,
    lang_code: str,
) -> None:
    """
    Compute and render the full document intelligence panel.
    Called lazily — only when show_analysis is True in settings.
    """
    chapters     = document_structure["chapters"]
    full_text    = TextAnalyzer.get_full_text(chapters)
    word_count   = TextAnalyzer.get_word_count(full_text)
    reading_time = TextAnalyzer.get_reading_time_estimate(full_text)

    with st.spinner("Running document analysis…"):
        keywords           = TextAnalyzer.extract_keywords(full_text, lang_code)
        characters         = TextAnalyzer.detect_character_names(full_text, lang_code)
        summary            = TextAnalyzer.summarize_text(full_text, lang_code)
        sentiment          = TextAnalyzer.sentiment_analysis(full_text, lang_code)
        readability        = TextAnalyzer.compute_readability(full_text)
        text_stats         = TextAnalyzer.compute_text_stats(full_text, chapters)
        content_type       = TextAnalyzer.detect_content_type(full_text, chapters)
        topic_density      = TextAnalyzer.compute_topic_density(full_text, keywords)
        chapter_complexity = TextAnalyzer.detect_language_complexity_by_chapter(chapters)

    UIComponents.render_analysis_panel(
        keywords=keywords,
        characters=characters,
        summary=summary,
        sentiment=sentiment,
        reading_time=reading_time,
        word_count=word_count,
        readability=readability,
        text_stats=text_stats,
        content_type=content_type,
        topic_density=topic_density,
        chapter_complexity=chapter_complexity,
    )


# ================================================================== #
# MAIN                                                                 #
# ================================================================== #

def main() -> None:
    """
    Main Streamlit entry point. Called on every rerun.

    Structure:
      1. Page config (runs once per session)
      2. Session state init (safe to call every rerun)
      3. Sidebar settings (always rendered)
      4. File input section
      5. Processing trigger
      6. Results display (if processing_done)
    """
    # ---- Page config ----
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Session state ----
    initialize_session_state()

    # ---- Sidebar (always rendered, even before PDF is loaded) ----
    # Determine max_pages for the page range selector
    max_pages = 1
    if st.session_state.pdf_bytes:
        try:
            max_pages = PDFProcessor.get_page_count(st.session_state.pdf_bytes)
        except Exception:
            max_pages = 1

    settings = UIComponents.render_sidebar_settings(max_pages=max_pages)

    # ---- Header ----
    st.title(f"🎧 {Config.APP_NAME}")
    st.caption(Config.DESCRIPTION)

    # ---- File input section ----
    st.subheader("Upload PDF")

    file_source = st.radio(
        "File source",
        options=["Upload file", "Local file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    pdf_file   = None
    pdf_bytes  = None

    if file_source == "Upload file":
        pdf_file = UIComponents.file_uploader_widget()
        if pdf_file is not None:
            # READ THE FILE EXACTLY ONCE HERE
            # All downstream calls use pdf_bytes (bytes), not pdf_file
            pdf_bytes = pdf_file.read()
    else:
        local_path = UIComponents.local_file_selector(folder_path=".")
        if local_path:
            try:
                with open(local_path, "rb") as f:
                    pdf_bytes = f.read()
            except OSError as e:
                UIComponents.render_error(f"Could not read file: {e}")

    # ---- Processing trigger ----
    if pdf_bytes is not None:
        # Show basic file info
        size_mb = len(pdf_bytes) / (1024 * 1024)
        st.caption(f"File loaded: {size_mb:.2f} MB")

        st.divider()

        # Show cost estimate if OpenAI is selected and we have previous chapters
        if (
            settings["tts_engine"] == "openai"
            and st.session_state.processing_done
            and st.session_state.document_structure
        ):
            UIComponents.render_cost_estimate(
                st.session_state.document_structure["chapters"],
                settings["tts_engine"],
            )

        # Convert button
        if st.button(
            "🎙️ Convert to Audio",
            type="primary",
            use_container_width=True,
        ):
            # Reset previous results before running new pipeline
            st.session_state.processing_done    = False
            st.session_state.audio_data         = {}
            st.session_state.document_structure = None
            st.session_state.extraction_result  = None
            st.session_state.current_chapter    = None
            st.session_state.bookmarks          = []

            run_pipeline(pdf_bytes=pdf_bytes, settings=settings)

            if st.session_state.processing_done:
                UIComponents.render_success(
                    "Conversion complete! Use the chapter list on the left "
                    "to navigate between sections."
                )
                st.rerun()

    elif st.session_state.processing_done:
        # PDF was cleared after processing — show a soft warning
        # but keep the results displayed (session state still has them)
        UIComponents.render_info(
            "The uploaded file has been cleared. "
            "Previous audio is still available below."
        )

    else:
        # No file, no previous results — show welcome state
        st.divider()
        st.markdown(
            """
            ### How it works

            1. **Upload** a PDF — text-based or scanned
            2. **Configure** language, speed, and options in the sidebar
            3. **Click Convert** — the app extracts text, removes headers
               and footers, splits into chapters, and generates audio
            4. **Listen** using the chapter-by-chapter audio player

            **Supported languages:** English, Spanish, French, German,
            Italian, Russian, Chinese, Japanese, Arabic, Hindi,
            Portuguese, Dutch.

            **TTS engines:**
            - **Free (gTTS):** No setup required. Functional voice quality.
            - **Premium (OpenAI):** Set `OPENAI_API_KEY` environment variable.
              Natural-sounding voice. ~$0.015 per 1,000 characters.
            """
        )

    # ---- Results display ----
    if st.session_state.processing_done:
        st.divider()
        render_results()


# ================================================================== #
# ENTRY POINT                                                          #
# ================================================================== #

if __name__ == "__main__":
    main()