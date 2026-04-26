"""
streamlit_app.py
================
Entry point and pipeline orchestrator.

Pipeline stages:
  1.  Validate PDF
  2.  Detect extraction mode (text vs OCR)
  3.  PRE-SCAN full PDF for chapter boundaries (prevention layer)
  4.  Apply chapter-aware page range (auto-extend if cutting a chapter)
  5.  Show pre-processing summary to user
  6.  Extract blocks from final page range
  7.  Auto-detect language
  8.  Build document structure (chapters, headings)
  9.  Translate chapters if requested
  10. Compute page ranges per chapter
  11. Post-hoc: flag and remove incomplete pages (fingerprint verification)
  12. Post-hoc: flag and remove incomplete chapters if user opted in
  13. Generate audio per chapter (live queue with per-chapter status)
  14. Store all results in session state
"""

import logging
import traceback
from collections import OrderedDict

import streamlit as st

from config import Config
from text_cleaner import TextCleaner
import translator as Translator
from pdf_processor import PDFProcessor
from text_analyzer import TextAnalyzer
from audio_generator import AudioGenerator
from ui_components import UIComponents

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format=Config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


# ================================================================== #
# SESSION STATE                                                        #
# ================================================================== #

def initialize_session_state() -> None:
    defaults = {
        "pdf_bytes":               None,
        "audio_data":              {},
        "document_structure":      None,
        "extraction_result":       None,
        "current_chapter":         None,
        "bookmarks":               [],
        "processing_done":         False,
        "settings":                {},
        "detected_lang":           None,
        "translation_engine_used": None,
        "chapter_status":          {},
        "page_ranges":             {},
        "incomplete_pages":        set(),
        "incomplete_chapters":     set(),
        "pre_scan_warnings":       [],
        "original_end_page":       None,
        "extended_end_page":       None,
        "chapters_found_prescan":  [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_pipeline_state() -> None:
    """Reset all pipeline outputs before a new run."""
    st.session_state.processing_done         = False
    st.session_state.audio_data              = {}
    st.session_state.document_structure      = None
    st.session_state.extraction_result       = None
    st.session_state.current_chapter         = None
    st.session_state.bookmarks               = []
    # NOTE: detected_lang is NOT reset here — it persists from the pre-scan
    # so the sidebar language selector keeps showing the detected language.
    # It only resets when a new file is uploaded.
    st.session_state.translation_engine_used = None
    st.session_state.chapter_status          = {}
    st.session_state.page_ranges             = {}
    st.session_state.incomplete_pages        = set()
    st.session_state.chapters_for_fingerprint = None
    st.session_state.incomplete_chapters     = set()
    st.session_state.pre_scan_warnings       = []
    st.session_state.original_end_page       = None
    st.session_state.extended_end_page       = None


# ================================================================== #
# PIPELINE                                                             #
# ================================================================== #

def run_pipeline(pdf_bytes: bytes, settings: dict) -> None:
    """
    Full PDF → Audio pipeline. All results stored in session state.
    """
    lang_code                  = settings["lang_code"]
    tts_engine                 = settings["tts_engine"]
    rate                       = settings["rate"]
    pitch                      = settings["pitch"]
    start_page                 = settings["start_page"]
    end_page                   = settings["end_page"]
    remove_hf                  = settings["remove_headers_footers"]
    remove_incomplete_pages    = settings.get("remove_incomplete_pages",    True)
    remove_incomplete_chapters = settings.get("remove_incomplete_chapters", False)
    translate                  = settings.get("translate",           False)
    target_lang                = settings.get("target_lang",         lang_code)
    translation_engine         = settings.get("translation_engine",  "auto")

    # ------------------------------------------------------------------ #
    # STAGE 1: Validate                                                    #
    # ------------------------------------------------------------------ #
    with st.spinner("Validating PDF…"):
        valid, err = PDFProcessor.validate_pdf(pdf_bytes)
        if not valid:
            UIComponents.render_error(err)
            return

    # ------------------------------------------------------------------ #
    # STAGE 2: Detect extraction mode                                      #
    # ------------------------------------------------------------------ #
    with st.spinner("Analysing PDF structure…"):
        configured_mode = Config.validate_extraction_mode()
        detected_mode   = TextCleaner.detect_pdf_mode(pdf_bytes) if configured_mode == "auto" else configured_mode
        logger.info(f"Extraction mode: {detected_mode}")

    page_count   = PDFProcessor.get_page_count(pdf_bytes)
    resolved_end = end_page if end_page > 1 else page_count

    # ------------------------------------------------------------------ #
    # STAGE 3: Pre-scan for chapter boundaries (prevention)               #
    # ------------------------------------------------------------------ #
    with st.spinner("Scanning document structure for chapter boundaries…"):
        heading_pages = PDFProcessor.get_heading_pages(pdf_bytes)

        recommended_end, warnings = TextAnalyzer.get_chapter_aware_page_range(
            heading_pages=heading_pages,
            requested_start=start_page,
            requested_end=resolved_end,
            total_pages=page_count,
        )

        st.session_state.pre_scan_warnings      = warnings
        st.session_state.original_end_page      = resolved_end
        st.session_state.extended_end_page      = recommended_end
        st.session_state.chapters_found_prescan = []

        # Build chapter list from pre-scan for the summary panel
        if heading_pages:
            from collections import defaultdict as _dd
            seen: dict = {}
            for h in sorted(heading_pages, key=lambda x: x["page_num"]):
                pn = h["page_num"]
                if pn not in seen or h["font_size"] > seen[pn]["font_size"]:
                    seen[pn] = h
            unique = sorted(seen.values(), key=lambda x: x["page_num"])
            for i, h in enumerate(unique):
                ch_start = h["page_num"] + 1
                ch_end   = unique[i + 1]["page_num"] if i + 1 < len(unique) else page_count
                st.session_state.chapters_found_prescan.append({
                    "title": h["text"],
                    "start": ch_start,
                    "end":   ch_end,
                })

        if warnings:
            for w in warnings:
                if w.get("extended"):
                    logger.info(
                        f"Prevention: auto-extended end page "
                        f"{resolved_end} → {recommended_end} "
                        f"for chapter '{w['chapter']}'"
                    )

        final_end = recommended_end

    # ------------------------------------------------------------------ #
    # STAGE 4: Extract blocks                                              #
    # ------------------------------------------------------------------ #
    with st.spinner(
        f"Extracting text from pages {start_page}–{final_end} "
        f"({'text mode' if detected_mode == 'text' else 'OCR mode'})…"
    ):
        try:
            extraction_result = PDFProcessor.extract(
                pdf_bytes=pdf_bytes,
                lang_code=lang_code,
                start_page=start_page,
                end_page=final_end,
                mode=detected_mode,
            )
        except Exception as e:
            UIComponents.render_error(
                f"Text extraction failed: {e}\n\n"
                "If this is a scanned PDF, ensure packages.txt includes "
                "tesseract-ocr and the correct language is selected."
            )
            logger.error(traceback.format_exc())
            return

    blocks    = extraction_result["blocks"]
    mode_used = extraction_result["mode_used"]
    logger.info(f"[STAGE 4] Extraction complete: {len(blocks)} blocks, mode={mode_used}")
    if blocks:
        pages_with_blocks = len(set(b.get("page_num",0) for b in blocks))
        font_sizes = sorted(set(round(b.get("font_size",0),1) for b in blocks if b.get("font_size",0)>0))
        logger.info(f"[STAGE 4] Pages with blocks: {pages_with_blocks}, font sizes: {font_sizes}")

    if not blocks:
        UIComponents.render_error(
            "No text could be extracted from this PDF. "
            "If it is scanned, ensure Tesseract OCR is installed."
        )
        return

    # ------------------------------------------------------------------ #
    # STAGE 5: Auto-detect language                                        #
    # ------------------------------------------------------------------ #
    if Config.AUTO_DETECT_LANGUAGE and blocks:
        sample_text   = " ".join(
            b.get("text", "") for b in blocks[:50] if b.get("text", "").strip()
        )
        detected_lang = Translator.detect_language(sample_text)
        st.session_state.detected_lang = detected_lang
        logger.info(f"[STAGE 5] Auto-detected language: '{detected_lang}', sidebar was: '{lang_code}'")
        if detected_lang != lang_code:
            logger.info(f"[STAGE 5] Overriding lang_code: '{lang_code}' → '{detected_lang}'")
            lang_code = detected_lang

    # ------------------------------------------------------------------ #
    # STAGE 6: Build document structure                                    #
    # ------------------------------------------------------------------ #
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
    logger.info(f"[STAGE 6] Chapters built: {len(chapters)}")
    for ch_title, ch_text in chapters.items():
        logger.info(f"[STAGE 6]   '{ch_title[:60]}': {len(ch_text.split())} words, {len(ch_text)} chars")

    if not chapters:
        UIComponents.render_error(
            "Document structure could not be determined. "
            "The PDF may contain only images or non-standard encoding."
        )
        return

    logger.info(f"Chapters detected: {list(chapters.keys())}")

    # ------------------------------------------------------------------ #
    # STAGE 7: Translate if requested                                      #
    # ------------------------------------------------------------------ #
    if translate and target_lang and target_lang != lang_code:
        trans_config = Config.get_translation_config()
        if translation_engine != "auto":
            trans_config["primary_engine"]  = translation_engine
            trans_config["fallback_engine"] = "none"

        src_label = Config.SUPPORTED_LANGUAGES.get(lang_code,   {}).get("label", lang_code)
        tgt_label = Config.SUPPORTED_LANGUAGES.get(target_lang, {}).get("label", target_lang)

        with st.spinner(f"Translating {len(chapters)} chapters ({src_label} → {tgt_label})…"):
            translated   = OrderedDict()
            engine_used  = "original"

            for ch_title, ch_text in chapters.items():
                if not ch_text.strip():
                    translated[ch_title] = ch_text
                    continue
                logger.info(f"[STAGE 7] Translating '{ch_title[:50]}': {len(ch_text)} chars, {len(ch_text.split())} words")
                try:
                    result_text, engine_used = Translator.translate_with_fallback(
                        text=ch_text,
                        source_lang=lang_code,
                        target_lang=target_lang,
                        config=trans_config,
                    )
                    logger.info(f"[STAGE 7]   Result: {len(result_text.split())} words via {engine_used}")
                    if len(result_text.strip()) < 10:
                        logger.error(f"[STAGE 7]   CRITICAL: Translation returned nearly empty result! Keeping original.")
                        result_text = ch_text
                    translated[ch_title] = result_text
                except Exception as e:
                    logger.warning(f"[STAGE 7] Translation failed for '{ch_title}': {e} — keeping original")
                    translated[ch_title] = ch_text

            # Save pre-translation chapters for fingerprint verification.
            # Fingerprints are extracted from the ORIGINAL language blocks,
            # so they must be compared against the original language text.
            # Comparing against translated text would always fail.
            st.session_state.chapters_for_fingerprint = dict(chapters)

            chapters                               = translated
            document_structure["chapters"]         = chapters
            st.session_state.translation_engine_used = engine_used

            # Only switch lang_code to target if translation actually succeeded
            # (not "original" which means it failed or returned unchanged text)
            if "original" not in engine_used:
                lang_code = target_lang
                logger.info(f"Translation complete via {engine_used}: {src_label} → {tgt_label}")
            else:
                logger.warning(
                    f"Translation returned original text ({src_label} → {tgt_label}). "
                    f"Keeping lang_code={lang_code} for audio generation."
                )

    # ------------------------------------------------------------------ #
    # STAGE 8: Compute page ranges                                         #
    # ------------------------------------------------------------------ #
    page_ranges = TextAnalyzer.get_page_range_per_chapter(blocks, chapters)
    st.session_state.page_ranges = page_ranges
    logger.info(f"[STAGE 8] Page ranges: {page_ranges}")

    # ------------------------------------------------------------------ #
    # STAGE 9: Post-hoc incomplete page detection and removal              #
    # ------------------------------------------------------------------ #
    incomplete_pages    = set()
    incomplete_chapters = set()

    if remove_incomplete_pages:
        with st.spinner("Verifying page completeness via word fingerprinting…"):
            # Use pre-translation chapters for fingerprint matching.
            # Fingerprints come from original-language blocks, so they must
            # be compared against original-language text, not translated text.
            chapters_for_fp = st.session_state.get("chapters_for_fingerprint") or chapters
            logger.info(f"[STAGE 9] Running incomplete page detection on {len(chapters_for_fp)} chapters...")
            incomplete_pages = TextAnalyzer.flag_incomplete_pages(blocks, chapters_for_fp)
            st.session_state.incomplete_pages = incomplete_pages
            logger.info(f"[STAGE 9] Incomplete pages flagged: {[p+1 for p in sorted(incomplete_pages)]}")

            if incomplete_pages:
                # Only remove incomplete pages if we won't lose most content
                total_pages_in_blocks = len(set(b.get("page_num",0) for b in blocks))
                pct_flagged = len(incomplete_pages) / max(total_pages_in_blocks, 1)
                if pct_flagged > 0.5:
                    # More than 50% of pages flagged = fingerprint unreliable
                    # (likely single-chapter doc where last page = last page of doc)
                    logger.warning(
                        f"[STAGE 9] SAFETY: {pct_flagged:.0%} of pages flagged "
                        f"({len(incomplete_pages)}/{total_pages_in_blocks}) — "
                        f"likely false positives. Skipping removal to preserve content."
                    )
                    st.session_state.incomplete_pages = set()
                    incomplete_pages = set()
                else:
                    clean_blocks = [
                        b for b in blocks
                        if b.get("page_num") not in incomplete_pages
                    ]
                    rebuilt = TextAnalyzer.build_document_structure(
                        blocks=clean_blocks,
                        lang_code=lang_code,
                        remove_headers=remove_hf,
                    )
                    # Safety: only use rebuilt if it has substantial content
                    rebuilt_text = " ".join(rebuilt["chapters"].values())
                    original_text = " ".join(chapters.values())
                    pct_kept = len(rebuilt_text) / max(len(original_text), 1)
                    logger.info(f"[STAGE 9] Rebuild: original={len(original_text.split())} words, rebuilt={len(rebuilt_text.split())} words ({pct_kept:.0%} kept)")
                    if len(rebuilt_text) > len(original_text) * 0.3:
                        chapters                       = rebuilt["chapters"]
                        document_structure["chapters"] = chapters
                        logger.info(f"[STAGE 9] Removed content from {len(incomplete_pages)} incomplete pages. New chapters: {len(chapters)}")
                        for t, tx in chapters.items():
                            logger.info(f"[STAGE 9]   '{t[:50]}': {len(tx.split())} words")
                    else:
                        logger.warning(
                            f"[STAGE 9] SAFETY: Rebuild would lose {1-pct_kept:.0%} of content — skipping removal."
                        )
                        st.session_state.incomplete_pages = set()
                        incomplete_pages = set()

    # ------------------------------------------------------------------ #
    # STAGE 10: Post-hoc incomplete chapter detection and removal          #
    # ------------------------------------------------------------------ #
    if remove_incomplete_chapters:
        with st.spinner("Verifying chapter completeness…"):
            logger.info(f"[STAGE 10] Running incomplete chapter detection...")
            chapters_for_fp = st.session_state.get("chapters_for_fingerprint") or chapters
            incomplete_chapters = TextAnalyzer.flag_incomplete_chapters(blocks, chapters_for_fp)
            st.session_state.incomplete_chapters = incomplete_chapters
            logger.info(f"[STAGE 10] Incomplete chapters flagged: {list(incomplete_chapters)}")

            if incomplete_chapters:
                # Only remove if it won't wipe everything
                if len(incomplete_chapters) >= len(chapters):
                    logger.warning(
                        "Incomplete chapter detection would remove all chapters — skipping."
                    )
                    st.session_state.incomplete_chapters = set()
                    incomplete_chapters = set()
                else:
                    before_count = len(chapters)
                    chapters     = OrderedDict(
                        (t, v) for t, v in chapters.items()
                        if t not in incomplete_chapters
                    )
                    document_structure["chapters"] = chapters
                    logger.info(f"Removed {before_count - len(chapters)} incomplete chapters.")

    if not chapters:
        UIComponents.render_error(
            "All chapters were removed by completeness verification. "
            "This usually means the PDF was significantly truncated or "
            "the page range selection did not capture full chapter content. "
            "Try disabling 'Remove incomplete chapters' or extending the page range."
        )
        return

    # ------------------------------------------------------------------ #
    # STAGE 11: Generate audio (live chapter queue)                        #
    # ------------------------------------------------------------------ #
    audio_data:     dict[str, bytes] = {}
    chapter_titles = list(chapters.keys())
    total_chapters = len(chapter_titles)
    total_words = sum(len(t.split()) for t in chapters.values())
    logger.info(f"[STAGE 11] Starting audio generation: {total_chapters} chapters, {total_words} total words, lang={lang_code}, engine={tts_engine}")
    for t, tx in chapters.items():
        logger.info(f"[STAGE 11]   '{t[:50]}': {len(tx.split())} words")

    # Initialise chapter_status for all chapters
    chapter_status: dict[str, str] = {
        title: "queued" for title in chapter_titles
    }
    # Mark removed chapters
    for title in st.session_state.incomplete_chapters:
        if title not in chapter_status:
            chapter_status[title] = "removed"
    st.session_state.chapter_status = chapter_status

    # Live queue container — updates on each rerun
    queue_container = st.empty()

    progress_bar = st.progress(0, text="Generating audio…")

    for i, chapter_title in enumerate(chapter_titles):
        chapter_text = chapters[chapter_title]

        if not chapter_text.strip():
            chapter_status[chapter_title] = "removed"
            st.session_state.chapter_status = chapter_status
            continue

        # Update status to processing
        chapter_status[chapter_title] = "processing"
        st.session_state.chapter_status = chapter_status

        # Refresh queue display
        with queue_container.container():
            UIComponents.render_chapter_queue(
                chapters=chapters,
                audio_data=audio_data,
                chapter_status=chapter_status,
                page_ranges=page_ranges,
            )

        progress_bar.progress(i / total_chapters, text=f"Generating: '{chapter_title}' ({i+1}/{total_chapters})…")

        try:
            audio_bytes = AudioGenerator.generate_audio(
                text=chapter_text,
                lang_code=lang_code,
                rate=rate,
                pitch=pitch,
                engine=tts_engine,
            )
            audio_data[chapter_title]          = audio_bytes
            chapter_status[chapter_title]      = "done"
            st.session_state.chapter_status    = chapter_status
            st.session_state.audio_data        = audio_data

            logger.info(
                f"Audio OK: '{chapter_title}' — "
                f"{AudioGenerator.get_file_size_mb(audio_bytes):.2f} MB"
            )

        except Exception as e:
            logger.error(f"Audio generation failed for '{chapter_title}': {e}")
            audio_data[chapter_title]       = AudioGenerator._generate_silence_bytes()
            chapter_status[chapter_title]   = "failed"
            st.session_state.chapter_status = chapter_status

        # Refresh queue after each chapter completes
        with queue_container.container():
            UIComponents.render_chapter_queue(
                chapters=chapters,
                audio_data=audio_data,
                chapter_status=chapter_status,
                page_ranges=page_ranges,
            )

    progress_bar.progress(1.0, text="Audio generation complete.")

    if not audio_data or all(
        v == AudioGenerator._generate_silence_bytes() for v in audio_data.values()
    ):
        UIComponents.render_error(
            "Audio generation failed for all chapters. "
            "Check your internet connection (gTTS requires network). "
            "Try switching to a different TTS engine."
        )
        return

    # ------------------------------------------------------------------ #
    # STAGE 12: Store results                                              #
    # ------------------------------------------------------------------ #
    st.session_state.audio_data         = audio_data
    st.session_state.document_structure = document_structure
    st.session_state.extraction_result  = extraction_result
    st.session_state.pdf_bytes          = pdf_bytes
    # Reset detected_lang when new file is loaded so pre-scan runs fresh
    st.session_state.detected_lang      = None
    # Also clear the widget state so the selector defaults correctly
    if "lang_selector_value" in st.session_state:
        del st.session_state["lang_selector_value"]
    st.session_state.settings           = settings
    st.session_state.chapter_status     = chapter_status
    st.session_state.processing_done    = True

    if st.session_state.current_chapter not in audio_data:
        st.session_state.current_chapter = chapter_titles[0]

    logger.info(
        f"Pipeline complete: {len(audio_data)} chapters, "
        f"mode={mode_used}, engine={tts_engine}, "
        f"incomplete_pages={len(incomplete_pages)}, "
        f"incomplete_chapters={len(incomplete_chapters)}."
    )


# ================================================================== #
# RESULTS DISPLAY                                                      #
# ================================================================== #

def render_results() -> None:
    audio_data         = st.session_state.audio_data
    document_structure = st.session_state.document_structure
    extraction_result  = st.session_state.extraction_result
    pdf_bytes          = st.session_state.pdf_bytes
    settings           = st.session_state.settings
    current_chapter    = st.session_state.current_chapter
    chapter_status     = st.session_state.chapter_status
    page_ranges        = st.session_state.page_ranges
    incomplete_pages   = st.session_state.incomplete_pages
    incomplete_chaps   = st.session_state.incomplete_chapters

    chapters   = document_structure["chapters"]
    mode_used  = extraction_result["mode_used"]
    page_count = extraction_result["page_count"]
    metadata   = extraction_result["metadata"]

    if current_chapter not in audio_data:
        current_chapter = list(audio_data.keys())[0]
        st.session_state.current_chapter = current_chapter

    # Status banners
    UIComponents.render_processing_status(mode_used, page_count)

    if st.session_state.get("detected_lang"):
        UIComponents.render_detected_language_banner(
            detected_lang=st.session_state.detected_lang,
            selected_lang=settings.get("lang_code", "en"),
        )

    engine_used_val = st.session_state.get("translation_engine_used", "")
    if (settings.get("translate")
            and engine_used_val
            and "original" not in engine_used_val
            and settings.get("target_lang") != settings.get("lang_code")):
        UIComponents.render_translation_status(
            chapters_translated=len(chapters),
            total_chapters=len(chapters),
            source_lang=settings.get("lang_code", "en"),
            target_lang=settings.get("target_lang", "en"),
            engine_used=engine_used_val,
        )

    # Page range extension banner
    orig = st.session_state.get("original_end_page")
    ext  = st.session_state.get("extended_end_page")
    warn = st.session_state.get("pre_scan_warnings", [])
    if orig and ext and ext > orig and warn:
        UIComponents.render_page_range_warning(
            original_end=orig,
            extended_end=ext,
            chapters_cut=warn,
        )

    # Completeness report
    UIComponents.render_completeness_report(
        incomplete_pages=incomplete_pages,
        incomplete_chapters=incomplete_chaps,
        total_pages=page_count,
        total_chapters=len(chapters),
    )

    # Metadata
    UIComponents.render_metadata(metadata)

    # PDF preview
    UIComponents.render_pdf_preview(pdf_bytes)

    st.divider()

    # Cost estimate (OpenAI only)
    UIComponents.render_cost_estimate(chapters, settings.get("tts_engine", "gtts"))

    # Main layout: TOC left, player right
    col_toc, col_player = st.columns([1, 3])

    with col_toc:
        UIComponents.render_table_of_contents(
            chapters=chapters,
            current_chapter=current_chapter,
            page_ranges=page_ranges,
            chapter_status=chapter_status,
        )
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

    # Analysis panel
    if settings.get("show_analysis", False):
        st.divider()
        _render_analysis_section(document_structure, settings["lang_code"])


def _render_analysis_section(document_structure: dict, lang_code: str) -> None:
    chapters     = document_structure["chapters"]
    full_text    = TextAnalyzer.get_full_text(chapters)
    word_count   = TextAnalyzer.get_word_count(full_text)
    reading_time = TextAnalyzer.get_reading_time_estimate(full_text)

    with st.spinner("Running document analysis…"):
        keywords             = TextAnalyzer.extract_keywords(full_text, lang_code)
        characters           = TextAnalyzer.detect_character_names(full_text, lang_code)
        summary              = TextAnalyzer.summarize_text(full_text, lang_code)
        sentiment            = TextAnalyzer.sentiment_analysis(full_text, lang_code)
        readability          = TextAnalyzer.compute_readability(full_text)
        text_stats           = TextAnalyzer.compute_text_stats(full_text, chapters)
        content_type         = TextAnalyzer.detect_content_type(full_text, chapters)
        topic_density        = TextAnalyzer.compute_topic_density(full_text, keywords)
        chapter_complexity   = TextAnalyzer.detect_language_complexity_by_chapter(chapters)
        lexical_diversity    = TextAnalyzer.compute_lexical_diversity(full_text)
        sentence_complexity  = TextAnalyzer.compute_sentence_complexity(full_text)

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
        lexical_diversity=lexical_diversity,
        sentence_complexity=sentence_complexity,
    )


# ================================================================== #
# MAIN                                                                 #
# ================================================================== #

def main() -> None:
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()

    # Sidebar settings
    max_pages = 1
    if st.session_state.pdf_bytes:
        try:
            max_pages = PDFProcessor.get_page_count(st.session_state.pdf_bytes)
        except Exception:
            max_pages = 1

    # Pre-detect language from PDF before sidebar renders so the
    # language selector defaults to the correct language on first load.
    # This is a fast peek — just first 500 chars of text from page 2.
    if st.session_state.pdf_bytes and not st.session_state.get("detected_lang"):
        try:
            import fitz as _fitz
            _doc = _fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
            _sample = ""
            for _pn in range(min(3, len(_doc))):
                _sample += _doc[_pn].get_text("text")[:300]
            _doc.close()
            if _sample.strip():
                _pre_detected = Translator.detect_language(_sample)
                if _pre_detected and _pre_detected != "en":
                    st.session_state.detected_lang = _pre_detected
                    logger.info(f"Pre-scan language detection: '{_pre_detected}'")
        except Exception as _e:
            logger.debug(f"Pre-scan language detection failed: {_e}")

    settings = UIComponents.render_sidebar_settings(max_pages=max_pages)

    # Header
    st.title(f"🎧 {Config.APP_NAME}")
    st.caption(Config.DESCRIPTION)

    # File input
    st.subheader("Upload PDF")
    file_source = st.radio(
        "File source",
        options=["Upload file", "Local file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    pdf_bytes = None

    if file_source == "Upload file":
        pdf_file = UIComponents.file_uploader_widget()
        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
    else:
        local_path = UIComponents.local_file_selector(folder_path=".")
        if local_path:
            try:
                with open(local_path, "rb") as f:
                    pdf_bytes = f.read()
            except OSError as e:
                UIComponents.render_error(f"Could not read file: {e}")

    # Pre-processing summary (shown as soon as PDF is loaded)
    if pdf_bytes is not None:
        size_mb    = len(pdf_bytes) / (1024 * 1024)
        page_count = PDFProcessor.get_page_count(pdf_bytes)
        start_pg   = settings["start_page"]
        end_pg     = settings["end_page"] if settings["end_page"] > 1 else page_count

        # Quick pre-scan for summary (cached)
        heading_pages = PDFProcessor.get_heading_pages(pdf_bytes)
        recommended_end, warnings = TextAnalyzer.get_chapter_aware_page_range(
            heading_pages=heading_pages,
            requested_start=start_pg,
            requested_end=end_pg,
            total_pages=page_count,
        )

        # Build chapter list for summary
        chapters_for_summary = []
        if heading_pages:
            seen_pg: dict = {}
            for h in sorted(heading_pages, key=lambda x: x["page_num"]):
                pn = h["page_num"]
                if pn not in seen_pg or h["font_size"] > seen_pg[pn]["font_size"]:
                    seen_pg[pn] = h
            unique = sorted(seen_pg.values(), key=lambda x: x["page_num"])
            for i, h in enumerate(unique):
                ch_start = h["page_num"] + 1
                ch_end   = unique[i + 1]["page_num"] if i + 1 < len(unique) else page_count
                chapters_for_summary.append({
                    "title": h["text"], "start": ch_start, "end": ch_end
                })

        # Estimate processing time: ~30 seconds per chapter for gTTS
        est_chapters  = max(len(chapters_for_summary), 1)
        est_minutes   = est_chapters * 0.5  # rough: 30s per chapter
        selected_pages = end_pg - start_pg + 1
        # Estimate audio: ~150 wpm, ~1 min audio per 150 words, ~300 words/page
        est_words     = selected_pages * 300
        est_audio_hrs = (est_words / 150) / 60

        st.caption(f"File: {size_mb:.2f} MB · {page_count} pages")

        UIComponents.render_pre_processing_summary(
            total_pages=page_count,
            selected_pages=selected_pages,
            chapters_found=chapters_for_summary,
            est_minutes=est_minutes,
            est_audio_hrs=est_audio_hrs,
            warnings=warnings,
            extended_end=recommended_end,
            original_end=end_pg,
        )

        st.divider()

        # Convert button
        if st.button("🎙️ Convert to Audio", type="primary", use_container_width=True):
            _reset_pipeline_state()
            run_pipeline(pdf_bytes=pdf_bytes, settings=settings)
            if st.session_state.processing_done:
                UIComponents.render_success(
                    "Conversion complete! Navigate chapters using the list on the left."
                )
                st.rerun()

    elif st.session_state.processing_done:
        UIComponents.render_info(
            "The uploaded file has been cleared. Previous audio is still available below."
        )
    else:
        st.divider()
        st.markdown("""
### How it works

1. **Upload** a PDF — text-based or scanned
2. **Review** the pre-processing summary — total pages, detected chapters,
   estimated audio length, and any chapter boundary warnings
3. **Configure** language, speed, translation, and completeness options in the sidebar
4. **Click Convert** — the app extracts text, removes headers and footers,
   verifies completeness via word fingerprinting, and generates audio chapter by chapter
5. **Download** each chapter individually or the full book as one MP3

**Completeness guarantee:** before generating audio, the app verifies that the
last words of each chapter page were fully captured. Incomplete pages are removed.
If a chapter ends mid-word, it can be excluded entirely (opt-in).

**Translation:** upload an English article and generate Spanish audio, or any
combination of the 10 supported languages. Uses Google Translate with
LibreTranslate as fallback — both free.

**Supported languages:** English, Spanish, French, German, Italian,
Russian, Chinese, Japanese, Arabic, Hindi, Portuguese, Dutch.
        """)

    # Results display
    if st.session_state.processing_done:
        st.divider()
        render_results()


if __name__ == "__main__":
    main()