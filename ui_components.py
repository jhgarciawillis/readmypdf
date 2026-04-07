"""
ui_components.py
================
All Streamlit UI rendering. Pure presentation layer — takes data in,
renders widgets. Contains zero business logic.

Responsibilities:
  1. File input widgets (upload + local selector)
  2. Sidebar settings panel (language, engine, rate/pitch, page range, toggles)
  3. Document display (TOC, metadata, PDF preview)
  4. Audio player (native st.audio — no giant base64 inline HTML)
  5. Bookmarks (add, display, navigate)
  6. Analysis panel (keywords, characters, summary, sentiment, reading time)
  7. Status and feedback messages

Core design decisions:
  - No calls to PDFProcessor, TextAnalyzer, AudioGenerator, or TextCleaner.
    UI components receive pre-computed data and render it. All computation
    happens in streamlit_app.py.
  - Session state is read and written directly here for UI interactions
    (bookmark additions, chapter selection). This is the correct pattern
    for Streamlit — state mutations that are purely UI-driven belong in
    the component, not the orchestrator.
  - st.audio() replaces the base64-inline-HTML audio player from the
    original. The original embedded full MP3 bytes as a base64 string
    directly into the HTML DOM — for large files this caused memory issues
    and very slow page loads. st.audio() handles this correctly.
  - All methods are static. No instance state.
"""

import os
import io
import re
import base64
import logging
from collections import OrderedDict
from typing import Optional

import streamlit as st

from config import Config

logger = logging.getLogger(__name__)


class UIComponents:

    # ================================================================== #
    # SECTION 1 — FILE INPUT                                              #
    # ================================================================== #

    @staticmethod
    def file_uploader_widget() -> Optional[object]:
        """
        Render a PDF file uploader widget.

        Returns:
          Streamlit UploadedFile object if a file has been uploaded,
          None otherwise.

        The caller (streamlit_app.py) is responsible for calling
        .read() on the returned object exactly once and passing
        the resulting bytes everywhere else.
        """
        return st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help=(
                f"Maximum file size: {Config.MAX_PDF_SIZE_MB} MB. "
                "Text-based PDFs are processed in seconds; "
                "scanned PDFs require OCR and may take longer."
            ),
            label_visibility="collapsed",
        )

    @staticmethod
    def local_file_selector(folder_path: str = ".") -> Optional[str]:
        """
        Render a dropdown to select a PDF from a local folder.

        Guards against empty folders — shows an informational message
        instead of crashing with an empty selectbox.

        Args:
          folder_path: directory to scan for .pdf files

        Returns:
          Full path string to the selected PDF, or None if no PDFs found.
        """
        try:
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        except OSError:
            st.warning(f"Cannot read folder: {folder_path}")
            return None

        if not pdf_files:
            st.info("No PDF files found in the current folder.")
            return None

        selected = st.selectbox(
            "Select a PDF file",
            options=pdf_files,
            label_visibility="collapsed",
        )
        return os.path.join(folder_path, selected)

    # ================================================================== #
    # SECTION 2 — SETTINGS SIDEBAR                                        #
    # ================================================================== #

    @staticmethod
    def render_sidebar_settings(max_pages: int = 1) -> dict:
        """
        Render all sidebar settings controls and return them as a dict.

        This is the single method called by streamlit_app.py to get all
        user settings. It internally renders every sidebar control and
        packages the results.

        Args:
          max_pages: total page count of the current PDF (for page range
                     selector). Pass 1 if no PDF is loaded yet.

        Returns:
          dict: {
            "lang_code":             str,   # e.g. "en"
            "tts_engine":            str,   # "gtts" | "openai"
            "rate":                  float, # 0.5–2.0
            "pitch":                 float, # 0.5–2.0
            "start_page":            int,   # 1-indexed
            "end_page":              int,   # 1-indexed, inclusive
            "remove_headers_footers": bool,
            "show_analysis":         bool,
          }
        """
        with st.sidebar:
            st.title(Config.APP_NAME)
            st.caption(f"v{Config.VERSION}")
            st.divider()

            # Language
            st.subheader("Language")
            lang_code = UIComponents._language_selector()

            st.divider()

            # TTS Engine
            st.subheader("TTS Engine")
            tts_engine = UIComponents._tts_engine_selector()

            st.divider()

            # Audio settings
            st.subheader("Audio Settings")
            rate, pitch = UIComponents._rate_pitch_sliders()

            st.divider()

            # Page range
            st.subheader("Page Range")
            start_page, end_page = UIComponents._page_range_selector(max_pages)

            st.divider()

            # Feature toggles
            st.subheader("Options")
            remove_hf, show_analysis = UIComponents._feature_toggles()

        return {
            "lang_code":              lang_code,
            "tts_engine":             tts_engine,
            "rate":                   rate,
            "pitch":                  pitch,
            "start_page":             start_page,
            "end_page":               end_page,
            "remove_headers_footers": remove_hf,
            "show_analysis":          show_analysis,
        }

    @staticmethod
    def _language_selector() -> str:
        """
        Render a language selectbox. Returns the selected lang_code.
        """
        labels = Config.get_all_language_labels()   # { "en": "English", ... }
        label_list = list(labels.values())
        code_list  = list(labels.keys())

        selected_label = st.selectbox(
            "Document language",
            options=label_list,
            index=0,
            help="Select the language of the PDF for accurate OCR and TTS.",
        )
        selected_code = code_list[label_list.index(selected_label)]

        return selected_code

    @staticmethod
    def _tts_engine_selector() -> str:
        """
        Render the TTS engine selector. Returns "gtts" or "openai".

        Shows the OpenAI option only if the API key is configured.
        If OpenAI is selected but no key is available, warns and falls
        back to gTTS display.
        """
        openai_available = Config.is_openai_available()

        if openai_available:
            options = ["Free (gTTS)", "Premium (OpenAI)"]
            engine_map = {"Free (gTTS)": "gtts", "Premium (OpenAI)": "openai"}
            selected = st.radio(
                "TTS engine",
                options=options,
                index=0,
                help=(
                    "Free: Google Translate TTS — functional, robotic voice, no cost.\n\n"
                    "Premium: OpenAI tts-1 — natural voice, ~$0.015 per 1,000 characters."
                ),
            )
            engine = engine_map[selected]
            if engine == "openai":
                st.caption("✅ OpenAI API key detected.")
        else:
            st.radio(
                "TTS engine",
                options=["Free (gTTS)"],
                index=0,
                disabled=False,
            )
            engine = "gtts"
            st.caption(
                "Set `OPENAI_API_KEY` environment variable to enable "
                "the premium OpenAI TTS engine."
            )

        return engine

    @staticmethod
    def _rate_pitch_sliders() -> tuple[float, float]:
        """
        Render speech rate and pitch sliders.
        Returns (rate, pitch) as floats.
        """
        rate = st.slider(
            "Speech rate",
            min_value=0.5,
            max_value=2.0,
            value=Config.DEFAULT_RATE,
            step=0.1,
            format="%.1fx",
            help="1.0 = normal speed. 0.5 = half speed. 2.0 = double speed.",
        )
        pitch = st.slider(
            "Speech pitch",
            min_value=0.5,
            max_value=2.0,
            value=Config.DEFAULT_PITCH,
            step=0.1,
            format="%.1fx",
            help="1.0 = normal pitch. Values above/below shift the voice higher/lower.",
        )
        return rate, pitch

    @staticmethod
    def _page_range_selector(max_pages: int) -> tuple[int, int]:
        """
        Render start/end page number inputs.
        Returns (start_page, end_page) as 1-indexed ints.
        """
        if max_pages <= 1:
            # No PDF loaded yet — show disabled placeholder
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("Start page", value=1, disabled=True)
            with col2:
                st.number_input("End page", value=1, disabled=True)
            return 1, 1

        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input(
                "Start page",
                min_value=1,
                max_value=max_pages,
                value=1,
                step=1,
            )
        with col2:
            end_page = st.number_input(
                "End page",
                min_value=int(start_page),
                max_value=max_pages,
                value=max_pages,
                step=1,
            )
        return int(start_page), int(end_page)

    @staticmethod
    def _feature_toggles() -> tuple[bool, bool]:
        """
        Render feature toggle checkboxes.
        Returns (remove_headers_footers, show_analysis).
        """
        remove_hf = st.checkbox(
            "Remove headers & footers",
            value=True,
            help=(
                "Detects and removes running headers, footers, and page numbers "
                "before generating audio. Strongly recommended."
            ),
        )
        show_analysis = st.checkbox(
            "Show document analysis",
            value=False,
            help=(
                "After processing, display keywords, character names, "
                "a summary, and sentiment analysis. "
                "Requires a supported language with spaCy model installed."
            ),
        )
        return remove_hf, show_analysis

    # ================================================================== #
    # SECTION 3 — DOCUMENT DISPLAY                                        #
    # ================================================================== #

    @staticmethod
    def render_metadata(metadata: dict) -> None:
        """
        Display PDF metadata (title, author, page count) in a clean
        info box. Silently skips fields that are empty.

        Args:
          metadata: dict from PDFProcessor.get_metadata()
        """
        fields = []
        if metadata.get("title"):
            fields.append(f"**Title:** {metadata['title']}")
        if metadata.get("author"):
            fields.append(f"**Author:** {metadata['author']}")
        if metadata.get("subject"):
            fields.append(f"**Subject:** {metadata['subject']}")

        if fields:
            st.info("  \n".join(fields))

    @staticmethod
    def render_processing_status(mode_used: str, page_count: int) -> None:
        """
        Display a banner indicating which extraction mode was used.

        Args:
          mode_used:  "text" or "ocr"
          page_count: number of pages processed
        """
        if mode_used == "text":
            st.success(
                f"✅ **Text mode** — native text extraction "
                f"({page_count} pages processed). Fast and accurate."
            )
        else:
            st.warning(
                f"🔍 **OCR mode** — scanned PDF detected "
                f"({page_count} pages processed). "
                f"Quality depends on scan resolution. "
                f"Heading detection uses pattern matching (not font size)."
            )

    @staticmethod
    def render_table_of_contents(
        chapters:        OrderedDict,
        current_chapter: str,
    ) -> None:
        """
        Render the chapter list as a clickable table of contents.

        Clicking a chapter title sets st.session_state.current_chapter,
        which triggers a rerun that loads the correct audio in the player.

        The current chapter is displayed in bold. All others are buttons.

        Args:
          chapters:        OrderedDict[title → text] from build_document_structure()
          current_chapter: title of the currently active chapter
        """
        st.subheader("Chapters")

        if not chapters:
            st.caption("No chapters detected.")
            return

        for i, title in enumerate(chapters.keys()):
            # Truncate very long chapter titles for display
            display_title = title if len(title) <= 50 else title[:47] + "…"
            char_count    = len(chapters[title])
            caption       = f"{char_count:,} chars"

            if title == current_chapter:
                # Current chapter — shown as highlighted text, not a button
                st.markdown(
                    f"**▶ {display_title}**  \n"
                    f"<small style='color:gray'>{caption}</small>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"{display_title}",
                    key=f"toc_btn_{i}",
                    use_container_width=True,
                    help=f"{char_count:,} characters",
                ):
                    st.session_state.current_chapter = title
                    st.rerun()

    @staticmethod
    def render_pdf_preview(pdf_bytes: bytes) -> None:
        """
        Render a PDF preview inside a collapsible expander using an
        iframe with base64-encoded content.

        Lazy — only rendered when the user expands the section.
        Capped at a fixed height with scroll to avoid taking over the page.

        Args:
          pdf_bytes: raw PDF bytes
        """
        with st.expander("📄 Preview PDF", expanded=False):
            try:
                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_iframe = (
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                    f'width="100%" height="600px" type="application/pdf">'
                    f'</iframe>'
                )
                st.markdown(pdf_iframe, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not render PDF preview: {e}")

    # ================================================================== #
    # SECTION 4 — AUDIO PLAYER                                            #
    # ================================================================== #

    @staticmethod
    def render_audio_player(
        audio_bytes:     bytes,
        chapter_title:   str,
        chapter_index:   int,
        total_chapters:  int,
    ) -> None:
        """
        Render the audio player for the current chapter.

        Uses st.audio() — the native Streamlit audio widget — which
        handles byte streaming correctly without embedding giant base64
        strings inline in the DOM.

        Displays:
          - Chapter title
          - Native audio player with controls
          - Duration and file size metadata
          - Chapter X of N progress indicator

        Args:
          audio_bytes:    MP3 bytes for the current chapter
          chapter_title:  title of the current chapter
          chapter_index:  0-indexed position of current chapter
          total_chapters: total number of chapters
        """
        if not audio_bytes:
            st.warning("No audio available for this chapter.")
            return

        st.subheader(chapter_title)

        # Native audio player — no DOM bloat
        st.audio(audio_bytes, format="audio/mp3")

        # Audio metadata row
        col1, col2, col3 = st.columns(3)

        try:
            from audio_generator import AudioGenerator
            duration_sec = AudioGenerator.get_duration(audio_bytes)
            size_mb      = AudioGenerator.get_file_size_mb(audio_bytes)
            duration_str = AudioGenerator.format_duration(duration_sec)

            with col1:
                st.metric("Duration", duration_str)
            with col2:
                st.metric("File size", f"{size_mb:.2f} MB")
            with col3:
                st.metric("Chapter", f"{chapter_index + 1} of {total_chapters}")
        except Exception:
            # Non-critical — audio plays fine without these metrics
            pass

    @staticmethod
    def render_chapter_navigation(
        chapters:        OrderedDict,
        current_chapter: str,
    ) -> None:
        """
        Render Previous / Next chapter navigation buttons below the player.

        Clicking Previous/Next updates st.session_state.current_chapter
        and triggers a rerun.

        Args:
          chapters:        OrderedDict of all chapters
          current_chapter: currently active chapter title
        """
        titles = list(chapters.keys())
        if len(titles) <= 1:
            return

        current_idx = titles.index(current_chapter) if current_chapter in titles else 0

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_idx > 0:
                if st.button("◀ Previous", use_container_width=True):
                    st.session_state.current_chapter = titles[current_idx - 1]
                    st.rerun()

        with col3:
            if current_idx < len(titles) - 1:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.current_chapter = titles[current_idx + 1]
                    st.rerun()

    @staticmethod
    def render_progress(current_chapter: str, chapters: OrderedDict) -> None:
        """
        Render a chapter progress bar (X of N chapters).

        Args:
          current_chapter: currently active chapter title
          chapters:        OrderedDict of all chapters
        """
        titles = list(chapters.keys())
        if not titles:
            return

        current_idx = titles.index(current_chapter) if current_chapter in titles else 0
        progress    = (current_idx + 1) / len(titles)

        st.progress(progress)
        st.caption(
            f"Chapter {current_idx + 1} of {len(titles)}"
        )

    # ================================================================== #
    # SECTION 4B — DOWNLOAD BUTTONS                                       #
    # ================================================================== #

    @staticmethod
    def render_download_buttons(
        audio_data:      dict,
        current_chapter: str,
        chapters:        "OrderedDict",
    ) -> None:
        """
        Render download buttons for current chapter and full book.

        Chapter download: current chapter MP3 only.
        Full book download: all chapters stitched in order with 2s silence
        between chapters. Built on-demand from session state bytes —
        no temp files, no extra dependencies.

        Args:
          audio_data:      { chapter_title: mp3_bytes }
          current_chapter: title of the currently active chapter
          chapters:        OrderedDict of all chapters (for ordering)
        """
        st.markdown("**⬇️ Downloads**")
        col1, col2 = st.columns(2)

        # ── Chapter download ──────────────────────────────────────────
        with col1:
            chapter_audio = audio_data.get(current_chapter, b"")
            if chapter_audio:
                safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", current_chapter)[:40]
                st.download_button(
                    label="📥 This Chapter",
                    data=chapter_audio,
                    file_name=f"{safe_title}.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                    help=f"Download '{current_chapter}' as MP3",
                )
            else:
                st.button("📥 This Chapter", disabled=True, use_container_width=True)

        # ── Full book download ────────────────────────────────────────
        with col2:
            if len(audio_data) > 0:
                # Build full book bytes: chapters in order with 2s silence gap
                # 2 seconds of silence as minimal MP3 frames (~32 frames)
                silence_gap = bytes([
                    0xFF, 0xFB, 0x90, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                ]) * 32  # ~2 seconds

                # Stitch in chapter order
                buf = io.BytesIO()
                chapter_order = list(chapters.keys())
                for i, title in enumerate(chapter_order):
                    ch_audio = audio_data.get(title, b"")
                    if ch_audio:
                        buf.write(ch_audio)
                        if i < len(chapter_order) - 1:
                            buf.write(silence_gap)

                full_book_bytes = buf.getvalue()

                if full_book_bytes:
                    st.download_button(
                        label="📚 Full Book",
                        data=full_book_bytes,
                        file_name="full_book.mp3",
                        mime="audio/mpeg",
                        use_container_width=True,
                        help=f"Download all {len(audio_data)} chapters as one MP3",
                    )
                else:
                    st.button("📚 Full Book", disabled=True, use_container_width=True)
            else:
                st.button("📚 Full Book", disabled=True, use_container_width=True)

    # ================================================================== #
    # SECTION 5 — BOOKMARKS                                               #
    # ================================================================== #

    @staticmethod
    def render_bookmark_button(current_chapter: str) -> None:
        """
        Render an "Add Bookmark" button for the current chapter.

        Bookmarks are stored in st.session_state.bookmarks as a list of
        chapter title strings (position-level bookmarks are not feasible
        in Streamlit without a custom JS component, so we bookmark at the
        chapter level).

        Args:
          current_chapter: title of the chapter to bookmark
        """
        if st.button("🔖 Bookmark this chapter", use_container_width=True):
            if "bookmarks" not in st.session_state:
                st.session_state.bookmarks = []

            if current_chapter not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(current_chapter)
                st.success(f"Bookmarked: '{current_chapter}'")
            else:
                st.info("This chapter is already bookmarked.")

    @staticmethod
    def render_bookmarks_list(chapters: OrderedDict) -> None:
        """
        Display saved bookmarks with navigation buttons.

        Clicking "Go to" sets st.session_state.current_chapter to the
        bookmarked chapter and triggers a rerun. This is the correct
        implementation of what was a `pass` placeholder in the original.

        Args:
          chapters: OrderedDict of all chapters (used to verify bookmark
                    titles are still valid after reprocessing)
        """
        bookmarks = st.session_state.get("bookmarks", [])

        if not bookmarks:
            st.caption("No bookmarks yet.")
            return

        st.subheader("Bookmarks")

        valid_chapters = set(chapters.keys())
        stale_bookmarks = []

        for i, title in enumerate(bookmarks):
            if title not in valid_chapters:
                stale_bookmarks.append(title)
                continue

            col1, col2 = st.columns([3, 1])
            with col1:
                display = title if len(title) <= 40 else title[:37] + "…"
                st.write(f"🔖 {display}")
            with col2:
                if st.button("Go to", key=f"bookmark_goto_{i}"):
                    st.session_state.current_chapter = title
                    st.rerun()

        # Silently clean up stale bookmarks (chapter was removed after reprocessing)
        if stale_bookmarks:
            st.session_state.bookmarks = [
                b for b in bookmarks if b not in stale_bookmarks
            ]

    # ================================================================== #
    # SECTION 6 — ANALYSIS PANEL                                          #
    # ================================================================== #

    @staticmethod
    def render_analysis_panel(
        keywords:        list,
        characters:      list,
        summary:         str,
        sentiment:       dict,
        reading_time:    str,
        word_count:      int,
        readability:     dict  = None,
        text_stats:      dict  = None,
        content_type:    dict  = None,
        topic_density:   list  = None,
        chapter_complexity: list = None,
    ) -> None:
        """
        Rich document analysis panel with KPIs, readability scores,
        topic density, chapter complexity, and content type detection.
        """
        st.subheader("📊 Document Intelligence")

        # ── ROW 1: Core KPIs ──────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Words", f"{word_count:,}")
        with c2:
            st.metric("Est. Listening Time", reading_time)
        if text_stats:
            with c3:
                st.metric("Unique Words", f"{text_stats.get('unique_words', 0):,}")
            with c4:
                vr = text_stats.get("vocabulary_richness", 0)
                st.metric("Vocabulary Richness", f"{vr}%",
                          help="Unique words ÷ total words × 100. Higher = more diverse language.")

        st.divider()

        # ── ROW 2: Readability + Content Type ─────────────────────────
        col_read, col_type = st.columns([3, 2])

        with col_read:
            st.markdown("**📖 Readability**")
            if readability:
                fe  = readability.get("flesch_ease", 0)
                fk  = readability.get("flesch_kincaid", 0)
                gl  = readability.get("grade_label", "Unknown")
                el  = readability.get("ease_label", "Unknown")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Flesch Ease", f"{fe}/100",
                              help="0=hardest, 100=easiest. 60-70 is standard adult reading.")
                with r2:
                    st.metric("Grade Level", f"{fk}",
                              help="US school grade equivalent of the text complexity.")
                with r3:
                    st.metric("Reading Level", gl)

                # Visual ease bar
                bar_color = (
                    "🟢" if fe >= 70 else
                    "🟡" if fe >= 50 else
                    "🔴"
                )
                st.caption(f"{bar_color} {el} — {int(fe)}% ease score")

                if text_stats:
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("Avg Sentence Length",
                                  f"{text_stats.get('avg_sentence_length', 0)} words",
                                  help="Sentences >25 words are hard to follow in audio.")
                    with s2:
                        st.metric("Avg Word Length",
                                  f"{text_stats.get('avg_word_length', 0)} chars")
                    with s3:
                        st.metric("Total Sentences",
                                  f"{text_stats.get('total_sentences', 0):,}")
            else:
                st.caption("Readability data unavailable.")

        with col_type:
            st.markdown("**🔍 Content Type**")
            if content_type:
                ct   = content_type.get("type", "General")
                conf = content_type.get("confidence", 0)

                type_icons = {
                    "Academic":           "🎓",
                    "Report / Analysis":  "📈",
                    "Fiction / Narrative": "📚",
                    "News / Article":     "📰",
                    "Technical":          "⚙️",
                    "Legal":              "⚖️",
                    "General":            "📄",
                }
                icon = type_icons.get(ct, "📄")
                st.metric(f"{icon} Detected Type", ct)
                st.caption(f"Confidence: {conf}%")

                # Show signal breakdown
                all_scores = content_type.get("all_scores", {})
                if all_scores:
                    top = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    for label, score in top:
                        if score > 0:
                            st.caption(f"  • {label}: {score} signals")
            else:
                st.caption("Content type detection unavailable.")

        st.divider()

        # ── ROW 3: Summary + Sentiment ────────────────────────────────
        col_sum, col_sent = st.columns([3, 2])

        with col_sum:
            st.markdown("**📝 Summary**")
            if summary:
                st.write(summary)
                st.caption("Extractive — first 3 sentences of document body.")
            else:
                st.caption("Summary unavailable.")

        with col_sent:
            st.markdown("**🎭 Tone & Sentiment**")
            if sentiment:
                label = sentiment.get("label", "Unknown")
                conf  = sentiment.get("confidence", "low")
                pos   = sentiment.get("positive_count", 0)
                neg   = sentiment.get("negative_count", 0)
                total = pos + neg or 1

                color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                icon = color_map.get(label, "⚪")

                st.metric(f"{icon} Tone", label)
                st.caption(f"Confidence: {conf}")

                # Mini signal bar
                pos_pct = round(pos / total * 100)
                neg_pct = 100 - pos_pct
                st.caption(
                    f"🟢 {pos} positive  •  🔴 {neg} negative"
                )
            else:
                st.caption("Sentiment data unavailable.")

        st.divider()

        # ── ROW 4: Topic Density ──────────────────────────────────────
        if topic_density:
            st.markdown("**🔑 Topic Density** *(occurrences per 1,000 words)*")
            for item in topic_density[:12]:
                word    = item["word"]
                density = item["density_per_1k"]
                bar_pct = item["bar_pct"]
                count   = item["count"]
                # Simple inline bar using unicode blocks
                filled  = int(bar_pct / 5)  # max 20 chars
                bar     = "█" * filled + "░" * (20 - filled)
                st.markdown(
                    f"`{word:<18}` {bar} **{density}**/1k  *(×{count})*",
                    unsafe_allow_html=False,
                )
            st.divider()

        # ── ROW 5: Chapter Complexity ─────────────────────────────────
        if chapter_complexity:
            st.markdown("**📊 Complexity by Chapter** *(hardest → easiest to follow)*")
            st.caption(
                "Lower Flesch score = harder to follow while listening. "
                "Consider re-reading complex chapters after listening."
            )
            for item in chapter_complexity:
                title  = item["title"]
                ease   = item["flesch_ease"]
                label  = item["ease_label"]
                wc     = item["word_count"]
                icon   = "🔴" if ease < 50 else ("🟡" if ease < 70 else "🟢")
                short  = title if len(title) <= 40 else title[:37] + "…"
                st.markdown(
                    f"{icon} **{short}** — {label} ({ease}/100) · {wc:,} words"
                )

            st.divider()

        # ── ROW 6: Recurring Names ────────────────────────────────────
        if characters:
            st.markdown("**👥 Recurring Names / Entities**")
            st.write("  •  ".join(characters[:20]))
            st.divider()

        # ── ROW 7: Keywords ───────────────────────────────────────────
        if keywords:
            st.markdown("**💬 Top Keywords**")
            kw_parts = [f"**{w}** ({c})" for w, c in keywords[:15]]
            st.write("  •  ".join(kw_parts))

    # ================================================================== #
    # SECTION 7 — COST ESTIMATION                                         #
    # ================================================================== #

    @staticmethod
    def render_cost_estimate(
        chapters:   OrderedDict,
        tts_engine: str,
    ) -> None:
        """
        Display an estimated OpenAI TTS cost for the full document.
        Only rendered when tts_engine == "openai".

        Args:
          chapters:   OrderedDict of chapter_title → text
          tts_engine: "gtts" | "openai"
        """
        if tts_engine != "openai":
            return

        try:
            from audio_generator import AudioGenerator
            full_text = "\n\n".join(chapters.values())
            cost      = AudioGenerator.estimate_cost_openai(full_text)
            char_count = sum(len(t) for t in chapters.values())

            st.info(
                f"💰 **Estimated OpenAI TTS cost:** ${cost:.3f} USD  \n"
                f"({char_count:,} characters × $0.015 per 1,000 chars)"
            )
        except Exception:
            pass

    # ================================================================== #
    # SECTION 8 — FEEDBACK MESSAGES                                       #
    # ================================================================== #

    @staticmethod
    def render_error(message: str) -> None:
        """Display a red error banner."""
        st.error(f"❌ {message}")

    @staticmethod
    def render_success(message: str) -> None:
        """Display a green success banner."""
        st.success(f"✅ {message}")

    @staticmethod
    def render_warning(message: str) -> None:
        """Display a yellow warning banner."""
        st.warning(f"⚠️ {message}")

    @staticmethod
    def render_info(message: str) -> None:
        """Display a blue info banner."""
        st.info(f"ℹ️ {message}")

    @staticmethod
    def render_spinner_placeholder(message: str = "Processing…") -> None:
        """
        Display a processing message. Used outside of with-spinner blocks
        where a simple status message is sufficient.
        """
        st.caption(f"⏳ {message}")


# ------------------------------------------------------------------ #
# Standalone preview — run `streamlit run ui_components.py`           #
# to see all components rendered in isolation.                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    st.set_page_config(page_title="UIComponents Preview", layout="wide")
    st.title("UIComponents Preview")
    st.caption("Standalone preview of all UI components with mock data.")

    # Mock data
    from collections import OrderedDict as OD

    mock_chapters = OD([
        ("Introduction",
         "This is the introduction chapter with a good amount of content. " * 10),
        ("Chapter 1: Background",
         "This chapter covers the background of the topic in detail. " * 20),
        ("Chapter 2: Methods",
         "The methodology section describes how the research was conducted. " * 15),
        ("Chapter 3: Results",
         "The results show significant improvements across all metrics. " * 12),
        ("Conclusion",
         "In conclusion, the findings support the original hypothesis. " * 8),
    ])

    mock_sentiment = {
        "label":          "Positive",
        "positive_count": 12,
        "negative_count": 3,
        "confidence":     "medium",
    }

    mock_keywords  = [
        ("research", 15), ("methods", 12), ("results", 10),
        ("significant", 8), ("analysis", 7), ("data", 6),
        ("findings", 5), ("study", 4), ("evidence", 4), ("impact", 3),
    ]

    mock_characters = ["Smith", "Johnson", "Williams", "Brown"]

    # Initialize session state
    if "current_chapter" not in st.session_state:
        st.session_state.current_chapter = list(mock_chapters.keys())[0]
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []

    # Sidebar
    settings = UIComponents.render_sidebar_settings(max_pages=100)
    st.sidebar.divider()
    st.sidebar.write("**Settings returned:**")
    st.sidebar.json(settings)

    # Main layout
    st.subheader("Processing Status")
    UIComponents.render_processing_status("text", 47)

    st.divider()

    st.subheader("Metadata")
    UIComponents.render_metadata({
        "title":  "Sample Research Paper",
        "author": "J.H. García Willis",
        "subject": "Document Analysis",
    })

    st.divider()

    col_left, col_right = st.columns([1, 3])

    with col_left:
        UIComponents.render_table_of_contents(
            mock_chapters,
            st.session_state.current_chapter,
        )
        st.divider()
        UIComponents.render_bookmark_button(st.session_state.current_chapter)
        UIComponents.render_bookmarks_list(mock_chapters)

    with col_right:
        # Render player with silence as placeholder audio
        try:
            from audio_generator import AudioGenerator
            placeholder_audio = AudioGenerator._generate_silence(2000)
        except Exception:
            placeholder_audio = b""

        UIComponents.render_audio_player(
            audio_bytes=placeholder_audio,
            chapter_title=st.session_state.current_chapter,
            chapter_index=list(mock_chapters.keys()).index(
                st.session_state.current_chapter
            ),
            total_chapters=len(mock_chapters),
        )
        UIComponents.render_chapter_navigation(
            mock_chapters,
            st.session_state.current_chapter,
        )
        UIComponents.render_progress(
            st.session_state.current_chapter,
            mock_chapters,
        )

    st.divider()

    UIComponents.render_cost_estimate(mock_chapters, "openai")

    UIComponents.render_analysis_panel(
        keywords=mock_keywords,
        characters=mock_characters,
        summary=(
            "This research examines significant improvements in NLP methods. "
            "The study demonstrates clear evidence of performance gains. "
            "Findings have major implications for future research directions."
        ),
        sentiment=mock_sentiment,
        reading_time="4min 32sec",
        word_count=3_847,
    )