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
        keywords:     list,
        characters:   list,
        summary:      str,
        sentiment:    dict,
        reading_time: str,
        word_count:   int,
    ) -> None:
        """
        Render the document analysis panel inside an expander.

        Displays NLP analysis results computed by TextAnalyzer.
        Gracefully handles empty/None values for any individual field —
        each section is only rendered if data is available.

        Args:
          keywords:     list of (word, frequency) tuples
          characters:   list of character name strings
          summary:      extractive summary string
          sentiment:    dict from TextAnalyzer.sentiment_analysis()
          reading_time: string from TextAnalyzer.get_reading_time_estimate()
          word_count:   int from TextAnalyzer.get_word_count()
        """
        with st.expander("📊 Document Analysis", expanded=True):

            # Stats row
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word count", f"{word_count:,}")
            with col2:
                st.metric("Est. listening time", reading_time)

            st.divider()

            # Sentiment
            if sentiment:
                label = sentiment.get("label", "Unknown")
                confidence = sentiment.get("confidence", "low")
                pos = sentiment.get("positive_count", 0)
                neg = sentiment.get("negative_count", 0)

                color_map = {
                    "Positive": "🟢",
                    "Negative": "🔴",
                    "Neutral":  "🟡",
                }
                icon = color_map.get(label, "⚪")

                st.markdown(
                    f"**Sentiment:** {icon} {label} "
                    f"<small style='color:gray'>({confidence} confidence — "
                    f"{pos} positive signals, {neg} negative signals)</small>",
                    unsafe_allow_html=True,
                )
                st.divider()

            # Summary
            if summary:
                st.markdown("**Summary**")
                st.write(summary)
                st.caption(
                    "Extractive summary: first 3 sentences of the document."
                )
                st.divider()

            # Keywords
            if keywords:
                st.markdown("**Top Keywords**")
                # Display as a compact word-frequency list
                kw_parts = [
                    f"**{word}** ({count})"
                    for word, count in keywords[:10]
                ]
                st.write("  •  ".join(kw_parts))
                st.divider()

            # Characters
            if characters:
                st.markdown("**Recurring Names / Characters**")
                st.write(", ".join(characters[:15]))
            elif keywords is not None:
                # Only show this message if NLP ran but found nothing
                st.caption(
                    "No recurring character names detected."
                )

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