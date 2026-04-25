"""
ui_components.py
================
All Streamlit UI rendering. Pure presentation layer.
Contains zero business logic — takes pre-computed data and renders it.
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
        return st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help=(
                f"Maximum file size: {Config.MAX_PDF_SIZE_MB} MB. "
                "Text-based PDFs process in seconds; "
                "scanned PDFs require OCR and take longer."
            ),
            label_visibility="collapsed",
        )

    @staticmethod
    def local_file_selector(folder_path: str = ".") -> Optional[str]:
        try:
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        except OSError:
            st.warning(f"Cannot read folder: {folder_path}")
            return None

        if not pdf_files:
            st.info("No PDF files found in the current folder.")
            return None

        selected = st.selectbox("Select a PDF file", options=pdf_files, label_visibility="collapsed")
        return os.path.join(folder_path, selected)

    # ================================================================== #
    # SECTION 2 — SETTINGS SIDEBAR                                        #
    # ================================================================== #

    @staticmethod
    def render_sidebar_settings(max_pages: int = 1) -> dict:
        """
        Render all sidebar settings. Returns complete settings dict.
        """
        with st.sidebar:
            st.title(Config.APP_NAME)
            st.caption(f"v{Config.VERSION}")
            st.divider()

            st.subheader("Language")
            lang_code = UIComponents._language_selector()
            st.divider()

            if Config.is_translation_available():
                st.subheader("🌐 Translation")
                translate, target_lang, trans_engine = UIComponents._translation_settings(lang_code)
                st.divider()
            else:
                translate, target_lang, trans_engine = False, lang_code, "none"

            st.subheader("TTS Engine")
            tts_engine = UIComponents._tts_engine_selector()
            st.divider()

            st.subheader("Audio Settings")
            rate, pitch = UIComponents._rate_pitch_sliders()
            st.divider()

            st.subheader("Page Range")
            start_page, end_page = UIComponents._page_range_selector(max_pages)
            st.divider()

            st.subheader("Options")
            remove_hf, show_analysis, remove_inc_pages, remove_inc_chapters = UIComponents._feature_toggles()

        return {
            "lang_code":                   lang_code,
            "tts_engine":                  tts_engine,
            "rate":                        rate,
            "pitch":                       pitch,
            "start_page":                  start_page,
            "end_page":                    end_page,
            "remove_headers_footers":      remove_hf,
            "show_analysis":               show_analysis,
            "translate":                   translate,
            "target_lang":                 target_lang,
            "translation_engine":          trans_engine,
            "remove_incomplete_pages":     remove_inc_pages,
            "remove_incomplete_chapters":  remove_inc_chapters,
        }

    @staticmethod
    def _language_selector() -> str:
        labels     = Config.get_all_language_labels()
        label_list = list(labels.values())
        code_list  = list(labels.keys())

        # Reflect auto-detected language if set
        default_idx = 0
        if st.session_state.get("detected_lang"):
            detected = st.session_state.detected_lang
            if detected in code_list:
                default_idx = code_list.index(detected)

        selected_label = st.selectbox(
            "Document language",
            options=label_list,
            index=default_idx,
            help=(
                "Language of the source PDF. Used for OCR accuracy and TTS voice. "
                "Auto-detection will override this if enabled."
            ),
        )
        return code_list[label_list.index(selected_label)]

    @staticmethod
    def _tts_engine_selector() -> str:
        openai_available = Config.is_openai_available()

        if openai_available:
            options    = ["Free (gTTS)", "Premium (OpenAI)"]
            engine_map = {"Free (gTTS)": "gtts", "Premium (OpenAI)": "openai"}
            selected   = st.radio(
                "TTS engine",
                options=options,
                index=0,
                help=(
                    "Free: Google Translate TTS — functional, no cost.\n\n"
                    "Premium: OpenAI tts-1 — natural voice, ~$0.015/1,000 chars."
                ),
            )
            return engine_map[selected]
        else:
            st.caption("🟢 Free (gTTS) — set OPENAI_API_KEY to unlock Premium")
            return "gtts"

    @staticmethod
    def _rate_pitch_sliders() -> tuple[float, float]:
        rate = st.slider(
            "Speed",
            min_value=0.5,
            max_value=2.0,
            value=Config.DEFAULT_RATE,
            step=0.05,
            help=(
                "Playback speed. 1.0 = normal. "
                "Below 0.8 uses gTTS slow mode natively. "
                "Other values are applied via MP3 frame rate adjustment."
            ),
        )
        pitch = st.slider(
            "Pitch",
            min_value=0.5,
            max_value=2.0,
            value=Config.DEFAULT_PITCH,
            step=0.05,
            help="Pitch adjustment. Currently informational — applied via OpenAI engine only.",
        )
        return rate, pitch

    @staticmethod
    def _page_range_selector(max_pages: int) -> tuple[int, int]:
        effective_max = max(max_pages, 1)
        col1, col2   = st.columns(2)

        # Persist the correct end-page default in session state so that
        # when the sidebar re-renders with the real page count, the
        # "To page" field shows the last page — not 1.
        # Streamlit clamps number_input value to [min, max] on first render,
        # so if max_pages=1 initially, value would be clamped to 1.
        # Using session state bypasses this.
        ss_key = "page_range_end_default"
        if effective_max > 1:
            # Update session state whenever we have a real page count
            st.session_state[ss_key] = effective_max
        stored_default = st.session_state.get(ss_key, effective_max)
        # Clamp stored default to current valid range
        default_end = min(max(stored_default, 1), effective_max)

        with col1:
            start_page = st.number_input(
                "From page",
                min_value=1,
                max_value=effective_max,
                value=1,
                step=1,
                help="First page to process (1-indexed, inclusive).",
            )
        with col2:
            end_page = st.number_input(
                "To page",
                min_value=1,
                max_value=effective_max,
                value=default_end,
                step=1,
                help=(
                    f"Last page to process. PDF has {effective_max} pages. "
                    "Auto-extends if a chapter boundary is detected mid-range."
                ),
            )

        if effective_max > 1:
            selected = int(end_page) - int(start_page) + 1
            st.caption(
                f"{effective_max} pages total · {selected} selected "
                + ("· Full document" if selected == effective_max else f"· Pages {int(start_page)}–{int(end_page)}")
            )
        return int(start_page), int(end_page)

    @staticmethod
    def _feature_toggles() -> tuple:
        """
        Render feature toggle checkboxes.
        Returns (remove_hf, show_analysis, remove_incomplete_pages, remove_incomplete_chapters)
        """
        remove_hf = st.checkbox(
            "Remove headers & footers",
            value=True,
            help=(
                "Detects and removes running headers, footers, and page numbers "
                "before generating audio. Uses Y-band repetition detection and "
                "noise pattern matching. Strongly recommended."
            ),
        )
        show_analysis = st.checkbox(
            "Show document analysis",
            value=False,
            help=(
                "After processing, display a full Document Intelligence panel: "
                "readability scores, vocabulary richness, topic density, "
                "content type detection, chapter complexity ranking, "
                "sentiment analysis, keyword frequency, and a summary."
            ),
        )

        st.markdown("**Content completeness**")
        st.caption(
            "These options verify and enforce that only fully captured "
            "pages and chapters are included in the audio output. "
            "Verification uses word fingerprinting: the last few words "
            "of each page are checked against the assembled text."
        )

        remove_incomplete_pages = st.checkbox(
            "Remove incomplete pages",
            value=True,
            help=(
                "For each chapter, the last page is verified by checking "
                "whether its final words appear in the assembled chapter text. "
                "If the fingerprint is not found, that page was cut off and "
                "is removed from the audio. "
                "Recommended ON — prevents audio cutting off mid-sentence."
            ),
        )
        remove_incomplete_chapters = st.checkbox(
            "Remove incomplete chapters",
            value=False,
            help=(
                "If a chapter's last page fails fingerprint verification, "
                "or the chapter text ends mid-word, the entire chapter is "
                "excluded from the audio output. "
                "More aggressive than page removal — off by default. "
                "Enable if you want strictly complete chapters only."
            ),
        )

        return remove_hf, show_analysis, remove_incomplete_pages, remove_incomplete_chapters

    @staticmethod
    def _translation_settings(source_lang: str) -> tuple:
        """
        Translation controls. Returns (translate, target_lang, engine).
        """
        translate = st.toggle(
            "Translate output",
            value=False,
            help=(
                "Translate extracted text to a different language before "
                "generating audio. Translation runs after header/footer removal "
                "and chapter splitting. Document analysis always runs on the "
                "original source language."
            ),
        )

        if not translate:
            return False, source_lang, "auto"

        trans_langs  = Config.TRANSLATION_SUPPORTED_LANGUAGES
        lang_labels  = list(trans_langs.values())
        lang_codes   = list(trans_langs.keys())

        default_target = "en" if source_lang != "en" else "es"
        default_idx    = lang_codes.index(default_target) if default_target in lang_codes else 0

        selected_label = st.selectbox(
            "Translate to",
            options=lang_labels,
            index=default_idx,
        )
        target_lang = lang_codes[lang_labels.index(selected_label)]

        # Check against auto-detected language too (auto-detect overrides sidebar)
        effective_source = st.session_state.get("detected_lang", source_lang) or source_lang
        if target_lang == effective_source:
            detected_label = Config.SUPPORTED_LANGUAGES.get(effective_source, {}).get("label", effective_source)
            target_label   = Config.SUPPORTED_LANGUAGES.get(target_lang, {}).get("label", target_lang)
            st.caption(
                f"Source ({detected_label}) and target ({target_label}) are the same "
                f"— no translation will run."
            )
        elif effective_source != source_lang:
            detected_label = Config.SUPPORTED_LANGUAGES.get(effective_source, {}).get("label", effective_source)
            st.caption(
                f"Source auto-detected as {detected_label}. "
                f"Translation will run after language detection."
            )

        st.markdown("**Engine status:**")
        g_icon = "🟢" if Config.is_google_translate_available() else "🔴"
        l_icon = "🟢" if Config.is_libretranslate_available()   else "🔴"
        st.caption(f"{g_icon} Google Translate (free, unofficial API)")
        st.caption(f"{l_icon} LibreTranslate ({Config.LIBRETRANSLATE_URL})")
        st.caption("Rate limits: Google — no hard limit for occasional use. LibreTranslate public — ~80 requests/hour × 4,500 chars each = ~360,000 chars/hour (~25–40 articles/hour).")

        engine_options = ["Auto (recommended)"]
        if Config.is_google_translate_available():
            engine_options.append("Google only")
        if Config.is_libretranslate_available():
            engine_options.append("LibreTranslate only")

        selected_engine = st.radio(
            "Engine preference",
            options=engine_options,
            index=0,
            label_visibility="collapsed",
        )

        engine_map    = {
            "Auto (recommended)": "auto",
            "Google only":        "google",
            "LibreTranslate only": "libretranslate",
        }
        return True, target_lang, engine_map.get(selected_engine, "auto")

    # ================================================================== #
    # SECTION 3 — STATUS BANNERS                                          #
    # ================================================================== #

    @staticmethod
    def render_detected_language_banner(detected_lang: str, selected_lang: str) -> None:
        if detected_lang == selected_lang:
            return
        detected_label = Config.SUPPORTED_LANGUAGES.get(detected_lang, {}).get("label", detected_lang)
        selected_label = Config.SUPPORTED_LANGUAGES.get(selected_lang, {}).get("label", selected_lang)
        st.info(
            f"🔍 Auto-detected language: **{detected_label}**. "
            f"Processing in {detected_label}. "
            f"(Sidebar was set to {selected_label} — overridden automatically.)"
        )

    @staticmethod
    def render_translation_status(
        chapters_translated: int,
        total_chapters:      int,
        source_lang:         str,
        target_lang:         str,
        engine_used:         str,
    ) -> None:
        source_label = Config.SUPPORTED_LANGUAGES.get(source_lang, {}).get("label", source_lang)
        target_label = Config.SUPPORTED_LANGUAGES.get(target_lang, {}).get("label", target_lang)
        engine_names = {"google": "Google Translate", "libretranslate": "LibreTranslate"}
        engine_name  = engine_names.get(engine_used, engine_used)
        st.success(
            f"🌐 Translated {chapters_translated}/{total_chapters} chapters — "
            f"{source_label} → {target_label} via {engine_name}"
        )

    @staticmethod
    def render_page_range_warning(
        original_end:    int,
        extended_end:    int,
        chapters_cut:    list[dict],
    ) -> None:
        """
        Show a banner explaining that the page range was auto-extended
        because it would have cut through one or more chapters.

        Args:
          original_end:  the page the user originally requested
          extended_end:  the page we extended to
          chapters_cut:  list of warning dicts from get_chapter_aware_page_range
        """
        if not chapters_cut:
            return

        chapter_names = ", ".join(
            f"'{w['chapter']}' (ends p.{w['chapter_end']})"
            for w in chapters_cut
        )

        st.warning(
            f"📖 **Page range extended: p.{original_end} → p.{extended_end}**  \n"
            f"Your selection would have cut through {chapter_names}. "
            f"The range was automatically extended to include complete chapters. "
            f"You can disable this in Config (AUTO_EXTEND_PAGE_RANGE=False) "
            f"if you prefer to process exact page ranges and let the "
            f"completeness checker remove incomplete content instead."
        )

    @staticmethod
    def render_completeness_report(
        incomplete_pages:    set,
        incomplete_chapters: set,
        total_pages:         int,
        total_chapters:      int,
    ) -> None:
        """
        Show a summary of what the completeness checker found and removed.
        Displayed after processing so the user knows exactly what they got.
        """
        if not incomplete_pages and not incomplete_chapters:
            st.success(
                "✅ **Completeness verified** — all pages and chapters "
                "passed fingerprint verification. Audio covers the full "
                "requested range."
            )
            return

        lines = ["**⚠️ Completeness report:**"]

        if incomplete_pages:
            page_list = ", ".join(str(p + 1) for p in sorted(incomplete_pages))
            lines.append(
                f"- **{len(incomplete_pages)} page(s) removed** "
                f"(fingerprint not found in assembled text): pages {page_list}"
            )

        if incomplete_chapters:
            ch_list = ", ".join(f"'{c}'" for c in sorted(incomplete_chapters))
            lines.append(
                f"- **{len(incomplete_chapters)} chapter(s) removed** "
                f"(last page failed verification or text ends mid-word): {ch_list}"
            )

        lines.append(
            f"Audio covers {total_chapters - len(incomplete_chapters)} of "
            f"{total_chapters} detected chapters."
        )

        st.warning("\n".join(lines))

    @staticmethod
    def render_pre_processing_summary(
        total_pages:    int,
        selected_pages: int,
        chapters_found: list[dict],
        est_minutes:    float,
        est_audio_hrs:  float,
        warnings:       list[dict],
        extended_end:   int,
        original_end:   int,
    ) -> None:
        """
        Show the user a complete picture of what they're about to process
        BEFORE they click Convert. Displayed immediately after a PDF is loaded.

        Covers:
          - Total pages vs selected pages
          - Chapters detected in pre-scan
          - Estimated processing time and audio output length
          - Any chapter-boundary warnings with auto-extend notice
        """
        st.markdown("### 📋 What you're about to process")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total pages", total_pages)
        with c2:
            st.metric("Selected pages", selected_pages)
        with c3:
            st.metric("Chapters detected", len(chapters_found) if chapters_found else "—")
        with c4:
            st.metric("Est. audio output", f"{est_audio_hrs:.1f} hrs" if est_audio_hrs >= 1 else f"{int(est_audio_hrs * 60)} min")

        st.caption(
            f"Estimated processing time: {est_minutes:.0f}–{est_minutes * 1.5:.0f} min "
            f"(depends on TTS engine and network). "
            f"Keep this tab open — Streamlit does not process in the background."
        )

        # Chapter list from pre-scan
        if chapters_found:
            with st.expander(f"Chapters detected in pre-scan ({len(chapters_found)})", expanded=False):
                for ch in chapters_found:
                    in_range = ch["start"] <= (extended_end or original_end)
                    icon     = "✅" if in_range else "⬜"
                    st.caption(
                        f"{icon} **{ch['title']}** — pp. {ch['start']}–{ch['end']}"
                    )

        # Warnings about chapter cuts
        if warnings:
            for w in warnings:
                if w.get("extended"):
                    st.info(
                        f"📖 Chapter boundary detected: **'{w['chapter']}'** "
                        f"starts within your selection but ends on p.{w['chapter_end']}. "
                        f"Page range auto-extended from p.{original_end} to p.{extended_end} "
                        f"so this chapter is fully included."
                    )
                else:
                    st.warning(
                        f"⚠️ Chapter **'{w['chapter']}'** ends on p.{w['chapter_end']} "
                        f"but your selection ends on p.{original_end}. "
                        f"This chapter will be cut. Enable AUTO_EXTEND_PAGE_RANGE "
                        f"in config to avoid this automatically."
                    )

    # ================================================================== #
    # SECTION 4 — DOCUMENT DISPLAY                                        #
    # ================================================================== #

    @staticmethod
    def render_metadata(metadata: dict) -> None:
        if not metadata:
            return
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
        if mode_used == "text":
            st.success(
                f"✅ **Text mode** — native text extraction "
                f"({page_count} pages). Fast and accurate."
            )
        else:
            st.warning(
                f"🔍 **OCR mode** — scanned PDF ({page_count} pages). "
                f"Quality depends on scan resolution."
            )

    @staticmethod
    def render_table_of_contents(
        chapters:        OrderedDict,
        current_chapter: str,
        page_ranges:     dict = None,
        chapter_status:  dict = None,
    ) -> None:
        """
        Clickable TOC with page ranges, word counts, and live status icons.
        """
        st.subheader("Chapters")
        if not chapters:
            st.caption("No chapters detected.")
            return

        status_icons = {
            "done":       "✅",
            "processing": "⏳",
            "failed":     "❌",
            "queued":     "⬜",
            "removed":    "🗑️",
        }

        for i, title in enumerate(chapters.keys()):
            display_title = title if len(title) <= 40 else title[:37] + "…"
            page_label    = ""
            if page_ranges and title in page_ranges:
                p1, p2 = page_ranges[title]
                if p1 > 0:
                    page_label = f"pp. {p1}–{p2}" if p1 != p2 else f"p. {p1}"

            status      = chapter_status.get(title, "") if chapter_status else ""
            icon        = status_icons.get(status, "")
            word_count  = len(chapters[title].split()) if chapters[title] else 0
            sub_parts   = []
            if page_label:
                sub_parts.append(page_label)
            if word_count:
                sub_parts.append(f"{word_count:,} words")
            sub         = " · ".join(sub_parts)
            is_disabled = status in ("processing", "queued", "failed", "removed")

            if title == current_chapter:
                st.markdown(f"**{icon} ▶ {display_title}**")
                if sub:
                    st.caption(sub)
            else:
                btn_label = f"{icon} {display_title}".strip() if icon else display_title
                if st.button(
                    btn_label,
                    key=f"toc_btn_{i}",
                    use_container_width=True,
                    help=sub or f"{word_count:,} words",
                    disabled=is_disabled,
                ):
                    st.session_state.current_chapter = title
                    st.rerun()

    @staticmethod
    def render_pdf_preview(pdf_bytes: bytes) -> None:
        with st.expander("📄 Preview PDF", expanded=False):
            try:
                b64_pdf  = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_html = (
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                    f'width="100%" height="500px" type="application/pdf"></iframe>'
                )
                st.markdown(pdf_html, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"PDF preview unavailable: {e}")

    # ================================================================== #
    # SECTION 5 — AUDIO PLAYER                                           #
    # ================================================================== #

    @staticmethod
    def render_audio_player(
        audio_bytes:    bytes,
        chapter_title:  str,
        chapter_index:  int,
        total_chapters: int,
    ) -> None:
        if not audio_bytes:
            st.warning("No audio available for this chapter.")
            return

        st.markdown(
            f"**{chapter_title}**  "
            f"<small style='color:gray'>Chapter {chapter_index + 1} of {total_chapters}</small>",
            unsafe_allow_html=True,
        )
        st.audio(audio_bytes, format="audio/mp3")

    @staticmethod
    def render_chapter_navigation(chapters: OrderedDict, current_chapter: str) -> None:
        chapter_list = list(chapters.keys())
        current_idx  = chapter_list.index(current_chapter) if current_chapter in chapter_list else 0

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_idx > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_chapter = chapter_list[current_idx - 1]
                    st.rerun()

        with col2:
            st.caption(f"Chapter {current_idx + 1} of {len(chapter_list)}")

        with col3:
            if current_idx < len(chapter_list) - 1:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_chapter = chapter_list[current_idx + 1]
                    st.rerun()

    @staticmethod
    def render_progress(current_chapter: str, chapters: OrderedDict) -> None:
        chapter_list = list(chapters.keys())
        if not chapter_list:
            return
        idx = chapter_list.index(current_chapter) if current_chapter in chapter_list else 0
        pct = (idx + 1) / len(chapter_list)
        st.progress(pct, text=f"Progress: {idx + 1}/{len(chapter_list)} chapters")

    # ================================================================== #
    # SECTION 5B — LIVE CHAPTER QUEUE                                    #
    # ================================================================== #

    @staticmethod
    def render_chapter_queue(
        chapters:       OrderedDict,
        audio_data:     dict,
        chapter_status: dict,
        page_ranges:    dict = None,
    ) -> None:
        """
        Live queue showing per-chapter status during and after processing.
        Each completed chapter gets an individual download button immediately.
        """
        status_icons = {
            "done":       "✅",
            "processing": "⏳",
            "failed":     "❌",
            "queued":     "⬜",
            "removed":    "🗑️",
        }

        done_count    = sum(1 for s in chapter_status.values() if s == "done")
        total_count   = sum(1 for s in chapter_status.values() if s != "removed")
        removed_count = sum(1 for s in chapter_status.values() if s == "removed")
        pct           = int(done_count / total_count * 100) if total_count > 0 else 0

        col_h1, col_h2 = st.columns([4, 1])
        with col_h1:
            st.markdown(
                f"**Processing queue** — {done_count}/{total_count} chapters complete"
                + (f" · {removed_count} removed" if removed_count else "")
            )
        with col_h2:
            st.caption(f"{pct}% done")

        st.progress(pct / 100.0)

        for title, text in chapters.items():
            status      = chapter_status.get(title, "queued")
            icon        = status_icons.get(status, "⬜")
            display     = title if len(title) <= 35 else title[:32] + "…"
            word_count  = len(text.split()) if text else 0

            page_label  = ""
            if page_ranges and title in page_ranges:
                p1, p2 = page_ranges[title]
                if p1 > 0:
                    page_label = f"pp. {p1}–{p2}" if p1 != p2 else f"p. {p1}"

            sub = f"{page_label} · {word_count:,} words" if page_label else f"{word_count:,} words"

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{icon} **{display}**")
                st.caption(sub)
            with col2:
                if status == "done" and title in audio_data:
                    audio_bytes = audio_data[title]
                    safe_name   = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:30]
                    # Estimate duration from file size (MP3 at ~128kbps)
                    dur_secs    = len(audio_bytes) / 16_000.0
                    mins        = int(dur_secs) // 60
                    secs        = int(dur_secs) % 60
                    st.download_button(
                        label=f"⬇️ {mins}:{secs:02d}",
                        data=audio_bytes,
                        file_name=f"{safe_name}.mp3",
                        mime="audio/mpeg",
                        key=f"dl_queue_{title[:20]}_{hash(title) % 9999}",
                        use_container_width=True,
                    )
                elif status == "processing":
                    st.caption("⏳")
                elif status == "failed":
                    st.caption("❌")
                elif status == "removed":
                    st.caption("🗑️")
                else:
                    st.caption("⬜")

    # ================================================================== #
    # SECTION 5C — DOWNLOAD BUTTONS                                       #
    # ================================================================== #

    @staticmethod
    def render_download_buttons(
        audio_data:      dict,
        current_chapter: str,
        chapters:        OrderedDict,
    ) -> None:
        """
        Chapter download + Full Book download.
        Full book label shows partial count when not all chapters are ready.
        """
        st.markdown("**⬇️ Downloads**")
        col1, col2 = st.columns(2)

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

        with col2:
            if audio_data:
                silence_gap = bytes([
                    0xFF, 0xFB, 0x90, 0x00,
                    *([0x00] * 40),
                ]) * 32

                buf           = io.BytesIO()
                chapter_order = list(chapters.keys())

                for i, title in enumerate(chapter_order):
                    ch_audio = audio_data.get(title, b"")
                    if ch_audio:
                        buf.write(ch_audio)
                        if i < len(chapter_order) - 1:
                            buf.write(silence_gap)

                full_book_bytes = buf.getvalue()

                if full_book_bytes:
                    available = len(audio_data)
                    total     = len(chapters)
                    label     = (
                        f"📚 Full Book ({available} of {total} chapters)"
                        if available < total
                        else f"📚 Full Book ({available} chapters)"
                    )
                    st.download_button(
                        label=label,
                        data=full_book_bytes,
                        file_name="full_book.mp3",
                        mime="audio/mpeg",
                        use_container_width=True,
                        help=f"Download {available} available chapters as one MP3",
                    )
                else:
                    st.button("📚 Full Book", disabled=True, use_container_width=True)
            else:
                st.button("📚 Full Book", disabled=True, use_container_width=True)

    # ================================================================== #
    # SECTION 6 — BOOKMARKS                                               #
    # ================================================================== #

    @staticmethod
    def render_bookmark_button(current_chapter: str) -> None:
        bookmarks = st.session_state.get("bookmarks", [])
        if current_chapter in bookmarks:
            if st.button("🔖 Remove bookmark", use_container_width=True):
                st.session_state.bookmarks.remove(current_chapter)
                st.rerun()
        else:
            if st.button("🔖 Add bookmark", use_container_width=True):
                if "bookmarks" not in st.session_state:
                    st.session_state.bookmarks = []
                st.session_state.bookmarks.append(current_chapter)
                st.rerun()

    @staticmethod
    def render_bookmarks_list(chapters: OrderedDict) -> None:
        bookmarks = st.session_state.get("bookmarks", [])
        if not bookmarks:
            return

        st.markdown("**🔖 Bookmarks**")
        stale = []
        for i, bm in enumerate(bookmarks):
            if bm not in chapters:
                stale.append(bm)
                continue
            display = bm if len(bm) <= 30 else bm[:27] + "…"
            if st.button(display, key=f"bm_btn_{i}", use_container_width=True):
                st.session_state.current_chapter = bm
                st.rerun()

        if stale:
            st.session_state.bookmarks = [b for b in bookmarks if b not in stale]

    # ================================================================== #
    # SECTION 7 — ANALYSIS PANEL                                          #
    # ================================================================== #

    @staticmethod
    def render_analysis_panel(
        keywords:             list,
        characters:           list,
        summary:              str,
        sentiment:            dict,
        reading_time:         str,
        word_count:           int,
        readability:          dict  = None,
        text_stats:           dict  = None,
        content_type:         dict  = None,
        topic_density:        list  = None,
        chapter_complexity:   list  = None,
        lexical_diversity:    dict  = None,
        sentence_complexity:  dict  = None,
    ) -> None:
        st.subheader("📊 Document Intelligence")

        # Row 1: core KPIs
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
                          help="Unique words / total words × 100. Higher = more diverse.")

        st.divider()

        # Row 2: readability + content type
        col_read, col_type = st.columns([3, 2])

        with col_read:
            st.markdown("**📖 Readability**")
            if readability:
                fe = readability.get("flesch_ease", 0)
                fk = readability.get("flesch_kincaid", 0)
                gl = readability.get("grade_label", "Unknown")
                el = readability.get("ease_label", "Unknown")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Flesch Ease", f"{fe}/100",
                              help="0=hardest, 100=easiest. 60–70 is standard adult reading.")
                with r2:
                    st.metric("Grade Level", f"{fk}")
                with r3:
                    st.metric("Reading Level", gl)
                bar_color = "🟢" if fe >= 70 else ("🟡" if fe >= 50 else "🔴")
                st.caption(f"{bar_color} {el} — {fe}/100 ease score")
                if text_stats:
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("Avg Sentence", f"{text_stats.get('avg_sentence_length', 0)} words",
                                  help="Sentences >25 words are harder to follow in audio.")
                    with s2:
                        st.metric("Avg Word Length", f"{text_stats.get('avg_word_length', 0)} chars")
                    with s3:
                        st.metric("Sentences", f"{text_stats.get('total_sentences', 0):,}")

        with col_type:
            st.markdown("**🔍 Content Type**")
            if content_type:
                ct   = content_type.get("type", "General")
                conf = content_type.get("confidence", 0)
                type_icons = {
                    "Academic":            "🎓",
                    "Report / Analysis":   "📈",
                    "Fiction / Narrative": "📚",
                    "News / Article":      "📰",
                    "Technical":           "⚙️",
                    "Legal":               "⚖️",
                    "General":             "📄",
                }
                st.metric(f"{type_icons.get(ct, '📄')} Detected Type", ct)
                st.caption(f"Confidence: {conf}%")
                for label, score in sorted(
                    content_type.get("all_scores", {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:3]:
                    if score > 0:
                        st.caption(f"  • {label}: {score} signals")

        st.divider()

        # Row 3: summary + sentiment
        col_sum, col_sent = st.columns([3, 2])

        with col_sum:
            st.markdown("**📝 Summary**")
            if summary:
                st.write(summary)
                st.caption("Extractive — first 3 content sentences.")
            else:
                st.caption("Summary unavailable.")

        with col_sent:
            st.markdown("**🎭 Tone & Sentiment**")
            if sentiment:
                label = sentiment.get("label", "Unknown")
                conf  = sentiment.get("confidence", "low")
                pos   = sentiment.get("positive_count", 0)
                neg   = sentiment.get("negative_count", 0)
                color_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                st.metric(f"{color_map.get(label, '⚪')} Tone", label)
                st.caption(f"Confidence: {conf}")
                st.caption(f"🟢 {pos} positive  •  🔴 {neg} negative")

        st.divider()

        # Row 4: topic density
        if topic_density:
            st.markdown("**🔑 Topic Density** *(occurrences per 1,000 words)*")
            for item in topic_density[:12]:
                word    = item["word"]
                density = item["density_per_1k"]
                bar_pct = item["bar_pct"]
                count   = item["count"]
                filled  = int(bar_pct / 5)
                bar     = "█" * filled + "░" * (20 - filled)
                st.markdown(f"`{word:<18}` {bar} **{density}**/1k  *(×{count})*")
            st.divider()

        # Row 5: chapter complexity
        if chapter_complexity:
            st.markdown("**📊 Complexity by Chapter** *(hardest → easiest to follow while listening)*")
            for item in chapter_complexity:
                title = item["title"]
                ease  = item["flesch_ease"]
                label = item["ease_label"]
                wc    = item["word_count"]
                icon  = "🔴" if ease < 50 else ("🟡" if ease < 70 else "🟢")
                short = title if len(title) <= 40 else title[:37] + "…"
                st.markdown(f"{icon} **{short}** — {label} ({ease}/100) · {wc:,} words")
            st.divider()

        # Row 6: Lexical diversity (new KPI)
        if lexical_diversity:
            st.markdown("**🔤 Lexical Diversity**")
            st.caption(
                "MATTR (Moving Average Type-Token Ratio) is length-independent and more "
                "reliable than simple vocabulary richness for longer documents."
            )
            ld1, ld2, ld3, ld4 = st.columns(4)
            with ld1:
                st.metric("MATTR", f"{lexical_diversity.get('mattr', 0)}%",
                          help="Higher = more diverse vocabulary (length-independent)")
            with ld2:
                st.metric("Type-Token Ratio", f"{lexical_diversity.get('ttr', 0)}%")
            with ld3:
                st.metric("Hapax Words", f"{lexical_diversity.get('hapax_count', 0):,}",
                          help="Words appearing exactly once — indicator of specialized vocabulary")
            with ld4:
                st.metric("Avg Word Frequency", f"{lexical_diversity.get('avg_word_frequency', 0)}×")
            st.divider()

        # Row 7: Sentence complexity (new KPI)
        if sentence_complexity:
            st.markdown("**📏 Sentence Complexity** *(audio comprehension impact)*")
            long_pct = sentence_complexity.get("long_sentence_pct", 0)
            long_n   = sentence_complexity.get("long_sentence_count", 0)
            longest  = sentence_complexity.get("longest_sentence_words", 0)
            sc_color = "🔴" if long_pct > 30 else ("🟡" if long_pct > 15 else "🟢")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric(
                    "Long Sentences (>30 words)",
                    f"{long_n} ({long_pct}%)",
                    help="Sentences over 30 words are difficult to follow while listening"
                )
            with sc2:
                st.metric("Longest Sentence", f"{longest} words")
            with sc3:
                st.metric("Short Fragments (<5 words)",
                          f"{sentence_complexity.get('short_sentence_count', 0)}")
            if long_pct > 20:
                st.caption(
                    f"{sc_color} {long_pct}% of sentences are long (>30 words). "
                    "Consider using a slower playback speed for complex sections."
                )
            st.divider()

        # Row 8: entities
        if characters:
            st.markdown("**👥 Recurring Names / Entities**")
            st.write("  •  ".join(characters[:20]))
            st.divider()

        # Row 9: keywords
        if keywords:
            st.markdown("**💬 Top Keywords**")
            kw_parts = [f"**{w}** ({ct})" for w, ct in keywords[:15]]
            st.write("  •  ".join(kw_parts))

    # ================================================================== #
    # SECTION 8 — COST + FEEDBACK                                         #
    # ================================================================== #

    @staticmethod
    def render_cost_estimate(chapters: OrderedDict, tts_engine: str) -> None:
        if tts_engine != "openai":
            return
        total_chars = sum(len(t) for t in chapters.values())
        cost_usd    = (total_chars / 1000) * 0.015
        st.info(
            f"💰 **OpenAI TTS estimate:** {total_chars:,} characters — "
            f"~${cost_usd:.3f} USD at $0.015/1,000 chars."
        )

    @staticmethod
    def render_error(message: str) -> None:
        st.error(f"❌ {message}")

    @staticmethod
    def render_success(message: str) -> None:
        st.success(f"✅ {message}")

    @staticmethod
    def render_warning(message: str) -> None:
        st.warning(f"⚠️ {message}")

    @staticmethod
    def render_info(message: str) -> None:
        st.info(f"ℹ️ {message}")

    @staticmethod
    def render_spinner_placeholder(message: str = "Processing…") -> None:
        st.info(f"⏳ {message}")