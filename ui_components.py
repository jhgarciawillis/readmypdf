"""
ui_components.py
================
All Streamlit UI rendering — sidebar settings, banners, chapter player,
downloads, audio adjustment, document analysis, bookmarks.

Structure:
  SECTION 1  — Sidebar settings (language, translation, TTS, page range, options)
  SECTION 2  — File input widgets
  SECTION 3  — Pre-processing summary
  SECTION 4  — Status banners (extraction mode, translation, completeness)
  SECTION 5  — Chapter queue (live progress + per-chapter downloads)
  SECTION 5B — Audio player + post-generation adjustment panel
  SECTION 5C — Download buttons
  SECTION 6  — Bookmarks
  SECTION 7  — Document analysis panel
  SECTION 8  — Feedback / error rendering
"""

import io
import re
from collections import OrderedDict

import streamlit as st

from config import Config


class UIComponents:
    """All UI rendering, organised as static methods."""

    # ================================================================== #
    # SECTION 1 — SIDEBAR SETTINGS                                        #
    # ================================================================== #

    # gTTS TLD map for voice gender by language.
    # (female_tld, male_tld) — different Google regional servers produce
    # perceptibly different voice characteristics at no cost.
    GTTS_VOICE_TLDS: dict = {
        "en":  ("com",    "co.uk"),    # US neutral vs British
        "es":  ("com",    "com.mx"),   # neutral vs Mexican (slightly deeper)
        "fr":  ("fr",     "ca"),       # France vs Canadian French
        "de":  ("de",     "at"),       # Germany vs Austrian German
        "pt":  ("com.br", "pt"),       # Brazilian vs European Portuguese
        "it":  ("it",     "com"),
        "ja":  ("co.jp",  "co.jp"),
        "zh":  ("com.hk", "com"),
        "ar":  ("com",    "com"),
        "ru":  ("ru",     "com"),
    }

    @staticmethod
    def _get_gtts_tld(lang_code: str, voice_gender: str) -> str:
        tlds = UIComponents.GTTS_VOICE_TLDS.get(lang_code, ("com", "com"))
        return tlds[0] if voice_gender == "Female" else tlds[1]

    @staticmethod
    def render_sidebar_settings(max_pages: int = 1) -> dict:
        """Render all sidebar settings. Returns complete settings dict."""
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

            # Voice gender — only for gTTS (free, no API key)
            # Uses regional TLD routing to produce slightly different voice
            voice_gender = "Female"
            if tts_engine == "gtts":
                st.subheader("🎙️ Voice")
                voice_gender = st.radio(
                    "Voice",
                    options=["Female", "Male"],
                    horizontal=True,
                    key="voice_gender_radio",
                    help=(
                        "Different Google TTS regional endpoints produce "
                        "perceptibly different voices. Free — no API key needed.\n\n"
                        "• Female: standard Google US/neutral voice\n"
                        "• Male: regional variant (British EN, Mexican ES, etc.)"
                    ),
                )
                st.divider()

            st.subheader("Page Range")
            start_page, end_page = UIComponents._page_range_selector(max_pages)
            st.divider()

            st.subheader("Options")
            remove_hf, show_analysis, remove_inc_pages, remove_inc_chapters = UIComponents._feature_toggles()

        return {
            "lang_code":                   lang_code,
            "tts_engine":                  tts_engine,
            "rate":                        1.0,   # generation rate fixed at 1.0
            "pitch":                       1.0,   # pitch fixed at 1.0 (post-gen adjustment)
            "start_page":                  start_page,
            "end_page":                    end_page,
            "remove_headers_footers":      remove_hf,
            "show_analysis":               show_analysis,
            "translate":                   translate,
            "target_lang":                 target_lang,
            "translation_engine":          trans_engine,
            "remove_incomplete_pages":     remove_inc_pages,
            "remove_incomplete_chapters":  remove_inc_chapters,
            "voice_gender":                voice_gender,
            "gtts_tld":                    UIComponents._get_gtts_tld(lang_code, voice_gender),
        }

    @staticmethod
    def _language_selector() -> str:
        labels     = Config.get_all_language_labels()
        label_list = list(labels.values())
        code_list  = list(labels.keys())

        WIDGET_KEY = "lang_selector_value"
        detected = st.session_state.get("detected_lang")
        if detected and detected in code_list:
            detected_label = label_list[code_list.index(detected)]
            if st.session_state.get(WIDGET_KEY) != detected_label:
                st.session_state[WIDGET_KEY] = detected_label

        if WIDGET_KEY not in st.session_state:
            st.session_state[WIDGET_KEY] = label_list[0]

        selected_label = st.selectbox(
            "Document language",
            options=label_list,
            key=WIDGET_KEY,
            help=(
                "Language of the source PDF. Used for OCR accuracy and TTS voice. "
                "Auto-detection runs on upload and updates this automatically."
            ),
        )
        return code_list[label_list.index(selected_label)]

    @staticmethod
    def _translation_settings(source_lang: str) -> tuple:
        """Translation toggle + engine selection. Returns (translate, target_lang, engine)."""
        translate = st.toggle(
            "Translate output",
            value=False,
            help="Translate extracted text before generating audio.",
        )

        target_lang  = source_lang
        trans_engine = "auto"

        if translate:
            langs        = Config.get_all_language_labels()
            label_list   = list(langs.values())
            code_list    = list(langs.keys())
            default_target = "en"
            default_idx    = code_list.index(default_target) if default_target in code_list else 0

            target_label = st.selectbox(
                "Translate to",
                options=label_list,
                index=default_idx,
                key="translation_target_lang",
            )
            target_lang = code_list[label_list.index(target_label)]

            # Engine status
            g_ok = Config.is_google_translate_available()
            l_ok = Config.is_libretranslate_available()
            st.caption("Engine status:")
            st.caption(f"{'🟢' if g_ok else '🔴'} Google Translate (free, unofficial API)")
            st.caption(f"{'🟢' if l_ok else '🔴'} LibreTranslate ({Config.LIBRETRANSLATE_URL})")
            st.caption(
                "Rate limits: Google — no hard limit for occasional use. "
                "LibreTranslate public — ~80 requests/hour × 4,500 chars each "
                "= ~360,000 chars/hour (~25–40 articles/hour)."
            )

            trans_engine = st.radio(
                "Translation engine",
                options=["auto", "google", "libretranslate"],
                captions=["Auto (recommended)", "Google only", "LibreTranslate only"],
                index=0,
                key="trans_engine_radio",
            )

            effective_source = st.session_state.get("detected_lang", source_lang) or source_lang
            if target_lang == effective_source:
                detected_label = Config.SUPPORTED_LANGUAGES.get(effective_source, {}).get("label", effective_source)
                target_label_str = Config.SUPPORTED_LANGUAGES.get(target_lang, {}).get("label", target_lang)
                st.caption(
                    f"Source ({detected_label}) and target ({target_label_str}) are the same "
                    f"— no translation will run."
                )
            elif effective_source != source_lang:
                detected_label = Config.SUPPORTED_LANGUAGES.get(effective_source, {}).get("label", effective_source)
                st.caption(
                    f"Source auto-detected as {detected_label}. "
                    f"Translation will run after language detection."
                )

        return translate, target_lang, trans_engine

    @staticmethod
    def _tts_engine_selector() -> str:
        openai_available = Config.is_openai_available()
        options = ["gtts"]
        captions = ["🟢 Free (gTTS) — set OPENAI_API_KEY to unlock Premium"]
        if openai_available:
            options.append("openai")
            captions.append("🟢 Premium: OpenAI tts-1 — natural voice, ~$0.015/1,000 chars.")
        else:
            captions[0] = "🟢 Free (gTTS) — set OPENAI_API_KEY to unlock Premium"

        engine = st.radio(
            "TTS Engine",
            options=options,
            captions=captions,
            label_visibility="collapsed",
        )
        return engine

    @staticmethod
    def _page_range_selector(max_pages: int) -> tuple[int, int]:
        effective_max = max(max_pages, 1)

        # Detect when the PDF changes (different page count) and reset
        # widget states. Critically: we must DELETE the widget keys from
        # session state before rendering, not just overwrite them —
        # Streamlit ignores ss writes to already-rendered widget keys
        # within the same script run.
        stored_max = st.session_state.get("_prs_last_max", 0)
        if effective_max != stored_max:
            st.session_state["_prs_last_max"] = effective_max
            # Delete keys so Streamlit treats them as fresh on next render
            for key in ("page_end_input", "page_start_input"):
                if key in st.session_state:
                    del st.session_state[key]
            # Set desired defaults — will be picked up since keys are fresh
            st.session_state["page_end_input"]   = effective_max
            st.session_state["page_start_input"] = 1

        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input(
                "From page",
                min_value=1,
                max_value=effective_max,
                step=1,
                key="page_start_input",
                help="First page to process (1-indexed, inclusive).",
            )
        with col2:
            end_page = st.number_input(
                "To page",
                min_value=1,
                max_value=effective_max,
                step=1,
                key="page_end_input",
                help=f"Last page to process. PDF has {effective_max} pages.",
            )

        selected = int(end_page) - int(start_page) + 1
        st.caption(
            f"{effective_max} pages total · {selected} selected "
            + ("· Full document" if selected == effective_max
               else f"· Pages {int(start_page)}–{int(end_page)}")
        )
        return int(start_page), int(end_page)

    @staticmethod
    def _feature_toggles() -> tuple:
        remove_hf = st.toggle(
            "Remove headers & footers",
            value=True,
            help=(
                "Detect and strip repeating header/footer text "
                "(running titles, page numbers, publisher info)."
            ),
        )
        show_analysis = st.toggle(
            "Show document analysis",
            value=True,
            help="Display NLP KPIs, readability scores, keyword density, and more.",
        )

        # Completeness verification runs automatically and is shown as an
        # informational report — content is never removed automatically.
        st.caption(
            "💡 Completeness fingerprinting runs automatically and is shown "
            "in the results — content is never removed."
        )

        # Return False for the removed toggles so callers don't break
        return remove_hf, show_analysis, False, False

    # ================================================================== #
    # SECTION 2 — FILE INPUT                                              #
    # ================================================================== #

    @staticmethod
    def file_uploader_widget():
        return st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload a PDF file to convert to audio.",
        )

    @staticmethod
    def local_file_selector(folder_path: str = ".") -> str | None:
        import os, glob
        pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
        if not pdf_files:
            st.info("No PDF files found in the current directory.")
            return None
        options   = ["— select —"] + [os.path.basename(f) for f in pdf_files]
        selected  = st.selectbox("Local PDF file", options, key="local_file_select")
        if selected == "— select —":
            return None
        return os.path.join(folder_path, selected)

    # ================================================================== #
    # SECTION 3 — PRE-PROCESSING SUMMARY                                  #
    # ================================================================== #

    @staticmethod
    def render_preprocessing_summary(
        size_mb:          float,
        page_count:       int,
        selected_pages:   int,
        chapter_list:     list[dict],
        est_audio_min:    float,
        est_time_range:   str,
        mode_hint:        str,
        warnings:         list[dict],
        start_page:       int,
        end_page:         int,
    ) -> None:
        with st.container(border=True):
            st.markdown("**📋 What you're about to process**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total pages",    page_count)
                st.metric("Selected pages", selected_pages)
            with c2:
                ch_count = len(chapter_list)
                st.metric("Chapters detected", ch_count if ch_count else "—")
                st.metric("Est. audio output", f"{est_audio_min:.0f} min")

            st.caption(
                f"Estimated processing time: {est_time_range} "
                "(depends on TTS engine and network). "
                "Keep this tab open — Streamlit does not process in the background."
            )

            if chapter_list:
                with st.expander(f"Chapters detected in pre-scan ({len(chapter_list)})"):
                    for ch in chapter_list:
                        p1, p2 = ch.get("start", 1), ch.get("end", page_count)
                        pg_str = f"pp. {p1}–{p2}" if p1 != p2 else f"p. {p1}"
                        st.markdown(f"✅ **{ch['title']}** — {pg_str}")

            for w in warnings:
                if w.get("extended"):
                    st.warning(
                        f"⚡ Chapter '{w['chapter']}' extends to p.{w['chapter_end']}. "
                        f"Page range auto-extended from {w['cut_at']} → {w['chapter_end']}."
                    )
                else:
                    st.info(
                        f"ℹ️ Chapter '{w['chapter']}' appears to continue beyond "
                        f"selected end page {w['cut_at']}."
                    )

    # ================================================================== #
    # SECTION 4 — STATUS BANNERS                                          #
    # ================================================================== #

    @staticmethod
    def render_extraction_mode_banner(mode: str, page_count: int) -> None:
        if mode == "text":
            st.success(
                f"✅ Text mode — native text extraction ({page_count} pages). "
                "Fast and accurate."
            )
        elif mode == "ocr":
            st.info(
                f"🔍 OCR mode — image-based extraction ({page_count} pages). "
                "Slower; accuracy depends on scan quality."
            )
        else:
            st.info(f"ℹ️ Extraction mode: {mode} ({page_count} pages)")

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

        if "original" in engine_used:
            st.warning(
                f"⚠️ Translation requested ({source_label} → {target_label}) but "
                f"both engines failed or returned unchanged text. "
                f"Audio will be generated in the original language. "
                f"Tip: try LibreTranslate or check your network connection."
            )
        else:
            st.success(
                f"🌐 Translated {chapters_translated}/{total_chapters} chapters — "
                f"{source_label} → {target_label} via {engine_name}"
            )

    @staticmethod
    def render_completeness_banner(
        incomplete_pages:    set,
        incomplete_chapters: set,
    ) -> None:
        """
        Completeness report — informational only. Content is NEVER removed.
        Fingerprint verification flags pages/chapters whose last words could
        not be matched in the assembled text (often references with URLs,
        last pages of well-formed PDFs, or multilingual documents).
        """
        if not incomplete_pages and not incomplete_chapters:
            st.success(
                "✅ Completeness verified — all pages and chapters passed "
                "fingerprint verification. Audio covers the full requested range."
            )
            return

        items = []
        if incomplete_pages:
            pg_nums = ", ".join(str(p + 1) for p in sorted(incomplete_pages))
            items.append(
                f"Page(s) {pg_nums} — fingerprint could not be verified "
                f"(often caused by reference/URL-heavy pages)."
            )
        if incomplete_chapters:
            ch_names = ", ".join(f"'{t}'" for t in sorted(incomplete_chapters))
            items.append(f"Chapter(s) {ch_names} — last page fingerprint unverified.")

        body = "ℹ️ Completeness note (audio is unaffected — full content included):\n\n"
        for item in items:
            body += f"• {item}\n"
        body += (
            "\nThis is informational only. No content was removed. "
            "If you see missing text in the audio, try disabling "
            "'Remove headers & footers' or check the page range."
        )
        st.info(body)

    # ================================================================== #
    # SECTION 5 — CHAPTER QUEUE                                           #
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
        Completed chapters get individual download buttons immediately.
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

        pdf_title = st.session_state.get("pdf_title", "")

        for title, text in chapters.items():
            status     = chapter_status.get(title, "queued")
            icon       = status_icons.get(status, "⬜")
            display    = title if len(title) <= 35 else title[:32] + "…"
            word_count = len(text.split()) if text else 0

            page_label = ""
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
                    q_fname     = UIComponents._make_filename(pdf_title, title) + ".mp3"
                    dur_secs    = len(audio_bytes) / 16_000.0
                    mins        = int(dur_secs) // 60
                    secs        = int(dur_secs) % 60
                    # Unique key: monotonic render_id prevents duplicate key error
                    # when queue is rendered multiple times (during + after pipeline)
                    render_id   = st.session_state.get("_queue_render_id", 0)
                    st.session_state["_queue_render_id"] = render_id + 1
                    st.download_button(
                        label=f"⬇️ {mins}:{secs:02d}",
                        data=audio_bytes,
                        file_name=q_fname,
                        mime="audio/mpeg",
                        key=f"dl_q_{render_id}_{hash(title) % 99999}",
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
    # SECTION 5B — AUDIO PLAYER + POST-GENERATION ADJUSTMENT              #
    # ================================================================== #

    @staticmethod
    def render_audio_player(
        audio_data:      dict   = None,
        current_chapter: str    = "",
        chapters:        OrderedDict = None,
        page_ranges:     dict   = None,
        # Legacy params (from older streamlit_app versions)
        audio_bytes:     bytes  = None,
        chapter_title:   str    = "",
        chapter_index:   int    = 0,
        total_chapters:  int    = 1,
    ) -> None:
        """Audio player for the current chapter. Accepts both old and new call signatures."""
        # Handle legacy call: render_audio_player(audio_bytes=..., chapter_title=...)
        if audio_bytes is not None:
            if not audio_bytes:
                st.info("Audio not yet generated for this chapter.")
                return
            st.audio(audio_bytes, format="audio/mp3")
            if chapter_title:
                st.caption(f"**{chapter_title}** · Chapter {chapter_index + 1} of {total_chapters}")
            return

        # New call: render_audio_player(audio_data=..., current_chapter=..., chapters=...)
        chapter_audio = (audio_data or {}).get(current_chapter, b"")
        if not chapter_audio:
            st.info("Audio not yet generated for this chapter.")
            return

        page_label = ""
        if page_ranges and current_chapter in page_ranges:
            p1, p2 = page_ranges[current_chapter]
            if p1 > 0:
                page_label = f" · pp. {p1}–{p2}" if p1 != p2 else f" · p. {p1}"

        word_count = len(((chapters or {}).get(current_chapter) or "").split())
        st.audio(chapter_audio, format="audio/mp3")
        st.caption(f"**{current_chapter}**{page_label} · {word_count:,} words")

    @staticmethod
    def render_audio_adjustment_panel(
        audio_data: dict,
        chapters:   OrderedDict,
        pdf_title:  str = "",
    ) -> None:
        """
        Post-generation speed/pitch adjustment panel.
        Always visible (not in expander) so users can see and interact with it.
        Sliders update session state → download buttons below re-render with
        adjusted audio in the same pass (no extra click needed).
        """
        st.markdown("**🎛️ Playback Adjustments**")

        adj_col1, adj_col2 = st.columns(2)
        with adj_col1:
            adj_speed = st.slider(
                "⚡ Speed",
                min_value=0.5, max_value=2.5,
                value=1.0,
                step=0.05, format="%.2f×",
                key="adj_speed_slider",
                help="1.0 = original. Changes tempo without re-generating audio.",
            )
        with adj_col2:
            adj_pitch = st.slider(
                "🎵 Pitch",
                min_value=0.5, max_value=2.0,
                value=1.0,
                step=0.05, format="%.2f×",
                key="adj_pitch_slider",
                help="1.0 = original. Shifts voice pitch up or down.",
            )

        speed_changed = abs(adj_speed - 1.0) > 0.02
        pitch_changed = abs(adj_pitch - 1.0) > 0.02
        if speed_changed or pitch_changed:
            tags = []
            if speed_changed:
                faster = adj_speed > 1.0
                tags.append(f"{adj_speed:.2f}× ({'faster' if faster else 'slower'})")
            if pitch_changed:
                higher = adj_pitch > 1.0
                tags.append(f"pitch {adj_pitch:.2f}× ({'higher' if higher else 'lower'})")
            st.caption(f"✏️ Active: {' · '.join(tags)} — reflected in downloads below")
        else:
            st.caption("Downloads use original audio. Adjust sliders to modify.")

    @staticmethod
    def _apply_audio_adjustments(audio_bytes: bytes) -> bytes:
        """Apply current speed/pitch settings to audio bytes. Returns adjusted bytes.
        
        Reads from widget session state keys directly — always reflects
        what the user currently has the sliders set to.
        """
        from audio_generator import AudioGenerator
        # Read from widget keys (most reliable — always in sync with slider)
        speed = float(st.session_state.get("adj_speed_slider", 1.0))
        pitch = float(st.session_state.get("adj_pitch_slider", 1.0))
        result = audio_bytes
        if abs(speed - 1.0) > 0.02:
            result = AudioGenerator.apply_speed(result, speed)
        if abs(pitch - 1.0) > 0.02:
            result = AudioGenerator.apply_pitch(result, pitch)
        return result

    # ================================================================== #
    # SECTION 5C — DOWNLOAD BUTTONS                                       #
    # ================================================================== #

    @staticmethod
    def _make_filename(pdf_title: str, chapter_title: str = "", suffix: str = "") -> str:
        """Build a clean, descriptive filename from PDF title + chapter."""
        def slug(s: str, max_len: int = 35) -> str:
            s = re.sub(r"[^a-zA-Z0-9 _-]", "", s).strip()
            s = re.sub(r" +", "_", s)
            return s[:max_len].rstrip("_")

        base = slug(pdf_title, 35) if pdf_title else "ReadMyPDF"
        if chapter_title and chapter_title.lower() not in ("document", "introduction"):
            ch = slug(chapter_title, 25)
            name = f"{base}__{ch}" if ch else base
        else:
            name = base
        if suffix:
            name = f"{name}{suffix}"
        return name or "ReadMyPDF"

    @staticmethod
    def render_download_buttons(
        audio_data:      dict,
        current_chapter: str,
        chapters:        OrderedDict,
        pdf_title:       str = "",
    ) -> None:
        """Chapter + Full Book download with adjustments applied."""
        st.markdown("**⬇️ Downloads**")
        st.caption(
            "Downloads apply current Speed/Pitch adjustments from the panel above."
        )
        col1, col2 = st.columns(2)

        with col1:
            chapter_audio = audio_data.get(current_chapter, b"")
            if chapter_audio:
                adjusted = UIComponents._apply_audio_adjustments(chapter_audio)
                fname = UIComponents._make_filename(pdf_title, current_chapter) + ".mp3"
                st.download_button(
                    label="📥 This Chapter",
                    data=adjusted,
                    file_name=fname,
                    mime="audio/mpeg",
                    use_container_width=True,
                    help=f"Download '{current_chapter}' as MP3",
                )
            else:
                st.button("📥 This Chapter", disabled=True, use_container_width=True)

        with col2:
            if audio_data:
                silence_gap = bytes([0xFF, 0xFB, 0x90, 0x00, *([0x00] * 40)]) * 32
                buf         = io.BytesIO()
                chapter_order = list(chapters.keys())

                for i, title in enumerate(chapter_order):
                    ch_audio = audio_data.get(title, b"")
                    if ch_audio:
                        buf.write(ch_audio)
                        if i < len(chapter_order) - 1:
                            buf.write(silence_gap)

                combined = buf.getvalue()
                if combined:
                    adjusted_full = UIComponents._apply_audio_adjustments(combined)
                    available     = len(audio_data)
                    total         = len(chapters)
                    label         = (
                        f"📚 Full Book ({available} of {total} chapters)"
                        if available < total
                        else f"📚 Full Book ({available} chapters)"
                    )
                    full_fname = UIComponents._make_filename(pdf_title, "", "_full.mp3")
                    st.download_button(
                        label=label,
                        data=adjusted_full,
                        file_name=full_fname,
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
                st.session_state.bookmarks.append(current_chapter)
                st.rerun()

    @staticmethod
    def render_bookmarks_panel(
        audio_data:    dict,
        bookmarks:     list,
        chapters:      OrderedDict,
        pdf_title:     str = "",
    ) -> None:
        if not bookmarks:
            return
        st.markdown("**🔖 Bookmarks**")
        for bm in bookmarks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"• {bm}")
            with col2:
                if bm in audio_data:
                    bm_audio  = UIComponents._apply_audio_adjustments(audio_data[bm])
                    bm_fname  = UIComponents._make_filename(pdf_title, bm) + "_bookmark.mp3"
                    render_id = st.session_state.get("_queue_render_id", 0)
                    st.session_state["_queue_render_id"] = render_id + 1
                    st.download_button(
                        label="⬇️",
                        data=bm_audio,
                        file_name=bm_fname,
                        mime="audio/mpeg",
                        key=f"dl_bm_{render_id}_{hash(bm) % 99999}",
                    )

    # ================================================================== #
    # SECTION 7 — DOCUMENT ANALYSIS                                       #
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
        st.markdown("---")
        st.subheader("📊 Document Intelligence")

        # Row 0: Quick KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Words", f"{word_count:,}")
        with m2:
            st.metric("Est. Listening Time", reading_time)
        if text_stats:
            with m3:
                st.metric("Unique Words", f"{text_stats.get('unique_words', 0):,}")
            with m4:
                st.metric(
                    "Vocabulary Richness",
                    f"{text_stats.get('vocabulary_richness', 0):.1f}%",
                )

        st.divider()

        # Row 1: Readability
        if readability:
            st.markdown("**📖 Readability**")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric(
                    "Flesch Ease",
                    f"{readability['flesch_ease']}/100",
                    help="0=Very Hard, 100=Very Easy",
                )
            with r2:
                st.metric("Grade Level", f"{readability['flesch_kincaid']}")
            with r3:
                st.metric("Reading Level", readability["grade_label"])

            ease  = readability["flesch_ease"]
            color = "🔴" if ease < 30 else ("🟡" if ease < 60 else "🟢")
            st.caption(f"{color} {readability['ease_label']} — {ease}/100 ease score")

            if text_stats:
                t1, t2, t3 = st.columns(3)
                with t1:
                    st.metric("Avg Sentence", f"{text_stats.get('avg_sentence_length', 0):.1f} words")
                with t2:
                    st.metric("Avg Word Length", f"{text_stats.get('avg_word_length', 0):.1f} chars")
                with t3:
                    st.metric("Sentences", f"{text_stats.get('total_sentences', 0):,}")
            st.divider()

        # Row 2: Content type
        if content_type:
            st.markdown("**🔍 Content Type**")
            ctype  = content_type.get("type", "General")
            cconf  = content_type.get("confidence", 0)
            scores = content_type.get("all_scores", {})
            icon_map = {
                "Academic":            "🎓",
                "Report / Analysis":   "📈",
                "Fiction / Narrative": "📖",
                "News / Article":      "📰",
                "Technical":           "⚙️",
                "Legal":               "⚖️",
                "General":             "📄",
            }
            st.metric("Detected Type", f"{icon_map.get(ctype, '📄')} {ctype}")
            st.caption(f"Confidence: {cconf}%")
            top_scores = sorted(scores.items(), key=lambda x: -x[1])[:3]
            for t, s in top_scores:
                if s > 0:
                    st.caption(f"• {t}: {s} signals")
            st.divider()

        # Row 3: Summary
        if summary:
            st.markdown("**📝 Summary**")
            st.write(summary)
            st.caption("Extractive — first 3 content sentences.")
            st.divider()

        # Row 4: Sentiment
        if sentiment:
            st.markdown("**🎭 Tone & Sentiment**")
            tone       = sentiment.get("label", "Neutral")
            conf       = sentiment.get("confidence", "low")
            pos_count  = sentiment.get("positive_count", 0)
            neg_count  = sentiment.get("negative_count", 0)
            tone_color = "🟢" if tone == "Positive" else ("🔴" if tone == "Negative" else "🟡")
            st.metric("Tone", f"{tone_color} {tone}")
            st.caption(f"Confidence: {conf}")
            st.caption(f"🟢 {pos_count} positive • 🔴 {neg_count} negative")
            st.divider()

        # Row 5: Topic density
        if topic_density:
            st.markdown("**🔑 Topic Density (occurrences per 1,000 words)**")
            for item in topic_density[:12]:
                word    = item["word"]
                density = item["density_per_1k"]
                count   = item["count"]
                bar_pct = item["bar_pct"]
                filled  = int(bar_pct / 5)
                empty   = 20 - filled
                bar     = "█" * filled + "░" * empty
                st.markdown(
                    f"`{word:<18}` `{bar}` {density:.2f}/1k (×{count})"
                )
            st.divider()

        # Row 6: Chapter complexity
        if chapter_complexity:
            st.markdown("**📊 Complexity by Chapter (hardest → easiest to follow while listening)**")
            for item in chapter_complexity:
                ease   = item["flesch_ease"]
                label  = item["ease_label"]
                title  = item["title"]
                wc     = item["word_count"]
                color  = "🔴" if ease < 30 else ("🟡" if ease < 60 else "🟢")
                short  = title if len(title) <= 40 else title[:37] + "…"
                st.caption(f"{color} **{short}** — {label} ({ease}/100) · {wc:,} words")
            st.divider()

        # Row 7: Lexical diversity
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
                          help="Words appearing exactly once")
            with ld4:
                st.metric("Avg Word Frequency", f"{lexical_diversity.get('avg_word_frequency', 0)}×")
            st.divider()

        # Row 8: Sentence complexity
        if sentence_complexity:
            st.markdown("**📏 Sentence Complexity** *(audio comprehension impact)*")
            long_pct = sentence_complexity.get("long_sentence_pct", 0)
            long_n   = sentence_complexity.get("long_sentence_count", 0)
            longest  = sentence_complexity.get("longest_sentence_words", 0)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Long Sentences (>30 words)", f"{long_n} ({long_pct}%)",
                          help="Hard to follow while listening")
            with sc2:
                st.metric("Longest Sentence", f"{longest} words")
            with sc3:
                st.metric("Short Fragments (<5 words)",
                          f"{sentence_complexity.get('short_sentence_count', 0)}")
            if long_pct > 20:
                sc_color = "🔴" if long_pct > 30 else "🟡"
                st.caption(
                    f"{sc_color} {long_pct}% of sentences are long (>30 words). "
                    "Consider using a slower playback speed for complex sections."
                )
            st.divider()

        # Row 9: Entities
        if characters:
            st.markdown("**👥 Recurring Names / Entities**")
            st.write("  •  ".join(characters[:20]))
            st.divider()

        # Row 10: Keywords
        if keywords:
            st.markdown("**💬 Top Keywords**")
            kw_parts = [f"**{w}** ({ct})" for w, ct in keywords[:15]]
            st.write("  •  ".join(kw_parts))


    # ================================================================== #
    # SECTION 5D — ADDITIONAL DISPLAY METHODS                             #
    # ================================================================== #

    @staticmethod
    def render_page_range_warning(
        original_end: int,
        extended_end: int,
        chapters_cut: list,
    ) -> None:
        """Banner when page range was auto-extended to avoid cutting a chapter."""
        if extended_end > original_end:
            ch_names = ", ".join(
                f"'{w.get("chapter", "")}'" for w in chapters_cut[:3]
            )
            st.info(
                f"⚡ Page range auto-extended: {original_end} → {extended_end} "
                f"to avoid cutting chapter(s): {ch_names}."
            )

    @staticmethod
    def render_completeness_report(
        incomplete_pages:    set,
        incomplete_chapters: set,
        total_pages:         int = 0,
        total_chapters:      int = 0,
    ) -> None:
        """Alias for render_completeness_banner."""
        UIComponents.render_completeness_banner(incomplete_pages, incomplete_chapters)

    @staticmethod
    def render_metadata(metadata: dict) -> None:
        """Show PDF metadata (title, author, etc.) if meaningful."""
        if not metadata:
            return
        title  = (metadata.get("title")  or "").strip()
        author = (metadata.get("author") or "").strip()
        if not title and not author:
            return
        parts = []
        if title:
            parts.append(f"**Title:** {title}")
        if author:
            parts.append(f"**Author:** {author}")
        st.caption("  ·  ".join(parts))

    @staticmethod
    def render_pdf_preview(pdf_bytes: bytes) -> None:
        """Show a PDF title extracted from metadata."""
        if not pdf_bytes:
            return
        try:
            import fitz
            doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
            meta  = doc.metadata
            doc.close()
            title = (meta.get("title") or "").strip()
            if title:
                st.caption(f"Title: {title}")
        except Exception:
            pass

    @staticmethod
    def render_cost_estimate(chapters: OrderedDict, tts_engine: str) -> None:
        """Show OpenAI cost estimate. No-op for gTTS (free)."""
        if tts_engine != "openai":
            return
        total_chars = sum(len(t) for t in chapters.values())
        cost_usd    = (total_chars / 1000) * 0.015  # $0.015 per 1k chars
        st.caption(
            f"💰 Estimated OpenAI TTS cost: ~${cost_usd:.3f} "
            f"({total_chars:,} chars × $0.015/1k)"
        )

    @staticmethod
    def render_table_of_contents(
        chapters:       OrderedDict,
        current_chapter: str,
        page_ranges:    dict = None,
        chapter_status: dict = None,
    ) -> None:
        """Clickable TOC in the left column."""
        st.markdown("**📑 Chapters**")
        status_icons = {"done": "✅", "processing": "⏳", "failed": "❌",
                        "queued": "⬜", "removed": "🗑️"}
        for title in chapters.keys():
            status  = (chapter_status or {}).get(title, "done")
            icon    = status_icons.get(status, "✅")
            display = title if len(title) <= 30 else title[:27] + "…"

            page_label = ""
            if page_ranges and title in page_ranges:
                p1, p2 = page_ranges[title]
                if p1 > 0:
                    page_label = f" · pp. {p1}–{p2}"

            word_count = len((chapters.get(title) or "").split())
            is_current = title == current_chapter

            if is_current:
                st.markdown(f"**▶ {icon} {display}**")
            else:
                if st.button(
                    f"{icon} {display}",
                    key=f"toc_{hash(title) % 99999}",
                    use_container_width=True,
                ):
                    st.session_state.current_chapter = title
                    st.rerun()
            st.caption(f"{page_label} · {word_count:,} words".lstrip(" · "))

    @staticmethod
    def render_bookmarks_list(chapters: OrderedDict) -> None:
        """List bookmarks with jump buttons."""
        bookmarks = st.session_state.get("bookmarks", [])
        audio_data = st.session_state.get("audio_data", {})
        pdf_title  = st.session_state.get("pdf_title", "")
        UIComponents.render_bookmarks_panel(
            audio_data=audio_data,
            bookmarks=bookmarks,
            chapters=chapters,
            pdf_title=pdf_title,
        )

    @staticmethod
    def render_chapter_navigation(chapters: OrderedDict, current_chapter: str) -> None:
        """Previous / Next chapter buttons."""
        chapter_list = list(chapters.keys())
        if len(chapter_list) <= 1:
            return
        idx    = chapter_list.index(current_chapter) if current_chapter in chapter_list else 0
        c1, c2 = st.columns(2)
        with c1:
            if idx > 0:
                if st.button("⬅️ Previous", use_container_width=True, key="nav_prev"):
                    st.session_state.current_chapter = chapter_list[idx - 1]
                    st.rerun()
        with c2:
            if idx < len(chapter_list) - 1:
                if st.button("Next ➡️", use_container_width=True, key="nav_next"):
                    st.session_state.current_chapter = chapter_list[idx + 1]
                    st.rerun()

    @staticmethod
    def render_progress(current_chapter: str, chapters: OrderedDict) -> None:
        """Show chapter N of M progress."""
        chapter_list = list(chapters.keys())
        if not chapter_list:
            return
        idx   = chapter_list.index(current_chapter) if current_chapter in chapter_list else 0
        total = len(chapter_list)
        st.caption(f"Chapter {idx + 1} of {total}")
        st.progress((idx + 1) / total)

    # ================================================================== #
    # SECTION 8 — FEEDBACK / ERROR RENDERING                              #
    # ================================================================== #

    @staticmethod
    def render_error(message: str) -> None:
        st.error(f"❌ {message}")

    @staticmethod
    def render_success(message: str) -> None:
        st.success(f"✅ {message}")

    @staticmethod
    def render_info(message: str) -> None:
        st.info(f"ℹ️ {message}")