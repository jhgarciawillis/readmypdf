"""
text_analyzer.py
================
Document structure: heading detection, chapter splitting, merging.
NLP and KPI analysis delegated to document_analysis.py.
Completeness verification delegated to completeness.py.

This module is the primary interface for streamlit_app.py —
it re-exports all methods under the TextAnalyzer class so
callers don't need to know which module a method lives in.
"""

import logging
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from typing import Optional

import streamlit as st

from config import Config
from document_analysis import DocumentAnalysis
from completeness import CompletenessChecker

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Unified interface. Structure methods live here.
    Analysis + completeness methods delegate to sub-modules and are
    available as class methods for backward compatibility.
    """

    # ================================================================== #
    # SECTION 1 — DOCUMENT STRUCTURE                                      #
    # ================================================================== #

    @staticmethod
    def build_document_structure(
        blocks:         list[dict],
        lang_code:      str  = "en",
        remove_headers: bool = True,
    ) -> dict:
        """
        Build chapters, headings, and TOC from extracted blocks.
        Returns {chapters: OrderedDict, headings: list, toc: list}
        """
        from text_cleaner import TextCleaner

        if not blocks:
            return {"chapters": OrderedDict(), "headings": [], "toc": []}

        # Group by page
        page_groups: dict[int, list[dict]] = defaultdict(list)
        for block in blocks:
            page_groups[block.get("page_num", 0)].append(block)

        all_page_blocks = [page_groups[pn] for pn in sorted(page_groups.keys())]

        # Collect repeated header/footer strings
        repeated_texts: set[str] = set()
        if remove_headers:
            repeated_texts = TextCleaner.collect_repeated_blocks(all_page_blocks)

        # Filter each page
        filtered_blocks: list[dict] = []
        for pn in sorted(page_groups.keys()):
            pg_blocks = page_groups[pn]
            clean     = TextCleaner.filter_blocks(
                page_blocks=pg_blocks,
                repeated_texts=repeated_texts,
                remove_margin_zones=remove_headers,
                remove_repeated=remove_headers,
            )
            filtered_blocks.extend(clean)

        # Detect headings
        filtered_blocks = TextCleaner.extract_headings_from_blocks(filtered_blocks)

        # Split into chapters
        chapters = TextAnalyzer.split_into_chapters(filtered_blocks)
        chapters = TextAnalyzer.merge_short_chapters(chapters)

        headings = [
            b.get("text", "") for b in filtered_blocks
            if b.get("is_heading", False)
        ]

        return {
            "chapters": chapters,
            "headings": headings,
            "toc":      headings[:20],
        }

    @staticmethod
    def split_into_chapters(filtered_blocks: list[dict]) -> OrderedDict:
        """Split filtered blocks into chapters at heading boundaries."""
        chapters: OrderedDict = OrderedDict()

        heading_indices = [
            i for i, b in enumerate(filtered_blocks)
            if b.get("is_heading", False)
        ]

        if not heading_indices:
            full_text = " ".join(b.get("text", "") for b in filtered_blocks)
            return TextAnalyzer._split_by_regex(full_text)

        # Preamble before first heading
        if heading_indices[0] > 0:
            pre_text = " ".join(
                b.get("text", "") for b in filtered_blocks[:heading_indices[0]]
            ).strip()
            if len(pre_text) > Config.MIN_CHAPTER_CHARS:
                chapters["Introduction"] = pre_text

        # Chapters between headings
        for idx, heading_pos in enumerate(heading_indices):
            title = filtered_blocks[heading_pos].get("text", f"Chapter {idx+1}").strip()
            start = heading_pos + 1
            end   = heading_indices[idx+1] if idx+1 < len(heading_indices) else len(filtered_blocks)
            text  = " ".join(b.get("text", "") for b in filtered_blocks[start:end]).strip()

            if title and len(text) >= Config.MIN_CHAPTER_CHARS:
                chapters[title] = text
            elif title and chapters:
                last_key = list(chapters.keys())[-1]
                chapters[last_key] += f" {title} {text}"

        return chapters

    @staticmethod
    def _split_by_regex(text: str) -> OrderedDict:
        """Fallback: regex-based chapter splitting for flat/OCR PDFs."""
        chapters: OrderedDict = OrderedDict()
        pattern  = re.compile(
            r"(?:^|\n)(?:Chapter|CHAPTER|Part|PART|Section|SECTION|"
            r"Capítulo|CAPÍTULO|Sección|SECCIÓN)\s+\w+[^\n]*",
            re.MULTILINE,
        )
        splits = [(m.start(), m.group(0).strip()) for m in pattern.finditer(text)]

        if not splits:
            chapters["Document"] = text.strip()
            return chapters

        if splits[0][0] > 0:
            pre = text[:splits[0][0]].strip()
            if len(pre) > Config.MIN_CHAPTER_CHARS:
                chapters["Introduction"] = pre

        for i, (pos, title) in enumerate(splits):
            end  = splits[i+1][0] if i+1 < len(splits) else len(text)
            body = text[pos + len(title):end].strip()
            if title and len(body) >= Config.MIN_CHAPTER_CHARS:
                chapters[title] = body

        return chapters

    @staticmethod
    def merge_short_chapters(
        chapters:  OrderedDict,
        min_chars: int = None,
    ) -> OrderedDict:
        """Merge stub chapters (below min_chars) into preceding chapter."""
        if min_chars is None:
            min_chars = Config.MIN_CHAPTER_CHARS

        merged:   OrderedDict = OrderedDict()
        last_key: Optional[str] = None

        for title, text in chapters.items():
            if len(text) < min_chars and last_key is not None:
                merged[last_key] += f"\n\n{title}\n{text}"
            else:
                merged[title] = text
                last_key      = title

        return merged

    # ================================================================== #
    # SECTION 2 — UTILITIES                                               #
    # ================================================================== #

    @staticmethod
    def get_reading_time_estimate(text: str, wpm: int = None) -> str:
        if wpm is None:
            wpm = Config.AUDIO_READING_WPM
        words   = len(re.findall(r"\b\w+\b", text))
        minutes = words / wpm if wpm > 0 else 0
        mins    = int(minutes)
        secs    = int((minutes - mins) * 60)
        if mins == 0:
            return f"{secs} sec"
        return f"{mins} min {secs:02d} sec"

    @staticmethod
    def get_word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    @staticmethod
    def get_full_text(chapters: OrderedDict) -> str:
        return " ".join(chapters.values())

    @staticmethod
    def split_text_on_sentences(
        text:      str,
        lang_code: str = "en",
        max_chars: int = None,
    ) -> list[str]:
        """Split text into sentence-bounded chunks under max_chars."""
        if max_chars is None:
            max_chars = Config.MAX_TTS_CHUNK_CHARS

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks:  list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if current_len + len(sent) + 1 > max_chars and current:
                chunks.append(" ".join(current))
                current     = []
                current_len = 0
            if len(sent) > max_chars:
                words     = sent.split()
                sub_chunk = []
                sub_len   = 0
                for word in words:
                    if sub_len + len(word) + 1 > max_chars and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = []
                        sub_len   = 0
                    sub_chunk.append(word)
                    sub_len += len(word) + 1
                if sub_chunk:
                    current.extend(sub_chunk)
                    current_len += sub_len
            else:
                current.append(sent)
                current_len += len(sent) + 1

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]

    # ================================================================== #
    # SECTION 3 — DELEGATED: NLP ANALYSIS (→ document_analysis.py)       #
    # ================================================================== #

    @staticmethod
    def extract_keywords(text, lang_code="en", top_n=15):
        return DocumentAnalysis.extract_keywords(text, lang_code, top_n)

    @staticmethod
    def detect_character_names(text, lang_code="en"):
        return DocumentAnalysis.detect_character_names(text, lang_code)

    @staticmethod
    def summarize_text(text, lang_code="en", num_sentences=3):
        return DocumentAnalysis.summarize_text(text, lang_code, num_sentences)

    @staticmethod
    def sentiment_analysis(text, lang_code="en"):
        return DocumentAnalysis.sentiment_analysis(text, lang_code)

    @staticmethod
    def compute_readability(text):
        return DocumentAnalysis.compute_readability(text)

    @staticmethod
    def compute_text_stats(text, chapters):
        return DocumentAnalysis.compute_text_stats(text, chapters)

    @staticmethod
    def detect_content_type(text, chapters):
        return DocumentAnalysis.detect_content_type(text, chapters)

    @staticmethod
    def compute_topic_density(text, keywords):
        return DocumentAnalysis.compute_topic_density(text, keywords)

    @staticmethod
    def detect_language_complexity_by_chapter(chapters):
        return DocumentAnalysis.detect_language_complexity_by_chapter(chapters)

    @staticmethod
    def compute_lexical_diversity(text):
        return DocumentAnalysis.compute_lexical_diversity(text)

    @staticmethod
    def compute_sentence_complexity(text):
        return DocumentAnalysis.compute_sentence_complexity(text)

    # ================================================================== #
    # SECTION 4 — DELEGATED: COMPLETENESS (→ completeness.py)            #
    # ================================================================== #

    @staticmethod
    def build_chapter_page_map(blocks, chapters):
        return CompletenessChecker.build_chapter_page_map(blocks, chapters)

    @staticmethod
    def get_chapter_aware_page_range(heading_pages, requested_start, requested_end, total_pages):
        return CompletenessChecker.get_chapter_aware_page_range(
            heading_pages, requested_start, requested_end, total_pages
        )

    @staticmethod
    def flag_incomplete_pages(blocks, chapters):
        return CompletenessChecker.flag_incomplete_pages(blocks, chapters)

    @staticmethod
    def flag_incomplete_chapters(blocks, chapters):
        return CompletenessChecker.flag_incomplete_chapters(blocks, chapters)

    @staticmethod
    def get_page_range_per_chapter(blocks, chapters):
        return CompletenessChecker.get_page_range_per_chapter(blocks, chapters)