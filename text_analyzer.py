"""
text_analyzer.py
================
Takes structured block data from PDFProcessor and produces meaningful
document structure: chapters, headings, and basic analysis.

spaCy removed entirely — incompatible with Python 3.14 due to
Pydantic v1 / confection dependency chain. All NLP features that
required spaCy now use simple regex/stdlib fallbacks.

Sentence splitting uses '. ' boundary detection (fast, reliable enough
for TTS chunking purposes).
"""

import re
import logging
from collections import Counter, OrderedDict
from typing import Optional

import streamlit as st

from config import Config
from text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class TextAnalyzer:

    # ================================================================== #
    # SECTION 1 - DOCUMENT STRUCTURE BUILDER                             #
    # ================================================================== #

    @staticmethod
    def build_document_structure(
        blocks:         list[dict],
        lang_code:      str  = "en",
        remove_headers: bool = True,
        repeated_texts: Optional[set] = None,
    ) -> dict:
        """
        Takes raw blocks from PDFProcessor and returns full document structure.

        Pipeline:
          1. Group blocks by page
          2. Collect repeated header/footer text strings
          3. Filter each page's blocks
          4. Flatten filtered blocks
          5. Extract headings by font size
          6. Tag heading blocks
          7. Split into chapters
          8. Merge short chapters
        """
        if not blocks:
            logger.warning("build_document_structure called with empty block list.")
            return {
                "chapters": OrderedDict({"Document": ""}),
                "headings": [],
                "toc":      [],
            }

        # Step 1 - Group by page
        pages_dict: dict[int, list[dict]] = {}
        for block in blocks:
            pn = block.get("page_num", 0)
            pages_dict.setdefault(pn, []).append(block)

        sorted_page_nums = sorted(pages_dict.keys())
        all_page_blocks  = [pages_dict[pn] for pn in sorted_page_nums]

        # Step 2 - Collect repeated texts
        if remove_headers and repeated_texts is None:
            repeated_texts = TextCleaner.collect_repeated_blocks(all_page_blocks)
        elif not remove_headers:
            repeated_texts = set()

        # Step 3 - Filter each page
        filtered_page_blocks = []
        for page_blocks in all_page_blocks:
            filtered = TextCleaner.filter_blocks(
                page_blocks,
                repeated_texts,
                remove_margin_zones=remove_headers,
                remove_repeated=remove_headers,
            )
            filtered_page_blocks.append(filtered)

        # Step 4 - Flatten
        filtered_blocks = [
            block
            for page_blocks in filtered_page_blocks
            for block in page_blocks
        ]

        logger.info(
            f"Block filtering: {len(blocks)} -> {len(filtered_blocks)} blocks "
            f"({len(blocks) - len(filtered_blocks)} removed as headers/footers)."
        )

        # Step 5 - Extract headings
        headings = TextCleaner.extract_headings_from_blocks(filtered_blocks)

        # Step 6 - Tag heading blocks
        heading_lookup: set[tuple] = {
            (h["page_num"], h["text"]) for h in headings
        }
        heading_level_map: dict[tuple, int] = {
            (h["page_num"], h["text"]): h["level"] for h in headings
        }

        for block in filtered_blocks:
            key = (block.get("page_num", 0), block.get("text", "").strip())
            if key in heading_lookup:
                block["is_heading"]    = True
                block["heading_level"] = heading_level_map[key]

        # Step 7 - Split into chapters
        chapters_raw = TextAnalyzer.split_into_chapters(filtered_blocks)

        # Step 8 - Merge short chapters
        chapters_merged = TextAnalyzer.merge_short_chapters(
            chapters_raw,
            min_chars=Config.MIN_CHAPTER_CHARS,
        )

        chapters_final: OrderedDict[str, str] = OrderedDict()
        for title, text in chapters_merged.items():
            chapters_final[title] = text.strip()

        toc = [
            {"title": h["text"], "level": h["level"], "page_num": h["page_num"]}
            for h in headings
        ]

        logger.info(
            f"Document structure built: {len(chapters_final)} chapters, "
            f"{len(headings)} headings detected."
        )

        return {
            "chapters": chapters_final,
            "headings": headings,
            "toc":      toc,
        }

    # ================================================================== #
    # SECTION 2 - CHAPTER SPLITTING                                       #
    # ================================================================== #

    @staticmethod
    def split_into_chapters(filtered_blocks: list[dict]) -> OrderedDict:
        """Split block list into chapters using heading blocks as boundaries."""
        has_headings = any(b.get("is_heading", False) for b in filtered_blocks)

        if not has_headings:
            logger.info("No heading blocks found - using regex chapter detection.")
            full_text = "\n".join(b.get("text", "") for b in filtered_blocks)
            return TextAnalyzer._split_by_regex(full_text)

        chapters:      OrderedDict = OrderedDict()
        current_title: str         = "Introduction"
        current_text:  list[str]   = []

        for block in filtered_blocks:
            text = block.get("text", "").strip()
            if not text:
                continue

            if block.get("is_heading", False):
                if current_text:
                    existing = chapters.get(current_title, "")
                    chapters[current_title] = (
                        existing + "\n" + " ".join(current_text)
                    ).strip()
                    current_text = []

                new_title = text
                if new_title in chapters:
                    suffix = 2
                    while f"{new_title} ({suffix})" in chapters:
                        suffix += 1
                    new_title = f"{new_title} ({suffix})"

                current_title = new_title
            else:
                current_text.append(text)

        if current_text:
            existing = chapters.get(current_title, "")
            chapters[current_title] = (
                existing + "\n" + " ".join(current_text)
            ).strip()

        if "Introduction" in chapters and not chapters["Introduction"].strip():
            del chapters["Introduction"]

        if not chapters:
            full_text = "\n".join(b.get("text", "") for b in filtered_blocks)
            chapters["Document"] = full_text.strip()

        return chapters

    @staticmethod
    def _split_by_regex(text: str) -> OrderedDict:
        """Regex-based chapter splitter for OCR mode or no-heading documents."""
        chapter_patterns = [
            r"^(Chapter\s+\d+[:\.\s].{0,80})$",
            r"^(CHAPTER\s+\d+[:\.\s].{0,80})$",
            r"^(Part\s+\d+[:\.\s].{0,80})$",
            r"^(PART\s+\d+[:\.\s].{0,80})$",
            r"^(Section\s+\d+[\.\s].{0,80})$",
            r"^(SECTION\s+\d+[\.\s].{0,80})$",
            r"^(\d{1,2}\.\s+[A-Z].{2,80})$",
        ]
        combined = "|".join(chapter_patterns)
        lines    = text.split("\n")
        chapters: OrderedDict = OrderedDict()
        current_title = "Introduction"
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            match    = re.match(combined, stripped, re.MULTILINE)

            if match:
                content = "\n".join(current_lines).strip()
                if content:
                    existing = chapters.get(current_title, "")
                    chapters[current_title] = (existing + "\n" + content).strip()
                current_title = stripped
                current_lines = []
            else:
                if stripped:
                    current_lines.append(stripped)

        content = "\n".join(current_lines).strip()
        if content:
            existing = chapters.get(current_title, "")
            chapters[current_title] = (existing + "\n" + content).strip()

        if "Introduction" in chapters and not chapters["Introduction"].strip():
            del chapters["Introduction"]

        if not chapters:
            chapters["Document"] = text.strip()

        return chapters

    @staticmethod
    def merge_short_chapters(
        chapters:  OrderedDict,
        min_chars: int = 200,
    ) -> OrderedDict:
        """Merge chapters shorter than min_chars into their predecessor."""
        if len(chapters) <= 1:
            return chapters

        titles = list(chapters.keys())
        texts  = list(chapters.values())
        merged_titles: list[str] = [titles[0]]
        merged_texts:  list[str] = [texts[0]]

        for i in range(1, len(titles)):
            title = titles[i]
            text  = texts[i]

            if len(text) < min_chars and merged_texts:
                merged_texts[-1] = merged_texts[-1] + "\n\n" + text
                logger.debug(f"Merged short chapter '{title}' into '{merged_titles[-1]}'.")
            else:
                merged_titles.append(title)
                merged_texts.append(text)

        result: OrderedDict = OrderedDict()
        for title, text in zip(merged_titles, merged_texts):
            result[title] = text

        return result

    # ================================================================== #
    # SECTION 3 - NLP ANALYSIS (stdlib only, no spaCy)                  #
    # ================================================================== #

    @staticmethod
    def extract_keywords(
        text:         str,
        lang_code:    str = "en",
        num_keywords: int = 10,
    ) -> list[tuple[str, int]]:
        """
        Extract frequent words as keywords using simple tokenization.
        No spaCy — uses regex word extraction with a basic stopword list.
        """
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "that", "this",
            "these", "those", "it", "its", "as", "not", "also", "than", "then",
            "they", "their", "there", "we", "our", "you", "your", "he", "she",
            "his", "her", "i", "my", "me", "us", "which", "who", "what", "how",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in stopwords]
        return Counter(filtered).most_common(num_keywords)

    @staticmethod
    def detect_character_names(
        text:      str,
        lang_code: str = "en",
    ) -> list[str]:
        """
        Detect recurring capitalized names using regex heuristic.
        Finds capitalized words that appear 2+ times and are not
        common sentence starters.
        """
        # Find sequences of 1-3 capitalized words (likely proper names)
        candidates = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text
        )
        counts = Counter(candidates)
        # Filter out very common single-word false positives
        common_words = {
            "The", "This", "That", "These", "Those", "However", "Therefore",
            "Moreover", "Furthermore", "Although", "Because", "Since", "While",
        }
        return [
            name for name, count in counts.most_common(20)
            if count >= 2 and name not in common_words and len(name) > 3
        ]

    @staticmethod
    def summarize_text(
        text:          str,
        lang_code:     str = "en",
        num_sentences: int = 3,
    ) -> str:
        """
        Extractive summary: return first N sentences using period splitting.
        Simple but reliable without spaCy.
        """
        # Split on sentence-ending punctuation followed by space + capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return " ".join(sentences[:num_sentences])

    @staticmethod
    def sentiment_analysis(
        text:      str,
        lang_code: str = "en",
    ) -> dict:
        """Basic positive/negative/neutral sentiment using word lexicons."""
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "positive", "success", "successful", "effective", "improve",
            "improved", "benefit", "beneficial", "better", "best", "strong",
            "growth", "increase", "gain", "opportunity", "innovative",
            "leading", "significant", "outstanding", "achieve", "achieved",
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "disappointing",
            "negative", "failure", "fail", "failed", "weak", "decline",
            "decrease", "loss", "risk", "problem", "issue", "concern",
            "difficult", "challenge", "crisis", "threat", "damage",
            "harmful", "dangerous", "critical", "serious", "severe",
        }

        words          = re.findall(r"\b\w+\b", text.lower())
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        total_signal   = positive_count + negative_count

        if total_signal == 0:
            label, confidence = "Neutral", "low"
        elif positive_count > negative_count:
            label      = "Positive"
            ratio      = positive_count / total_signal
            confidence = "high" if ratio > 0.7 else "medium"
        elif negative_count > positive_count:
            label      = "Negative"
            ratio      = negative_count / total_signal
            confidence = "high" if ratio > 0.7 else "medium"
        else:
            label, confidence = "Neutral", "low"

        return {
            "label":          label,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "confidence":     confidence,
        }

    # ================================================================== #
    # SECTION 4 - UTILITIES                                              #
    # ================================================================== #

    @staticmethod
    def get_reading_time_estimate(text: str, wpm: int = None) -> str:
        """Estimate TTS listening time based on word count."""
        if wpm is None:
            wpm = Config.AUDIO_READING_WPM
        word_count    = len(text.split())
        total_seconds = int((word_count / wpm) * 60)
        minutes       = total_seconds // 60
        seconds       = total_seconds % 60
        if minutes == 0:
            return f"{seconds} sec"
        return f"{minutes} min {seconds} sec"

    @staticmethod
    def get_word_count(text: str) -> int:
        """Return word count of a text string."""
        return len(text.split())

    @staticmethod
    def get_full_text(chapters: OrderedDict) -> str:
        """Concatenate all chapter texts into a single string."""
        return "\n\n".join(chapters.values())

    @staticmethod
    def split_text_on_sentences(
        text:      str,
        lang_code: str = "en",
        max_chars: int = None,
    ) -> list[str]:
        """
        Split text into sentence-boundary chunks under max_chars.

        Uses regex sentence splitting (no spaCy). Splits on
        period/exclamation/question mark followed by whitespace + capital.
        Falls back to comma splitting then hard char-limit if needed.
        """
        if max_chars is None:
            max_chars = Config.MAX_TTS_CHUNK_CHARS

        if not text or not text.strip():
            return []

        # Split on sentence boundaries
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            sentences = [text.strip()]

        # Group into chunks <= max_chars
        chunks:         list[str] = []
        current_chunk:  list[str] = []
        current_length: int       = 0

        for sentence in sentences:
            # Handle sentences longer than max_chars
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk  = []
                    current_length = 0
                # Break on commas
                parts    = [p.strip() for p in sentence.split(",") if p.strip()]
                sub_chunk: list[str] = []
                sub_len:   int       = 0
                for part in parts:
                    if sub_len + len(part) + 2 > max_chars and sub_chunk:
                        chunks.append(", ".join(sub_chunk))
                        sub_chunk = []
                        sub_len   = 0
                    sub_chunk.append(part)
                    sub_len += len(part) + 2
                if sub_chunk:
                    remainder = ", ".join(sub_chunk)
                    while len(remainder) > max_chars:
                        chunks.append(remainder[:max_chars])
                        remainder = remainder[max_chars:]
                    if remainder:
                        chunks.append(remainder)
                continue

            sentence_len = len(sentence) + 1
            if current_length + sentence_len > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk  = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]