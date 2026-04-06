"""
text_analyzer.py
================
Takes structured block data from PDFProcessor and produces meaningful
document structure: chapters, headings, and NLP analysis.

Responsibilities:
  1. Build complete document structure from raw blocks
     (calls TextCleaner for filtering and heading detection)
  2. Split filtered blocks into chapters using heading structure
  3. Merge very short chapters into their predecessor
  4. Provide NLP analysis: keywords, character names, summary, sentiment
  5. Estimate reading time for TTS output

This module knows about:
  - spaCy (for NLP analysis)
  - TextCleaner (for filtering and heading extraction)
  - config.Config

This module knows NOTHING about:
  - Streamlit UI rendering
  - PDF extraction internals
  - Audio processing
  - gTTS / OpenAI

All methods are static. spaCy models are loaded with @st.cache_resource
keyed on the model name string (safe to cache — string arg, not file object).
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
    # SECTION 1 — spaCy MODEL MANAGEMENT                                  #
    # ================================================================== #

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_nlp_model(model_name: str):
        """
        Load and cache a spaCy language model by name.

        Uses @st.cache_resource (not @st.cache_data) because spaCy models
        are large objects that should be shared across reruns, not
        serialized and deserialized on every call.

        Args:
          model_name: spaCy model package name (e.g. "en_core_web_sm")
                      Must be installed via: python -m spacy download <name>

        Returns:
          spaCy Language object, or None if model is not installed.

        Callers must handle None gracefully — NLP features are optional.
        """
        try:
            import spacy
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}\n"
                f"NLP analysis features will be unavailable for this language."
            )
            return None
        except ImportError:
            logger.warning("spaCy is not installed. NLP features unavailable.")
            return None

    @staticmethod
    def get_nlp(lang_code: str):
        """
        Get the spaCy model for a given language code.

        Returns spaCy Language object or None.
        None means NLP features are gracefully disabled for this language.
        """
        model_name = Config.get_spacy_model(lang_code)
        if model_name is None:
            logger.info(
                f"No spaCy model configured for language '{lang_code}'. "
                f"NLP analysis skipped."
            )
            return None
        return TextAnalyzer._load_nlp_model(model_name)

    # ================================================================== #
    # SECTION 2 — MASTER DOCUMENT STRUCTURE BUILDER                      #
    # ================================================================== #

    @staticmethod
    def build_document_structure(
        blocks:              list[dict],
        lang_code:           str  = "en",
        remove_headers:      bool = True,
        repeated_texts:      Optional[set] = None,
    ) -> dict:
        """
        Master method. Takes raw blocks from PDFProcessor.extract() and
        returns the full document structure ready for TTS processing.

        Pipeline:
          Step 1 — Organise blocks by page (for header/footer detection)
          Step 2 — Collect repeated text strings (header/footer candidates)
          Step 3 — Filter each page's blocks (remove headers/footers)
          Step 4 — Flatten filtered blocks back to a single list
          Step 5 — Extract headings using font size analysis
          Step 6 — Tag heading blocks in the flat list
          Step 7 — Split into chapters by heading boundaries
          Step 8 — Merge very short chapters
          Step 9 — Build plain-text strings per chapter

        Args:
          blocks:         flat list of block dicts from PDFProcessor.extract()
          lang_code:      language code (for future NLP-based enhancements)
          remove_headers: whether to apply header/footer removal
          repeated_texts: pre-computed repeated text set (optional).
                          If None, it is computed here from blocks.
                          Pass it in if you computed it earlier to avoid
                          redundant work.

        Returns dict:
          {
            "chapters":  OrderedDict[str, str],  # title → plain text
            "headings":  list[dict],             # [{text, level, page_num}]
            "toc":       list[dict],             # [{title, level, page_num}]
          }
        """
        if not blocks:
            logger.warning("build_document_structure called with empty block list.")
            return {
                "chapters": OrderedDict({"Document": ""}),
                "headings": [],
                "toc":      [],
            }

        # Step 1 — Group blocks by page number
        pages_dict: dict[int, list[dict]] = {}
        for block in blocks:
            pn = block.get("page_num", 0)
            pages_dict.setdefault(pn, []).append(block)

        # Sort page numbers and build ordered list of pages
        sorted_page_nums = sorted(pages_dict.keys())
        all_page_blocks  = [pages_dict[pn] for pn in sorted_page_nums]

        # Step 2 — Collect repeated texts (headers/footers)
        if remove_headers and repeated_texts is None:
            repeated_texts = TextCleaner.collect_repeated_blocks(all_page_blocks)
        elif not remove_headers:
            repeated_texts = set()

        # Step 3 — Filter each page
        filtered_page_blocks = []
        for page_blocks in all_page_blocks:
            filtered = TextCleaner.filter_blocks(
                page_blocks,
                repeated_texts,
                remove_margin_zones=remove_headers,
                remove_repeated=remove_headers,
            )
            filtered_page_blocks.append(filtered)

        # Step 4 — Flatten back to single list (preserving page order)
        filtered_blocks = [
            block
            for page_blocks in filtered_page_blocks
            for block in page_blocks
        ]

        logger.info(
            f"Block filtering: {len(blocks)} → {len(filtered_blocks)} blocks "
            f"({len(blocks) - len(filtered_blocks)} removed as headers/footers)."
        )

        # Step 5 — Extract headings from filtered blocks
        headings = TextCleaner.extract_headings_from_blocks(filtered_blocks)

        # Step 6 — Tag heading blocks in the flat list
        # Build a lookup set of (page_num, text) for fast O(1) membership tests
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

        # Step 7 — Split into chapters
        chapters_raw = TextAnalyzer.split_into_chapters(filtered_blocks)

        # Step 8 — Merge very short chapters
        chapters_merged = TextAnalyzer.merge_short_chapters(
            chapters_raw,
            min_chars=Config.MIN_CHAPTER_CHARS,
        )

        # Step 9 — Build plain-text per chapter
        # At this point, chapter values are already strings (from split_into_chapters)
        # Ensure they are stripped
        chapters_final: OrderedDict[str, str] = OrderedDict()
        for title, text in chapters_merged.items():
            chapters_final[title] = text.strip()

        # Build flat TOC from detected headings
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
    # SECTION 3 — CHAPTER SPLITTING                                       #
    # ================================================================== #

    @staticmethod
    def split_into_chapters(filtered_blocks: list[dict]) -> OrderedDict:
        """
        Walk the filtered block list and split into chapters using heading
        blocks as chapter boundaries.

        Strategy:
          - Walk blocks in order (they are already page-ordered from the
            filtering step)
          - When a heading block (is_heading=True) is encountered, start
            a new chapter with that heading's text as the title
          - Accumulate body text blocks under the current chapter
          - Text before the first heading goes into "Introduction"
          - If no headings exist at all (OCR mode or plain document),
            fall back to the regex-based chapter detection

        Args:
          filtered_blocks: flat list of block dicts, heading blocks tagged

        Returns:
          OrderedDict mapping chapter_title (str) → chapter_text (str)
          Preserves insertion order (document order).
        """
        has_headings = any(b.get("is_heading", False) for b in filtered_blocks)

        if not has_headings:
            # Fallback: regex-based chapter detection on concatenated text
            logger.info(
                "No heading blocks found — using regex chapter detection fallback."
            )
            full_text = "\n".join(
                b.get("text", "") for b in filtered_blocks
            )
            return TextAnalyzer._split_by_regex(full_text)

        # Heading-based splitting
        chapters:      OrderedDict = OrderedDict()
        current_title: str         = "Introduction"
        current_text:  list[str]   = []

        for block in filtered_blocks:
            text = block.get("text", "").strip()
            if not text:
                continue

            if block.get("is_heading", False):
                # Save previous chapter if it has content
                if current_text:
                    existing = chapters.get(current_title, "")
                    chapters[current_title] = (
                        existing + "\n" + " ".join(current_text)
                    ).strip()
                    current_text = []

                # Handle duplicate heading titles by appending a suffix
                new_title = text
                if new_title in chapters:
                    suffix = 2
                    while f"{new_title} ({suffix})" in chapters:
                        suffix += 1
                    new_title = f"{new_title} ({suffix})"

                current_title = new_title

            else:
                current_text.append(text)

        # Flush the last chapter
        if current_text:
            existing = chapters.get(current_title, "")
            chapters[current_title] = (
                existing + "\n" + " ".join(current_text)
            ).strip()

        # Remove the "Introduction" entry if it ended up empty
        if "Introduction" in chapters and not chapters["Introduction"].strip():
            del chapters["Introduction"]

        # If nothing survived, return full text as one chapter
        if not chapters:
            full_text = "\n".join(b.get("text", "") for b in filtered_blocks)
            chapters["Document"] = full_text.strip()

        return chapters

    @staticmethod
    def _split_by_regex(text: str) -> OrderedDict:
        """
        Fallback chapter splitter using regex patterns on plain text.
        Used in OCR mode or when no font-size headings are detected.

        Detects common chapter heading patterns:
          - "Chapter 1", "CHAPTER 1"
          - "Part 1", "PART 1"
          - "1.", "2." (numbered sections at start of line)
          - "Section 1", "SECTION 1"

        Returns OrderedDict of chapter_title → chapter_text.
        If no patterns match, returns {"Document": full_text}.
        """
        chapter_patterns = [
            r"^(Chapter\s+\d+[:\.\s].{0,80})$",
            r"^(CHAPTER\s+\d+[:\.\s].{0,80})$",
            r"^(Part\s+\d+[:\.\s].{0,80})$",
            r"^(PART\s+\d+[:\.\s].{0,80})$",
            r"^(Section\s+\d+[\.\s].{0,80})$",
            r"^(SECTION\s+\d+[\.\s].{0,80})$",
            r"^(\d{1,2}\.\s+[A-Z].{2,80})$",   # "1. Introduction"
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
                # Save previous chapter
                content = "\n".join(current_lines).strip()
                if content:
                    existing = chapters.get(current_title, "")
                    chapters[current_title] = (existing + "\n" + content).strip()
                current_title = stripped
                current_lines = []
            else:
                if stripped:
                    current_lines.append(stripped)

        # Flush last chapter
        content = "\n".join(current_lines).strip()
        if content:
            existing = chapters.get(current_title, "")
            chapters[current_title] = (existing + "\n" + content).strip()

        # Clean up
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
        """
        Merge chapters that are too short into their predecessor.

        A chapter with fewer than min_chars of body text is appended to
        the previous chapter's content. Its title is lost — it becomes
        part of the previous chapter.

        This prevents 1-sentence chapters from becoming separate audio
        files with only 2 seconds of audio.

        Edge case: if the first chapter is too short and there is no
        predecessor, it is kept as-is (cannot merge backward).

        Args:
          chapters:  OrderedDict from split_into_chapters()
          min_chars: minimum character count to keep a chapter separate

        Returns:
          New OrderedDict with short chapters merged.
        """
        if len(chapters) <= 1:
            return chapters

        titles   = list(chapters.keys())
        texts    = list(chapters.values())
        merged_titles: list[str]  = [titles[0]]
        merged_texts:  list[str]  = [texts[0]]

        for i in range(1, len(titles)):
            title = titles[i]
            text  = texts[i]

            if len(text) < min_chars and merged_texts:
                # Append to previous chapter's text with a blank line separator
                merged_texts[-1] = merged_texts[-1] + "\n\n" + text
                logger.debug(
                    f"Merged short chapter '{title}' ({len(text)} chars) "
                    f"into '{merged_titles[-1]}'."
                )
            else:
                merged_titles.append(title)
                merged_texts.append(text)

        result: OrderedDict = OrderedDict()
        for title, text in zip(merged_titles, merged_texts):
            result[title] = text

        merged_count = len(titles) - len(result)
        if merged_count > 0:
            logger.info(f"Merged {merged_count} short chapter(s) into predecessors.")

        return result

    # ================================================================== #
    # SECTION 4 — NLP ANALYSIS                                            #
    # These methods are called on-demand from the UI analysis panel,      #
    # not as part of the main processing pipeline.                        #
    # ================================================================== #

    @staticmethod
    def extract_keywords(
        text:         str,
        lang_code:    str = "en",
        num_keywords: int = 10,
    ) -> list[tuple[str, int]]:
        """
        Extract the most frequent meaningful keywords from text using spaCy.

        Keeps: nouns (NOUN), proper nouns (PROPN), adjectives (ADJ)
        Drops: stop words, punctuation, whitespace tokens, numbers

        Args:
          text:         plain text string to analyse
          lang_code:    language code for spaCy model selection
          num_keywords: how many top keywords to return

        Returns:
          List of (word, frequency) tuples, sorted by frequency descending.
          Empty list if spaCy model unavailable.
        """
        nlp = TextAnalyzer.get_nlp(lang_code)
        if nlp is None:
            return []

        try:
            # Truncate very long texts for performance
            doc = nlp(text[:50_000])
            keywords = [
                token.lemma_.lower()
                for token in doc
                if (
                    not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and not token.like_num
                    and token.pos_ in ("NOUN", "PROPN", "ADJ")
                    and len(token.text) > 2
                )
            ]
            return Counter(keywords).most_common(num_keywords)
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    @staticmethod
    def detect_character_names(
        text:      str,
        lang_code: str = "en",
    ) -> list[str]:
        """
        Detect recurring character names using spaCy Named Entity Recognition.

        Identifies entities labelled PERSON that appear more than once.
        Single-occurrence names are excluded (likely misidentifications).

        Args:
          text:      plain text string
          lang_code: language code for spaCy model selection

        Returns:
          List of character name strings, sorted by frequency descending.
          Empty list if spaCy model unavailable.
        """
        nlp = TextAnalyzer.get_nlp(lang_code)
        if nlp is None:
            return []

        try:
            doc = nlp(text[:50_000])
            names = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "PERSON" and len(ent.text.strip()) > 1
            ]
            counts = Counter(names)
            # Return names that appear more than once, sorted by frequency
            return [
                name
                for name, count in counts.most_common()
                if count > 1
            ]
        except Exception as e:
            logger.error(f"Character name detection failed: {e}")
            return []

    @staticmethod
    def summarize_text(
        text:          str,
        lang_code:     str = "en",
        num_sentences: int = 3,
    ) -> str:
        """
        Produce a simple extractive summary by taking the first N sentences.

        This is an extractive approach (not generative) — it selects
        existing sentences rather than generating new text. This is
        intentional: we are honest about the limitation rather than
        hallucinating a summary.

        Uses spaCy's sentence segmenter for accurate sentence boundary
        detection (handles abbreviations like "Dr.", "e.g." correctly).

        Args:
          text:          plain text string
          lang_code:     language code for spaCy model selection
          num_sentences: number of sentences to include

        Returns:
          Summary string (N sentences joined by spaces).
          Falls back to first 500 characters if spaCy unavailable.
        """
        nlp = TextAnalyzer.get_nlp(lang_code)

        if nlp is None:
            # Simple fallback: first 500 chars, cropped to last full stop
            truncated = text[:500]
            last_stop = truncated.rfind(".")
            return truncated[: last_stop + 1] if last_stop > 0 else truncated

        try:
            doc = nlp(text[:20_000])
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return " ".join(sentences[:num_sentences])
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return text[:500]

    @staticmethod
    def sentiment_analysis(
        text:      str,
        lang_code: str = "en",
    ) -> dict:
        """
        Basic positive/negative/neutral sentiment classification.

        Uses a small curated lexicon of sentiment words. This is
        explicitly a lightweight heuristic, not an ML model.
        It is honest about its limitations — not presented as deep
        sentiment analysis.

        Args:
          text:      plain text string
          lang_code: language code (currently English lexicon only)

        Returns:
          dict: {
            "label":          "Positive" | "Negative" | "Neutral",
            "positive_count": int,
            "negative_count": int,
            "confidence":     "low" | "medium" | "high",
          }
        """
        # Lexicons — could be expanded significantly
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
            label      = "Neutral"
            confidence = "low"
        elif positive_count > negative_count:
            label      = "Positive"
            ratio      = positive_count / total_signal
            confidence = "high" if ratio > 0.7 else "medium"
        elif negative_count > positive_count:
            label      = "Negative"
            ratio      = negative_count / total_signal
            confidence = "high" if ratio > 0.7 else "medium"
        else:
            label      = "Neutral"
            confidence = "low"

        return {
            "label":          label,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "confidence":     confidence,
        }

    # ================================================================== #
    # SECTION 5 — UTILITIES                                               #
    # ================================================================== #

    @staticmethod
    def get_reading_time_estimate(text: str, wpm: int = None) -> str:
        """
        Estimate TTS playback time for a text string.

        Uses word count divided by TTS reading speed (words per minute).
        Default WPM is taken from Config.AUDIO_READING_WPM (150 WPM).

        Args:
          text: plain text string
          wpm:  words per minute override (uses Config default if None)

        Returns:
          Human-readable string, e.g. "3 min 42 sec"
        """
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
        """Return the number of words in a text string."""
        return len(text.split())

    @staticmethod
    def get_full_text(chapters: OrderedDict) -> str:
        """
        Concatenate all chapter texts into a single string.
        Used when NLP analysis should run over the whole document.

        Args:
          chapters: OrderedDict from build_document_structure()

        Returns:
          Single string with all chapters joined by double newlines.
        """
        return "\n\n".join(chapters.values())

    @staticmethod
    def split_text_on_sentences(
        text:      str,
        lang_code: str = "en",
        max_chars: int = None,
    ) -> list[str]:
        """
        Split a text string into sentence-boundary chunks, each under
        max_chars characters.

        Used by AudioGenerator to chunk text before sending to TTS engines,
        which have per-request character limits.

        Strategy:
          1. If spaCy is available, use its sentence segmenter (accurate)
          2. Otherwise, split on ". " (simple fallback)
          3. If a sentence is longer than max_chars, split on commas as
             a secondary boundary
          4. If still too long, hard-split at max_chars

        Args:
          text:      text to split
          lang_code: language code for spaCy model selection
          max_chars: maximum characters per chunk (default: Config.MAX_TTS_CHUNK_CHARS)

        Returns:
          List of text chunk strings, each ≤ max_chars characters.
        """
        if max_chars is None:
            max_chars = Config.MAX_TTS_CHUNK_CHARS

        if not text or not text.strip():
            return []

        # Step 1: Get sentences
        nlp = TextAnalyzer.get_nlp(lang_code)
        if nlp is not None:
            try:
                doc       = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = [s.strip() for s in text.split(". ") if s.strip()]
        else:
            sentences = [s.strip() for s in text.split(". ") if s.strip()]

        # Step 2: Group sentences into chunks ≤ max_chars
        chunks:         list[str] = []
        current_chunk:  list[str] = []
        current_length: int       = 0

        for sentence in sentences:
            # If a single sentence exceeds max_chars, break it further
            if len(sentence) > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk  = []
                    current_length = 0
                # Break the long sentence on commas
                parts = [p.strip() for p in sentence.split(",") if p.strip()]
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
                    # Hard split if still too long
                    while len(remainder) > max_chars:
                        chunks.append(remainder[:max_chars])
                        remainder = remainder[max_chars:]
                    if remainder:
                        chunks.append(remainder)
                continue

            # Normal case: accumulate sentences into chunk
            sentence_len = len(sentence) + 1  # +1 for the space separator

            if current_length + sentence_len > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk  = [sentence]
                current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        # Flush remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if c.strip()]


# ------------------------------------------------------------------ #
# Standalone test — run `python text_analyzer.py`                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_text = """
    Introduction

    This paper examines the effects of climate change on global food security.
    Rising temperatures and shifting rainfall patterns threaten crop yields worldwide.
    The consequences are particularly severe in sub-Saharan Africa and South Asia.

    Chapter 1: Background

    Climate change refers to long-term shifts in temperatures and weather patterns.
    These shifts may be natural, but since the 1800s, human activities have been
    the main driver. Burning fossil fuels generates greenhouse gas emissions that
    act as a blanket around the Earth, trapping heat.

    Chapter 2: Impact on Agriculture

    Global food production is highly sensitive to climate variability.
    Droughts, floods, and extreme heat events reduce crop yields significantly.
    According to recent studies [1,2], wheat yields could decline by up to 6% per
    degree Celsius of warming. See Fig. 3 for regional projections.

    Chapter 3: Policy Responses

    Governments worldwide are implementing adaptation strategies. e.g., drought-
    resistant crop varieties, improved irrigation systems, and early warning systems.
    The cost of inaction is estimated at $1.2T annually by 2050.
    """

    print("=== Sentence Splitting Test ===")
    chunks = TextAnalyzer.split_text_on_sentences(sample_text, lang_code="en", max_chars=200)
    print(f"Split into {len(chunks)} chunks:")
    for i, c in enumerate(chunks[:5]):
        print(f"  [{i+1}] ({len(c)} chars) {c[:80]}...")

    print("\n=== Reading Time Test ===")
    estimate = TextAnalyzer.get_reading_time_estimate(sample_text)
    words    = TextAnalyzer.get_word_count(sample_text)
    print(f"Word count: {words}, Estimated TTS time: {estimate}")

    print("\n=== Sentiment Analysis Test ===")
    sentiment = TextAnalyzer.sentiment_analysis(sample_text)
    print(f"Label: {sentiment['label']} (confidence: {sentiment['confidence']})")
    print(f"Positive signals: {sentiment['positive_count']}, "
          f"Negative signals: {sentiment['negative_count']}")

    print("\n=== Regex Chapter Split Test ===")
    chapters = TextAnalyzer._split_by_regex(sample_text)
    print(f"Detected {len(chapters)} chapters:")
    for title, text in chapters.items():
        print(f"  '{title}' — {len(text)} chars")

    print("\n=== Short Chapter Merge Test ===")
    from collections import OrderedDict as OD
    test_chapters = OD([
        ("Introduction", "Short intro."),
        ("Chapter 1", "This is a much longer chapter with a lot of content. " * 10),
        ("Chapter 2", "Tiny."),
        ("Chapter 3", "This chapter also has enough content to stand alone. " * 5),
    ])
    merged = TextAnalyzer.merge_short_chapters(test_chapters, min_chars=100)
    print(f"Before merge: {len(test_chapters)} chapters")
    print(f"After merge:  {len(merged)} chapters")
    for title, text in merged.items():
        print(f"  '{title}' — {len(text)} chars")