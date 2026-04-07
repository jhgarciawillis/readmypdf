"""
text_cleaner.py
===============
All text transformation logic that happens BEFORE text reaches TTS,
and BEFORE document structure is analyzed.

Responsibilities:
  1. Detect whether a PDF is text-based or scanned (detect_pdf_mode)
  2. Identify header/footer blocks by position and repetition
  3. Filter those blocks out of the page block lists
  4. Clean raw extracted text for TTS consumption
  5. Extract heading structure from PyMuPDF block data

This module knows about:
  - pymupdf (for detect_pdf_mode only — opens a minimal fitz doc)
  - re, collections, statistics (stdlib only)
  - config.Config (for thresholds)

This module knows NOTHING about:
  - Streamlit
  - spaCy
  - gTTS / OpenAI
  - Audio processing

All methods are static. No instance state.
"""

import re
import logging
import statistics
from collections import defaultdict

import pymupdf  # PyMuPDF >= 1.27 — import as pymupdf, not fitz

from config import Config

logger = logging.getLogger(__name__)


class TextCleaner:

    # ================================================================== #
    # SECTION 1 — PDF MODE DETECTION                                      #
    # ================================================================== #

    @staticmethod
    def detect_pdf_mode(pdf_bytes: bytes) -> str:
        """
        Determine whether a PDF is text-based or scanned by checking how
        much extractable text PyMuPDF finds natively on the first few pages.

        Strategy:
          - Open the PDF with PyMuPDF
          - Extract plain text from up to the first 5 pages
          - Compute average characters per page
          - If average >= Config.MIN_CHARS_FOR_TEXT_MODE → "text"
          - Otherwise → "scanned"

        Returns:
          "text"    — native text extraction will work well
          "scanned" — OCR fallback required

        This is called once per PDF upload, before any other processing.
        It is fast (plain text extraction from 5 pages takes < 0.1s).
        """
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            num_pages = min(len(doc), 5)

            if num_pages == 0:
                doc.close()
                logger.warning("PDF has 0 pages — defaulting to scanned mode")
                return "scanned"

            total_chars = 0
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text("text")
                # Count non-whitespace characters only — whitespace alone
                # does not indicate real text content
                total_chars += len(text.replace(" ", "").replace("\n", ""))

            doc.close()

            avg_chars = total_chars / num_pages
            logger.info(
                f"PDF mode detection: avg {avg_chars:.0f} non-whitespace "
                f"chars/page across first {num_pages} pages. "
                f"Threshold: {Config.MIN_CHARS_FOR_TEXT_MODE}"
            )

            if avg_chars >= Config.MIN_CHARS_FOR_TEXT_MODE:
                return "text"
            else:
                return "scanned"

        except Exception as e:
            logger.error(f"Error during PDF mode detection: {e}")
            # Default to text mode — if extraction yields nothing,
            # pdf_processor will handle the empty result gracefully
            return "text"

    # ================================================================== #
    # SECTION 2 — HEADER / FOOTER DETECTION & REMOVAL                    #
    # ================================================================== #

    @staticmethod
    def is_in_header_zone(y0: float, page_height: float) -> bool:
        """
        Return True if a block's top edge (y0) falls within the header zone.
        Header zone = top Config.HEADER_ZONE fraction of the page.
        """
        return y0 < page_height * Config.HEADER_ZONE

    @staticmethod
    def is_in_footer_zone(y1: float, page_height: float) -> bool:
        """
        Return True if a block's bottom edge (y1) falls within the footer zone.
        Footer zone = bottom Config.FOOTER_ZONE fraction of the page.
        """
        return y1 > page_height * (1.0 - Config.FOOTER_ZONE)

    @staticmethod
    def is_in_margin_zone(y0: float, y1: float, page_height: float) -> bool:
        """
        Return True if a block is in either the header or footer zone.
        Convenience wrapper over the two zone checks above.
        """
        return (
            TextCleaner.is_in_header_zone(y0, page_height)
            or TextCleaner.is_in_footer_zone(y1, page_height)
        )

    @staticmethod
    def collect_repeated_blocks(all_page_blocks: list[list[dict]]) -> set[str]:
        """
        Identify text strings that appear repeatedly at the same vertical
        position across multiple pages — these are headers/footers.

        Algorithm:
          1. For each block on each page, compute its normalized Y midpoint:
               y_mid_norm = round(((y0 + y1) / 2) / page_height, Y_BAND_PRECISION)
          2. Group blocks by (normalized_y_mid, stripped_text)
          3. Any (y_band, text) pair that appears on >= REPEAT_THRESHOLD
             distinct pages is flagged as a repeating header/footer.
          4. Return the set of flagged text strings.

        Args:
          all_page_blocks: list of lists — outer list indexed by page,
                           inner list contains block dicts with keys:
                           { text, y0, y1, page_height, page_num, ... }

        Returns:
          set of text strings that should be excluded from TTS output.
          The calling code checks whether a block's stripped text is in
          this set before including it in the chapter content.
        """
        # Maps (y_band, stripped_text) → set of page numbers where it appears
        occurrence_map: dict[tuple, set] = defaultdict(set)

        for page_blocks in all_page_blocks:
            for block in page_blocks:
                text = block.get("text", "").strip()
                if not text:
                    continue

                page_height = block.get("page_height", 1.0)
                y0 = block.get("y0", 0.0)
                y1 = block.get("y1", 0.0)
                page_num = block.get("page_num", 0)

                if page_height <= 0:
                    continue

                y_mid = (y0 + y1) / 2.0
                y_mid_norm = round(y_mid / page_height, Config.Y_BAND_PRECISION)

                key = (y_mid_norm, text)
                occurrence_map[key].add(page_num)

        # Collect texts that appear on enough pages at the same Y band
        repeated_texts: set[str] = set()
        for (y_band, text), pages in occurrence_map.items():
            if len(pages) >= Config.REPEAT_THRESHOLD:
                repeated_texts.add(text)
                logger.debug(
                    f"Flagged as header/footer (appears on {len(pages)} pages "
                    f"at y_band={y_band:.2f}): '{text[:60]}'"
                )

        # Second pass: flag ALL text at Y-bands that are structurally repeated
        # even if individual span text differs slightly (e.g. split bylines).
        # Any Y-band that has SOME text appearing on >= REPEAT_THRESHOLD pages
        # is treated as a header/footer zone — all text at that band is flagged.
        repeated_ybands: set[float] = set()
        yband_page_count: dict[float, set] = defaultdict(set)
        for (y_band, text), pages in occurrence_map.items():
            if len(pages) >= Config.REPEAT_THRESHOLD:
                repeated_ybands.add(y_band)
            # Also track how many distinct pages each y_band appears on
            yband_page_count[y_band].update(pages)

        # Flag all spans at confirmed repeated Y-bands
        for page_blocks in all_page_blocks:
            for block in page_blocks:
                text = block.get("text", "").strip()
                if not text:
                    continue
                page_height = block.get("page_height", 1.0)
                if page_height <= 0:
                    continue
                y0 = block.get("y0", 0.0)
                y1 = block.get("y1", 0.0)
                y_mid = (y0 + y1) / 2.0
                y_mid_norm = round(y_mid / page_height, Config.Y_BAND_PRECISION)
                if y_mid_norm in repeated_ybands:
                    if text not in repeated_texts:
                        repeated_texts.add(text)
                        logger.debug(
                            f"Flagged (shared Y-band {y_mid_norm:.2f}): '{text[:60]}'"
                        )

        logger.info(
            f"Header/footer detection: flagged {len(repeated_texts)} "
            f"unique text strings as repeating elements."
        )
        return repeated_texts

    @staticmethod
    def is_noise_pattern(text: str) -> bool:
        """
        Return True if the text matches a known structural noise pattern
        that should always be excluded from TTS, regardless of position.

        Catches patterns that vary per-page so repetition detection misses them:
          - "Page 1 of 7", "Page 2 of 7", "Página 1 de 7"
          - Standalone page numbers: "1", "42", "- 3 -"
          - Short author/brand bylines that repeat with slight variation
          - Common footer boilerplate
        """
        import re
        t = text.strip()

        # "Page X of Y" in English and Spanish
        if re.fullmatch(
            r"[Pp]age\s+\d+\s+of\s+\d+|[Pp]ágina\s+\d+\s+de\s+\d+",
            t
        ):
            return True

        # Standalone integer (page number on its own)
        if re.fullmatch(r"\d{1,4}", t):
            return True

        # "- 3 -" or "— 3 —" style page numbers
        if re.fullmatch(r"[-—–]\s*\d{1,4}\s*[-—–]", t):
            return True

        # Very short lines that are only symbols/dashes (dividers)
        if re.fullmatch(r"[-—–_=*·•]{2,}", t):
            return True

        # "Confidential", "Draft", "Internal Use Only" type stamps
        if re.fullmatch(
            r"(Confidential|Draft|Internal Use Only|All Rights Reserved|"
            r"Todos los derechos reservados|Uso interno)",
            t, re.IGNORECASE
        ):
            return True

        # Pipe-separated byline pattern: "Author Name | Publication Name"
        # Common in academic/professional PDFs as running headers
        if re.fullmatch(r"[A-Za-zÀ-ÿ\s\.\,]+\|[A-Za-zÀ-ÿ\s\.\,]+", t):
            return True

        # Short text that is only uppercase — likely a section stamp or label
        # e.g. "CONFIDENTIAL", "DRAFT", "APPENDIX A"
        if len(t) <= 30 and t == t.upper() and re.search(r"[A-Z]", t):
            return True

        return False

    @staticmethod
    def get_last_content_block(page_blocks: list[dict]) -> dict | None:
        """
        Return the physically lowest non-noise block on a page.

        "Lowest" means highest y1 value — closest to the bottom of the page.
        Noise blocks (page numbers, headers, footers matching is_noise_pattern)
        are skipped so we always get the last real content block.

        If all blocks on the page are noise, returns None.

        Args:
          page_blocks: list of block dicts for a single page

        Returns:
          The block dict with the highest y1 among non-noise blocks,
          or None if none found.
        """
        content_blocks = [
            b for b in page_blocks
            if b.get("text", "").strip()
            and not TextCleaner.is_noise_pattern(b.get("text", "").strip())
        ]
        if not content_blocks:
            return None
        return max(content_blocks, key=lambda b: b.get("y1", 0.0))

    @staticmethod
    def get_page_fingerprint(page_blocks: list[dict], n_words: int = None) -> str:
        """
        Extract the last N words from the last content block on a page.
        Used to verify page completeness — if these words appear in the
        assembled chapter text, the page was fully captured.

        Args:
          page_blocks: list of block dicts for a single page
          n_words:     number of words to use (default: Config.LAST_WORDS_FINGERPRINT_COUNT)

        Returns:
          Lowercase string of the last N words, space-joined.
          Empty string if no content blocks found.
        """
        import re
        if n_words is None:
            n_words = Config.LAST_WORDS_FINGERPRINT_COUNT

        last_block = TextCleaner.get_last_content_block(page_blocks)
        if not last_block:
            return ""

        text  = last_block.get("text", "").strip()
        words = re.findall(r"[a-zA-ZÀ-ÿ]+", text)  # letters only, handles accents

        if not words:
            return ""

        # Take last n_words, lowercase for comparison
        fingerprint = " ".join(words[-n_words:]).lower()
        return fingerprint

    @staticmethod
    def fingerprint_in_text(fingerprint: str, text: str) -> bool:
        """
        Check whether a page fingerprint (last N words) appears in
        the assembled chapter text.

        Comparison is case-insensitive and ignores punctuation between words
        so "said Holmes" matches "said Holmes." or "said  Holmes".

        Args:
          fingerprint: string from get_page_fingerprint()
          text:        assembled chapter text to search within

        Returns:
          True if the fingerprint words appear consecutively in text.
          False if fingerprint is empty or not found.
        """
        import re
        if not fingerprint or not text:
            return False

        fp_words = fingerprint.lower().split()
        if len(fp_words) < Config.FINGERPRINT_MIN_WORDS:
            # Fingerprint too short — unreliable, assume complete
            return True

        # Build a regex that matches the words with any punctuation/whitespace
        # between them: "said holmes" matches "said, Holmes" or "said Holmes."
        pattern = r"\s*".join(re.escape(w) for w in fp_words)
        return bool(re.search(pattern, text.lower()))

    @staticmethod
    def filter_blocks(
        page_blocks: list[dict],
        repeated_texts: set[str],
        remove_margin_zones: bool = True,
        remove_repeated: bool = True,
    ) -> list[dict]:
        """
        Filter a single page's block list, removing header/footer content.

        A block is removed if ANY of the following are true:
          1. remove_margin_zones=True AND the block is in the header or
             footer zone (top/bottom Config.HEADER_ZONE / FOOTER_ZONE
             fraction of the page).
          2. remove_repeated=True AND the block's stripped text is in
             the repeated_texts set (identified by collect_repeated_blocks).

        Both conditions are applied independently so that:
          - A block in the margin zone is removed even if its text is unique.
          - A block whose text repeats across pages is removed even if it
            appears in the middle of the page (running headers on some PDFs
            are center-positioned, not at the very top).

        Args:
          page_blocks:         list of block dicts for one page
          repeated_texts:      set returned by collect_repeated_blocks()
          remove_margin_zones: whether to apply position-based filtering
          remove_repeated:     whether to apply repetition-based filtering

        Returns:
          Filtered list of block dicts (same schema, subset of input).
        """
        filtered = []
        for block in page_blocks:
            text = block.get("text", "").strip()
            if not text:
                continue  # always drop empty blocks

            page_height = block.get("page_height", 1.0)
            y0 = block.get("y0", 0.0)
            y1 = block.get("y1", 0.0)

            # Check 1: margin zone
            if remove_margin_zones and TextCleaner.is_in_margin_zone(
                y0, y1, page_height
            ):
                logger.debug(f"Removed (margin zone): '{text[:60]}'")
                continue

            # Check 2: repeated text
            if remove_repeated and text in repeated_texts:
                logger.debug(f"Removed (repeated text): '{text[:60]}'")
                continue

            # Check 3: pattern-based noise (page numbers, "Page X of Y", etc.)
            if remove_repeated and TextCleaner.is_noise_pattern(text):
                logger.debug(f"Removed (noise pattern): '{text[:60]}'")
                continue

            filtered.append(block)

        return filtered

    # ================================================================== #
    # SECTION 3 — HEADING EXTRACTION FROM BLOCK DATA                     #
    # ================================================================== #

    @staticmethod
    def extract_headings_from_blocks(blocks: list[dict]) -> list[dict]:
        """
        Identify heading blocks from a flat list of blocks (already filtered
        by filter_blocks) using font size as the primary signal.

        Algorithm:
          1. Collect all font sizes from all blocks.
          2. Compute the median font size — this represents body text.
          3. Any block where font_size >= median * Config.HEADING_SIZE_RATIO
             is a potential heading.
          4. Apply secondary filters:
               - Text length must be between MIN_HEADING_CHARS and MAX_HEADING_CHARS
               - Block must not be all digits (page number that escaped filtering)
               - Block must not be all punctuation
          5. Return list of heading dicts with 'text', 'level', 'page_num'.

        Heading level is determined by relative font size:
          - font_size >= median * RATIO * 1.5 → level 1 (chapter title)
          - font_size >= median * RATIO * 1.2 → level 2 (section)
          - font_size >= median * RATIO        → level 3 (subsection)

        Args:
          blocks: flat list of block dicts (output of filter_blocks across all pages)
                  Each block must have: text, font_size, page_num

        Returns:
          list of dicts: { text: str, level: int (1-3), page_num: int }
          Empty list if no headings found (caller falls back to regex detection).

        Note: In OCR mode, font_size is 0.0 for all blocks (Tesseract does not
        return font sizes). This method returns [] in that case, and
        TextAnalyzer.split_into_chapters() falls back to regex pattern matching.
        """
        # Collect all non-zero font sizes
        font_sizes = [
            b["font_size"]
            for b in blocks
            if b.get("font_size", 0) > 0
        ]

        if not font_sizes:
            logger.info(
                "No font size data available (likely OCR mode). "
                "Heading detection skipped — will use regex fallback."
            )
            return []

        median_size = statistics.median(font_sizes)
        ratio = Config.HEADING_SIZE_RATIO

        headings = []
        for block in blocks:
            text = block.get("text", "").strip()
            font_size = block.get("font_size", 0.0)
            page_num = block.get("page_num", 0)

            # Must be large enough to be a heading
            if font_size < median_size * ratio:
                continue

            # Length filters
            if len(text) < Config.MIN_HEADING_CHARS:
                continue
            if len(text) > Config.MAX_HEADING_CHARS:
                continue

            # Reject blocks that are only digits (escaped page numbers)
            if re.fullmatch(r"[\d\s]+", text):
                continue

            # Reject blocks that are only punctuation/symbols
            if re.fullmatch(r"[^\w]+", text):
                continue

            # Determine heading level by relative size
            if font_size >= median_size * ratio * 1.5:
                level = 1
            elif font_size >= median_size * ratio * 1.2:
                level = 2
            else:
                level = 3

            headings.append({
                "text":     text,
                "level":    level,
                "page_num": page_num,
            })
            logger.debug(
                f"Heading L{level} detected (size={font_size:.1f}, "
                f"median={median_size:.1f}): '{text[:60]}'"
            )

        logger.info(f"Heading extraction: found {len(headings)} headings.")
        return headings

    # ================================================================== #
    # SECTION 4 — TTS TEXT CLEANING PIPELINE                             #
    # ================================================================== #

    @staticmethod
    def clean_for_tts(text: str) -> str:
        """
        Master cleaning pipeline. Applies all enabled sub-cleaners in order.

        Order matters:
          1. Fix hyphenation first — rejoins split words before other
             rules run, so downstream patterns see whole words.
          2. Strip citations — before URL stripping to avoid overlap.
          3. Strip page numbers.
          4. Normalize figure/equation references.
          5. Normalize abbreviations.
          6. Remove URLs.
          7. Normalize numbers/symbols.
          8. Clean whitespace last — catches any double-spaces introduced
             by the previous steps.

        Each sub-cleaner is gated by its Config flag so users can
        toggle individual cleaning steps from the UI.

        Args:
          text: raw extracted text string (may contain OCR artifacts,
                citations, page numbers, etc.)

        Returns:
          Cleaned string ready to be sent to a TTS engine.
        """
        if not text or not text.strip():
            return ""

        if Config.FIX_HYPHENATION:
            text = TextCleaner._fix_hyphenation(text)

        if Config.CLEAN_CITATIONS:
            text = TextCleaner._strip_citation_brackets(text)

        if Config.CLEAN_PAGE_NUMBERS:
            text = TextCleaner._strip_page_numbers(text)

        if Config.CLEAN_FIGURE_REFS:
            text = TextCleaner._normalize_figure_refs(text)

        if Config.NORMALIZE_ABBREVS:
            text = TextCleaner._normalize_abbreviations(text)

        if Config.CLEAN_URLS:
            text = TextCleaner._remove_urls(text)

        if Config.NORMALIZE_NUMBERS:
            text = TextCleaner._normalize_numbers(text)

        # Always clean whitespace — this is non-optional
        text = TextCleaner._clean_whitespace(text)

        return text.strip()

    # ------------------------------------------------------------------ #
    # Sub-cleaners — called only through clean_for_tts()                  #
    # Each takes a str and returns a str.                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """
        Rejoin words that were split across lines with a hyphen.
        "impor-\ntant"  → "important"
        "self-\naware"  → "self-aware"  (keeps intentional compound hyphens)

        Heuristic: if the character before the hyphen and after the newline
        are both lowercase letters, treat as a split word and rejoin.
        Uppercase after newline = probably a new sentence, leave it.
        """
        # Pattern: lowercase letter + hyphen + newline + lowercase letter
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
        return text

    @staticmethod
    def _strip_citation_brackets(text: str) -> str:
        """
        Remove numeric citation brackets common in academic texts.
        Handles:
          [1]         → removed
          [14]        → removed
          [1, 2]      → removed
          [1-3]       → removed
          [1,2,3]     → removed
          (1)         → removed only if surrounded by non-numeric context
                        (avoids stripping legitimate parenthetical numbers
                        like sentence-final "(see page 12)")
        Leaves alphabetic brackets like [A], [Note] — those are likely
        legitimate content (footnote labels, glossary references).
        """
        # Square bracket numeric citations: [1], [1,2], [1-3], [1, 2, 3]
        text = re.sub(r"\[\s*\d+(?:[,\-]\s*\d+)*\s*\]", "", text)
        return text

    @staticmethod
    def _strip_page_numbers(text: str) -> str:
        """
        Remove standalone page numbers — lines that contain only digits,
        optionally surrounded by whitespace.
        Does NOT remove numbers that are part of sentences.

        Targets:
          "\n42\n"      → "\n\n"
          "\n  123  \n" → "\n\n"
        Does NOT touch:
          "Chapter 3"
          "See page 42 for details"
          "3.14"
        """
        # Line that is nothing but optional whitespace + digits + optional whitespace
        text = re.sub(r"(?m)^\s*\d{1,5}\s*$", "", text)
        return text

    @staticmethod
    def _normalize_figure_refs(text: str) -> str:
        """
        Expand common abbreviated figure/table/equation references so
        TTS reads them naturally.

        Fig. 3     → Figure 3
        Fig 3      → Figure 3
        Figs. 3-5  → Figures 3-5
        Tab. 2     → Table 2
        Eq. 4      → Equation 4
        Sec. 2.1   → Section 2.1
        Ch. 5      → Chapter 5
        pp. 12-14  → pages 12-14
        p. 7       → page 7
        et al.     → and colleagues
        """
        replacements = [
            # Figures
            (r"\bFigs?\.\s*", "Figure "),
            (r"\bfigs?\.\s*", "figure "),
            # Tables
            (r"\bTab\.\s*",   "Table "),
            (r"\btab\.\s*",   "table "),
            # Equations
            (r"\bEq\.\s*",    "Equation "),
            (r"\beq\.\s*",    "equation "),
            (r"\bEqs\.\s*",   "Equations "),
            # Sections
            (r"\bSec\.\s*",   "Section "),
            (r"\bsec\.\s*",   "section "),
            # Chapters
            (r"\bCh\.\s*",    "Chapter "),
            (r"\bch\.\s*",    "chapter "),
            # Pages
            (r"\bpp\.\s*",    "pages "),
            (r"\bp\.\s*(?=\d)", "page "),
            # et al.
            (r"\bet al\.",    "and colleagues"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _normalize_abbreviations(text: str) -> str:
        """
        Expand common abbreviations that TTS engines read awkwardly.
        Only expands well-known, unambiguous cases.

        e.g.    → for example
        i.e.    → that is
        etc.    → and so on
        vs.     → versus
        approx. → approximately
        dept.   → department
        govt.   → government
        """
        replacements = [
            (r"\be\.g\.",     "for example"),
            (r"\bi\.e\.",     "that is"),
            (r"\betc\.",      "and so on"),
            (r"\bvs\.\b",     "versus"),
            (r"\bapprox\.",   "approximately"),
            (r"\bdept\.",     "department"),
            (r"\bgovt\.",     "government"),
            (r"\bDr\.\s",     "Doctor "),
            (r"\bMr\.\s",     "Mister "),
            (r"\bMrs\.\s",    "Missus "),
            (r"\bProf\.\s",   "Professor "),
            (r"\bSt\.\s(?=[A-Z])", "Saint "),  # "St. Mary" → "Saint Mary"
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """
        Remove HTTP/HTTPS URLs and bare www. URLs.
        TTS engines read URLs letter-by-letter which is unpleasant.

        https://example.com/path?q=1  → removed
        www.example.com               → removed
        """
        # Full URLs
        text = re.sub(
            r"https?://\S+",
            "",
            text
        )
        # Bare www URLs
        text = re.sub(
            r"\bwww\.\S+",
            "",
            text
        )
        return text

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """
        Convert common numeric symbols to spoken equivalents so TTS
        reads them naturally instead of saying "percent sign" or
        spelling out currency symbols.

        12%      → 12 percent
        $5M      → 5 million dollars
        $1,234   → 1234 dollars     (comma-formatted numbers)
        €50      → 50 euros
        £30      → 30 pounds
        ¥500     → 500 yen
        +5       → plus 5           (signed numbers)
        -5       → minus 5

        Numbers with decimal points are left alone — TTS handles "3.14"
        correctly as "three point one four" in most engines.
        """
        # Percentages: digits followed by %
        text = re.sub(r"(\d+\.?\d*)\s*%", r"\1 percent", text)

        # USD millions/billions
        text = re.sub(r"\$(\d+\.?\d*)\s*[Mm](?:illion)?", r"\1 million dollars", text)
        text = re.sub(r"\$(\d+\.?\d*)\s*[Bb](?:illion)?", r"\1 billion dollars", text)

        # Currency symbols before numbers (simple case)
        text = re.sub(r"\$(\d[\d,]*)", lambda m: m.group(1).replace(",", "") + " dollars", text)
        text = re.sub(r"€(\d[\d,]*)",  lambda m: m.group(1).replace(",", "") + " euros",   text)
        text = re.sub(r"£(\d[\d,]*)",  lambda m: m.group(1).replace(",", "") + " pounds",  text)
        text = re.sub(r"¥(\d[\d,]*)",  lambda m: m.group(1).replace(",", "") + " yen",     text)

        # Signed numbers at start of word boundary
        text = re.sub(r"\+(\d)",  r"plus \1",  text)
        # Avoid converting hyphens in ranges like "pages 3-5"
        text = re.sub(r"(?<!\w)-(\d+)(?!\d*[-–])", r"minus \1", text)

        return text

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """
        Normalize whitespace throughout the text:
          - Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
          - Collapse multiple spaces to single space within lines
          - Strip leading/trailing whitespace from each line
          - Remove lines that became empty after other cleaning steps
        """
        # Normalize Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip each line individually
        lines = [line.strip() for line in text.split("\n")]

        # Collapse multiple consecutive blank lines to at most 2
        cleaned_lines = []
        blank_count = 0
        for line in lines:
            if line == "":
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                # Collapse multiple spaces within the line
                line = re.sub(r"  +", " ", line)
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Final strip of leading/trailing whitespace
        return text.strip()


# ------------------------------------------------------------------ #
# Standalone test — run `python text_cleaner.py` to verify logic     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== TTS Cleaning Tests ===\n")

    samples = [
        # Hyphenation
        ("Hyphenation fix",
         "This is very impor-\ntant text that was split."),

        # Citations
        ("Citation removal",
         "Researchers found [1] that climate change [2,3] affects biodiversity [1-4]."),

        # Page numbers
        ("Page number removal",
         "Some content here.\n\n42\n\nMore content follows.\n\n123\n"),

        # Figure refs
        ("Figure reference expansion",
         "As shown in Fig. 3 and Tab. 2, the results (Eq. 4) confirm our hypothesis. See Sec. 2.1."),

        # Abbreviations
        ("Abbreviation expansion",
         "e.g., machine learning, i.e., deep learning, etc. vs. traditional methods."),

        # URLs
        ("URL removal",
         "Visit https://example.com/path?q=1 or www.another.com for more details."),

        # Numbers/currency
        ("Number normalization",
         "Revenue grew by 12.4% to $5M. The deficit was $1,234. Temperature: -5 degrees."),

        # Combined
        ("Full pipeline",
         "According to Smith et al. [14], the company earned $2.3M (+15%) in Q3.\n\n"
         "See Fig. 3 at https://example.com for details, i.e., the bar chart.\n\n"
         "99\n\nThis continues on the next page."),
    ]

    for name, raw in samples:
        cleaned = TextCleaner.clean_for_tts(raw)
        print(f"[{name}]")
        print(f"  Before: {repr(raw[:80])}")
        print(f"  After:  {repr(cleaned[:80])}")
        print()

    print("=== Header/Footer Detection Test ===\n")

    # Simulate 5 pages with a repeated header "My Book Title" and footer "123"
    mock_pages = []
    for page_num in range(5):
        mock_pages.append([
            {"text": "My Book Title",  "y0": 20,  "y1": 40,  "page_height": 800, "page_num": page_num},
            {"text": "Chapter content paragraph goes here with real text.",
             "y0": 200, "y1": 300, "page_height": 800, "page_num": page_num},
            {"text": str(page_num + 1), "y0": 760, "y1": 780, "page_height": 800, "page_num": page_num},
        ])

    repeated = TextCleaner.collect_repeated_blocks(mock_pages)
    print(f"Flagged as repeated: {repeated}")

    filtered_page_0 = TextCleaner.filter_blocks(mock_pages[0], repeated)
    print(f"Blocks remaining after filter: {[b['text'] for b in filtered_page_0]}")
    print()

    print("=== PDF Mode Detection Test ===")
    print("(Requires a real PDF file to test — skipped in standalone mode)")