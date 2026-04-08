"""
completeness.py
===============
Page and chapter completeness verification via word fingerprinting.

Core idea: the last N words on a page's last content block are extracted
as a fingerprint. If those words appear in the assembled chapter text,
the page was fully captured. If not, the page (and potentially its chapter)
was truncated.

Also provides:
  - build_chapter_page_map: two-pass structural scan
  - get_chapter_aware_page_range: prevention layer (pre-extraction)
  - get_page_range_per_chapter: 1-indexed ranges for UI display
"""

import logging
import re
from collections import OrderedDict, defaultdict

from config import Config

logger = logging.getLogger(__name__)


class CompletenessChecker:
    """Static methods for completeness verification and prevention."""

    # ================================================================== #
    # SECTION 1 — CHAPTER PAGE MAP                                        #
    # ================================================================== #

    @staticmethod
    def build_chapter_page_map(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> dict[str, tuple[int, int]]:
        """
        Two-pass scan: map each chapter title to (first_page, last_page) 0-indexed.

        Pass 1: find heading block positions → chapter start pages.
        Pass 2: chapter N ends on the page before chapter N+1 starts.
                Last chapter ends at last page of document.

        Article fallback: no headings found → whole doc is one range.
        """
        if not blocks or not chapters:
            return {}

        chapter_titles = list(chapters.keys())
        all_pages      = sorted(set(b.get("page_num", 0) for b in blocks))
        last_page      = max(all_pages) if all_pages else 0
        first_page     = min(all_pages) if all_pages else 0

        # Pass 1: match heading blocks to chapter titles
        chapter_start_pages: dict[str, int] = {}
        for block in blocks:
            if not block.get("is_heading", False):
                continue
            block_text = block.get("text", "").strip()
            page_num   = block.get("page_num", 0)
            for title in chapter_titles:
                if title in chapter_start_pages:
                    continue
                if block_text == title or block_text in title or title in block_text:
                    chapter_start_pages[title] = page_num
                    break

        # Fallback: no headings → map all to full range
        if not chapter_start_pages:
            for title in chapter_titles:
                chapter_start_pages[title] = first_page

        # Pass 2: derive end pages
        sorted_chapters = sorted(chapter_start_pages.items(), key=lambda x: x[1])
        page_map: dict[str, tuple[int, int]] = {}

        for i, (title, start_pg) in enumerate(sorted_chapters):
            if i + 1 < len(sorted_chapters):
                end_pg = max(start_pg, sorted_chapters[i + 1][1] - 1)
            else:
                end_pg = last_page
            page_map[title] = (start_pg, end_pg)

        # Fill any unmatched chapters
        for title in chapter_titles:
            if title not in page_map:
                page_map[title] = (first_page, last_page)

        return page_map

    # ================================================================== #
    # SECTION 2 — PREVENTION LAYER                                        #
    # ================================================================== #

    @staticmethod
    def get_chapter_aware_page_range(
        heading_pages:   list[dict],
        requested_start: int,
        requested_end:   int,
        total_pages:     int,
    ) -> tuple[int, list[dict]]:
        """
        Prevention: check if requested page range cuts through a chapter.

        Uses the fast pre-scan heading list from PDFProcessor.get_heading_pages()
        to detect chapter boundaries before full extraction.

        Returns:
          (recommended_end, warnings_list)
          recommended_end: possibly extended page number (1-indexed)
          warnings: [{type, chapter, cut_at, chapter_end, extended}]
        """
        if not heading_pages:
            return requested_end, []

        # Deduplicate: keep highest font-size heading per page
        seen: dict[int, dict] = {}
        for h in sorted(heading_pages, key=lambda x: x["page_num"]):
            pn = h["page_num"]
            if pn not in seen or h["font_size"] > seen[pn]["font_size"]:
                seen[pn] = h
        unique = sorted(seen.values(), key=lambda x: x["page_num"])

        # Build chapter ranges (1-indexed for comparison with requested range)
        chapters_found = []
        for i, h in enumerate(unique):
            ch_start = h["page_num"] + 1
            ch_end   = unique[i + 1]["page_num"] if i + 1 < len(unique) else total_pages
            chapters_found.append({
                "title": h["text"],
                "start": ch_start,
                "end":   ch_end,
            })

        # Check for cuts
        warnings  = []
        final_end = requested_end

        for ch in chapters_found:
            if ch["start"] > requested_end:
                break
            if ch["start"] < requested_start:
                continue
            if ch["start"] <= requested_end < ch["end"]:
                warning = {
                    "type":        "chapter_cut",
                    "chapter":     ch["title"],
                    "cut_at":      requested_end,
                    "chapter_end": ch["end"],
                    "extended":    Config.AUTO_EXTEND_PAGE_RANGE,
                }
                warnings.append(warning)
                if Config.AUTO_EXTEND_PAGE_RANGE:
                    final_end = max(final_end, ch["end"])
                    logger.info(
                        f"Prevention: auto-extended end page "
                        f"{requested_end} → {final_end} for '{ch['title']}'"
                    )

        return final_end, warnings

    # ================================================================== #
    # SECTION 3 — POST-HOC FINGERPRINT VERIFICATION                       #
    # ================================================================== #

    @staticmethod
    def flag_incomplete_pages(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> set[int]:
        """
        Verify each chapter's last page by word fingerprinting.

        For each chapter, extracts the last N words from the last
        non-noise content block on its last page. If those words
        appear in the assembled chapter text → complete.
        If not found → page was truncated.

        Returns set of 0-indexed page numbers considered incomplete.
        """
        from text_cleaner import TextCleaner

        if not blocks or not chapters:
            return set()

        page_blocks_map: dict[int, list[dict]] = defaultdict(list)
        for block in blocks:
            page_blocks_map[block.get("page_num", 0)].append(block)

        page_map   = CompletenessChecker.build_chapter_page_map(blocks, chapters)
        incomplete = set()

        for title, chapter_text in chapters.items():
            if not chapter_text.strip() or title not in page_map:
                continue

            _start, last_pg = page_map[title]
            pg_blocks       = page_blocks_map.get(last_pg, [])

            if not pg_blocks:
                continue

            fingerprint = TextCleaner.get_page_fingerprint(pg_blocks)
            if not fingerprint:
                continue

            if not TextCleaner.fingerprint_in_text(fingerprint, chapter_text):
                incomplete.add(last_pg)
                logger.info(
                    f"Page {last_pg+1} flagged incomplete for '{title}': "
                    f"fingerprint '{fingerprint}' not found."
                )

        logger.info(f"Post-hoc page verification: {len(incomplete)} incomplete pages.")
        return incomplete

    @staticmethod
    def flag_incomplete_chapters(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> set[str]:
        """
        Chapter-level completeness: flags chapters whose last page
        failed fingerprint verification, or whose text ends mid-word.

        Returns set of chapter title strings considered incomplete.
        """
        if not blocks or not chapters:
            return set()

        incomplete_pages = CompletenessChecker.flag_incomplete_pages(blocks, chapters)
        page_map         = CompletenessChecker.build_chapter_page_map(blocks, chapters)
        incomplete_chaps = set()

        for title, chapter_text in chapters.items():
            if not chapter_text.strip():
                incomplete_chaps.add(title)
                continue

            # Page-level check
            if title in page_map:
                _start, last_pg = page_map[title]
                if last_pg in incomplete_pages:
                    incomplete_chaps.add(title)
                    continue

            # Text ends mid-word
            last_char = ""
            for ch in reversed(chapter_text):
                if ch.strip():
                    last_char = ch
                    break
            if last_char and last_char.isalpha():
                incomplete_chaps.add(title)

        logger.info(
            f"Post-hoc chapter verification: {len(incomplete_chaps)} incomplete."
        )
        return incomplete_chaps

    # ================================================================== #
    # SECTION 4 — PAGE RANGES FOR UI                                      #
    # ================================================================== #

    @staticmethod
    def get_page_range_per_chapter(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> dict[str, tuple[int, int]]:
        """1-indexed page ranges per chapter for TOC display."""
        raw = CompletenessChecker.build_chapter_page_map(blocks, chapters)
        return {title: (s + 1, e + 1) for title, (s, e) in raw.items()}