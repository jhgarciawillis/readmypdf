"""
text_analyzer.py
================
All text analysis. Takes extracted blocks and produces document structure,
NLP insights, completeness verification, and chapter/page maps.

Sections:
  1 — Document structure (headings, chapters, filtering)
  2 — NLP analysis (keywords, sentiment, summary, characters)
  3 — Extended analysis (readability, stats, content type, topic density)
  4 — Completeness verification (fingerprint-based page and chapter detection)
  5 — Utilities (word count, reading time, text helpers)
"""

import logging
import re
from collections import Counter, OrderedDict, defaultdict
from typing import Optional

import streamlit as st

from config import Config

logger = logging.getLogger(__name__)


class TextAnalyzer:

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
        Build the full document structure from extracted blocks.

        Pipeline:
          1. Collect repeated header/footer strings
          2. Filter blocks (remove margins, repeated text, noise)
          3. Detect headings by font size
          4. Split into chapters at heading boundaries
          5. Merge suspiciously short chapters
          6. Extract headings list and TOC

        Returns:
          {
            "chapters":  OrderedDict[title -> text],
            "headings":  list[str],
            "toc":       list[str],
          }
        """
        from text_cleaner import TextCleaner

        if not blocks:
            return {"chapters": OrderedDict(), "headings": [], "toc": []}

        # Step 1: collect repeated header/footer strings across all pages
        page_groups: dict[int, list[dict]] = defaultdict(list)
        for block in blocks:
            page_groups[block.get("page_num", 0)].append(block)

        all_page_blocks = [page_groups[pn] for pn in sorted(page_groups.keys())]

        repeated_texts: set[str] = set()
        if remove_headers:
            repeated_texts = TextCleaner.collect_repeated_blocks(all_page_blocks)

        # Step 2: filter each page's blocks
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

        # Step 3: detect headings by font size
        filtered_blocks = TextCleaner.extract_headings_from_blocks(filtered_blocks)

        # Step 4: split into chapters
        chapters = TextAnalyzer.split_into_chapters(filtered_blocks)

        # Step 5: merge short chapters
        chapters = TextAnalyzer.merge_short_chapters(chapters)

        # Step 6: extract heading list
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
        """
        Split filtered blocks into chapters at heading boundaries.
        Falls back to regex-based splitting if no headings detected.
        """
        chapters: OrderedDict = OrderedDict()

        heading_indices = [
            i for i, b in enumerate(filtered_blocks)
            if b.get("is_heading", False)
        ]

        if not heading_indices:
            # No headings — try regex split on the full text
            full_text = " ".join(b.get("text", "") for b in filtered_blocks)
            return TextAnalyzer._split_by_regex(full_text)

        # Chapter before first heading (preamble)
        if heading_indices[0] > 0:
            pre_text = " ".join(
                b.get("text", "") for b in filtered_blocks[:heading_indices[0]]
            ).strip()
            if pre_text:
                chapters["Introduction"] = pre_text

        # Chapters between headings
        for idx, heading_pos in enumerate(heading_indices):
            title = filtered_blocks[heading_pos].get("text", f"Chapter {idx + 1}").strip()
            start = heading_pos + 1
            end   = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(filtered_blocks)
            text  = " ".join(
                b.get("text", "") for b in filtered_blocks[start:end]
            ).strip()
            if title and text:
                chapters[title] = text
            elif title and not text:
                # Heading with no body — merge into previous or skip
                if chapters:
                    last_key = list(chapters.keys())[-1]
                    chapters[last_key] += f" {title}"

        return chapters

    @staticmethod
    def _split_by_regex(text: str) -> OrderedDict:
        """
        Fallback chapter splitting using common chapter heading patterns.
        Used when no font-size headings are detected (OCR mode, flat PDFs).
        """
        chapters: OrderedDict = OrderedDict()
        pattern = re.compile(
            r"(?:^|\n)(?:Chapter|CHAPTER|Part|PART|Section|SECTION)\s+\w+[^\n]*",
            re.MULTILINE,
        )
        splits = [(m.start(), m.group(0).strip()) for m in pattern.finditer(text)]

        if not splits:
            chapters["Document"] = text.strip()
            return chapters

        if splits[0][0] > 0:
            chapters["Introduction"] = text[:splits[0][0]].strip()

        for i, (pos, title) in enumerate(splits):
            end  = splits[i + 1][0] if i + 1 < len(splits) else len(text)
            body = text[pos + len(title):end].strip()
            if title and body:
                chapters[title] = body

        return chapters

    @staticmethod
    def merge_short_chapters(
        chapters:  OrderedDict,
        min_chars: int = None,
    ) -> OrderedDict:
        """
        Merge chapters below min_chars into the preceding chapter.
        Prevents TTS from generating a separate audio file for
        one-paragraph stub chapters.
        """
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
    # SECTION 2 — NLP ANALYSIS                                           #
    # ================================================================== #

    @staticmethod
    def extract_keywords(text: str, lang_code: str = "en", top_n: int = 15) -> list:
        """
        Extract top keywords by frequency, filtering stopwords.
        Returns list of (word, count) tuples.
        """
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "that", "this",
            "these", "those", "it", "its", "we", "they", "he", "she", "you",
            "i", "my", "our", "their", "his", "her", "not", "no", "if", "as",
            "than", "then", "so", "what", "which", "who", "how", "when",
            "where", "why", "all", "also", "more", "into", "about", "can",
            "just", "up", "out", "one", "each", "most", "some", "such",
        }
        words   = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
        counts  = Counter(w for w in words if w not in stopwords)
        return counts.most_common(top_n)

    @staticmethod
    def detect_character_names(text: str, lang_code: str = "en") -> list:
        """
        Detect recurring capitalized names (likely character names or entities).
        Returns list of name strings.
        """
        # Find capitalized words that appear multiple times
        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        counts     = Counter(candidates)
        # Filter out common sentence-starting words and short words
        non_names  = {
            "The", "This", "That", "These", "Those", "There", "Their",
            "They", "When", "Where", "What", "Which", "While", "With",
            "From", "Have", "Some", "Many", "Most", "More", "Such",
            "Also", "After", "Before", "During", "Under", "Over",
        }
        return [
            name for name, count in counts.most_common(20)
            if count >= 2 and name not in non_names and len(name) > 2
        ]

    @staticmethod
    def summarize_text(
        text:          str,
        lang_code:     str = "en",
        num_sentences: int = 3,
    ) -> str:
        """
        Extractive summary: first N content sentences.
        Skips noise lines: bylines, headers, short metadata lines.
        """
        from text_cleaner import TextCleaner
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        content   = []
        for s in sentences:
            s = s.strip()
            if len(s) < 30:
                continue
            if TextCleaner.is_noise_pattern(s):
                continue
            # Skip pipe-separated bylines: "Name | Role | Date"
            if s.count("|") >= 1 and len(s) < 100:
                continue
            # Skip authorship lines
            if re.match(r"^By\s+[A-Z]", s):
                continue
            # Skip numbered reference lines
            if re.match(r"^\d+\.\s+[A-Z]", s) and len(s) < 120:
                continue
            content.append(s)
            if len(content) >= num_sentences:
                break
        return " ".join(content)

    @staticmethod
    def sentiment_analysis(text: str, lang_code: str = "en") -> dict:
        """
        Simple lexicon-based sentiment scoring.
        Returns dict with label, confidence, positive_count, negative_count.
        """
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "outstanding", "success", "successful", "effective", "improve",
            "improvement", "benefit", "positive", "strong", "better", "best",
            "innovative", "opportunity", "growth", "achieve", "achievement",
            "valuable", "significant", "important", "clear", "proven", "trust",
        }
        negative_words = {
            "bad", "poor", "terrible", "awful", "failure", "fail", "failed",
            "problem", "issue", "risk", "danger", "loss", "decline", "weak",
            "difficult", "hard", "challenge", "concern", "worry", "threat",
            "ineffective", "wrong", "error", "mistake", "negative", "worse",
            "worst", "crisis", "collapse", "shutdown", "deny", "denial",
        }
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        pos   = sum(1 for w in words if w in positive_words)
        neg   = sum(1 for w in words if w in negative_words)
        total = pos + neg

        if total == 0:
            return {"label": "Neutral", "confidence": "low", "positive_count": 0, "negative_count": 0}

        ratio = pos / total
        if ratio > 0.6:
            label = "Positive"
        elif ratio < 0.4:
            label = "Negative"
        else:
            label = "Neutral"

        if abs(pos - neg) > total * 0.3:
            confidence = "high"
        elif abs(pos - neg) > total * 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "label":          label,
            "confidence":     confidence,
            "positive_count": pos,
            "negative_count": neg,
        }

    # ================================================================== #
    # SECTION 3 — EXTENDED ANALYSIS                                       #
    # ================================================================== #

    @staticmethod
    def compute_readability(text: str) -> dict:
        """
        Flesch Reading Ease and Flesch-Kincaid Grade Level.
        All stdlib — no external dependencies.
        """
        def count_syllables(word: str) -> int:
            word = word.lower().strip(".,!?;:'\"")
            if len(word) <= 3:
                return 1
            vowels = re.findall(r"[aeiouy]+", word)
            count  = len(vowels)
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        sentences     = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words         = re.findall(r"\b[a-zA-Z]+\b", text)

        if not sentences or not words:
            return {
                "flesch_ease": 0, "flesch_kincaid": 0,
                "grade_label": "Unknown", "ease_label": "Unknown",
            }

        num_sentences = len(sentences)
        num_words     = len(words)
        num_syllables = sum(count_syllables(w) for w in words)

        if num_sentences == 0 or num_words == 0:
            return {
                "flesch_ease": 0, "flesch_kincaid": 0,
                "grade_label": "Unknown", "ease_label": "Unknown",
            }

        asl            = num_words / num_sentences
        asw            = num_syllables / num_words
        flesch_ease    = round(max(0, min(100, 206.835 - 1.015 * asl - 84.6 * asw)), 1)
        flesch_kincaid = round(max(0, 0.39 * asl + 11.8 * asw - 15.59), 1)

        if flesch_ease >= 90:
            ease_label = "Very Easy"
        elif flesch_ease >= 70:
            ease_label = "Easy"
        elif flesch_ease >= 60:
            ease_label = "Standard"
        elif flesch_ease >= 50:
            ease_label = "Fairly Difficult"
        elif flesch_ease >= 30:
            ease_label = "Difficult"
        else:
            ease_label = "Very Difficult"

        if flesch_kincaid <= 6:
            grade_label = "Elementary"
        elif flesch_kincaid <= 9:
            grade_label = "Middle School"
        elif flesch_kincaid <= 12:
            grade_label = "High School"
        elif flesch_kincaid <= 16:
            grade_label = "University"
        else:
            grade_label = "Post-Graduate"

        return {
            "flesch_ease":      flesch_ease,
            "flesch_kincaid":   flesch_kincaid,
            "grade_label":      grade_label,
            "ease_label":       ease_label,
        }

    @staticmethod
    def compute_text_stats(text: str, chapters: OrderedDict) -> dict:
        """
        Vocabulary richness, sentence/word length, chapter distribution.
        """
        words         = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        sentences     = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        paras         = [p for p in re.split(r"\n\n+", text) if p.strip()]
        total_words   = len(words)
        unique_words  = len(set(words))
        vocab_rich    = round(unique_words / total_words * 100, 1) if total_words > 0 else 0
        avg_sent_len  = round(total_words / len(sentences), 1) if sentences else 0
        avg_word_len  = round(sum(len(w) for w in words) / total_words, 1) if total_words > 0 else 0

        chapter_dist  = []
        for title, ch_text in chapters.items():
            ch_words = len(re.findall(r"\b[a-zA-Z]+\b", ch_text))
            pct      = round(ch_words / total_words * 100, 1) if total_words > 0 else 0
            chapter_dist.append({"title": title, "word_count": ch_words, "pct": pct})

        longest  = max(chapter_dist, key=lambda x: x["word_count"]) if chapter_dist else {"title": "N/A", "word_count": 0}
        shortest = min(chapter_dist, key=lambda x: x["word_count"]) if chapter_dist else {"title": "N/A", "word_count": 0}

        return {
            "total_words":          total_words,
            "unique_words":         unique_words,
            "vocabulary_richness":  vocab_rich,
            "avg_sentence_length":  avg_sent_len,
            "avg_word_length":      avg_word_len,
            "total_sentences":      len(sentences),
            "paragraphs":           len(paras),
            "longest_chapter":      longest,
            "shortest_chapter":     shortest,
            "chapter_distribution": chapter_dist,
        }

    @staticmethod
    def detect_content_type(text: str, chapters: OrderedDict) -> dict:
        """
        Classify document as Academic / Report / Fiction / News / Technical / Legal.
        """
        full_lower = text.lower()
        scores     = {
            "Academic":            0,
            "Report / Analysis":   0,
            "Fiction / Narrative": 0,
            "News / Article":      0,
            "Technical":           0,
            "Legal":               0,
        }

        for term in ["abstract", "methodology", "hypothesis", "conclusion", "references",
                     "et al", "study", "research", "findings", "analysis", "literature"]:
            scores["Academic"] += full_lower.count(term)

        for term in ["framework", "kpi", "revenue", "market", "strategy", "growth",
                     "forecast", "quarter", "fiscal", "benchmark", "insight", "pattern",
                     "failure", "rate"]:
            scores["Report / Analysis"] += full_lower.count(term)

        for term in ["said", "whispered", "shouted", "felt", "thought", "walked",
                     "smiled", "chapter", "protagonist"]:
            scores["Fiction / Narrative"] += full_lower.count(term)
        scores["Fiction / Narrative"] += len(re.findall(r'"[^"]{5,}"', text)) * 3

        for term in ["according to", "reported", "announced", "government", "officials",
                     "sources said", "press release"]:
            scores["News / Article"] += full_lower.count(term)

        for term in ["function", "algorithm", "implementation", "api", "configuration",
                     "parameter", "database", "module", "variable", "syntax"]:
            scores["Technical"] += full_lower.count(term)
        scores["Technical"] += len(re.findall(r"[a-z_]+\([^)]*\)", text)) * 2

        for term in ["whereas", "hereby", "pursuant", "notwithstanding", "jurisdiction",
                     "liability", "clause", "agreement", "parties", "indemnify"]:
            scores["Legal"] += full_lower.count(term)

        top_type  = max(scores, key=scores.get)
        top_score = scores[top_type]
        total     = sum(scores.values()) or 1
        confidence = round(top_score / total * 100)

        if top_score < 5:
            top_type   = "General"
            confidence = 0

        return {
            "type":       top_type,
            "confidence": int(confidence),
            "all_scores": scores,
        }

    @staticmethod
    def compute_topic_density(text: str, keywords: list) -> list:
        """
        Keyword occurrences per 1,000 words with relative bar percentages.
        """
        total_words = len(re.findall(r"\b\w+\b", text))
        if not keywords or total_words == 0:
            return []

        max_count = keywords[0][1] if keywords else 1
        result    = []
        for word, count in keywords[:15]:
            density = round(count / total_words * 1000, 2)
            bar_pct = round(count / max_count * 100)
            result.append({
                "word":           word,
                "count":          count,
                "density_per_1k": density,
                "bar_pct":        bar_pct,
            })
        return result

    @staticmethod
    def detect_language_complexity_by_chapter(chapters: OrderedDict) -> list:
        """
        Flesch reading ease per chapter, sorted hardest to easiest.
        """
        results = []
        for title, text in chapters.items():
            if not text.strip():
                continue
            r  = TextAnalyzer.compute_readability(text)
            wc = len(re.findall(r"\b\w+\b", text))
            results.append({
                "title":       title,
                "flesch_ease": r["flesch_ease"],
                "ease_label":  r["ease_label"],
                "word_count":  wc,
            })
        return sorted(results, key=lambda x: x["flesch_ease"])

    # ================================================================== #
    # SECTION 4 — COMPLETENESS VERIFICATION                               #
    # ================================================================== #

    @staticmethod
    def build_chapter_page_map(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> dict[str, tuple[int, int]]:
        """
        Two-pass structural scan to map each chapter to its exact page range.

        Pass 1: find the page number where each chapter's heading appears.
        Pass 2: chapter N ends on the page before chapter N+1 starts.
                The last chapter ends on the last page of the document.

        Works for articles (no headings → whole document is one range)
        and books (multiple chapters) identically.

        Returns:
          Dict: chapter_title -> (first_page_0indexed, last_page_0indexed)
        """
        if not blocks or not chapters:
            return {}

        chapter_titles = list(chapters.keys())
        all_pages      = sorted(set(b.get("page_num", 0) for b in blocks))
        last_page      = max(all_pages) if all_pages else 0
        first_page     = min(all_pages) if all_pages else 0

        # Pass 1: find start page for each chapter via heading blocks
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

        # Article fallback: no headings found
        if not chapter_start_pages:
            for title in chapter_titles:
                chapter_start_pages[title] = first_page

        # Pass 2: derive end pages
        page_map:        dict[str, tuple[int, int]] = {}
        sorted_chapters  = sorted(chapter_start_pages.items(), key=lambda x: x[1])

        for i, (title, start_pg) in enumerate(sorted_chapters):
            if i + 1 < len(sorted_chapters):
                next_start = sorted_chapters[i + 1][1]
                end_pg     = max(start_pg, next_start - 1)
            else:
                end_pg = last_page
            page_map[title] = (start_pg, end_pg)

        # Fill any unmatched chapters
        for title in chapter_titles:
            if title not in page_map:
                page_map[title] = (first_page, last_page)

        return page_map

    @staticmethod
    def get_chapter_aware_page_range(
        heading_pages:   list[dict],
        requested_start: int,
        requested_end:   int,
        total_pages:     int,
    ) -> tuple[int, list[dict]]:
        """
        Prevention layer: check if the requested page range cuts through
        a chapter and return the recommended end page.

        Uses the heading_pages pre-scan from PDFProcessor.get_heading_pages()
        to determine chapter boundaries without doing a full extraction.

        Args:
          heading_pages:   list of {text, page_num, font_size} from pre-scan
          requested_start: user-selected start page (1-indexed)
          requested_end:   user-selected end page (1-indexed)
          total_pages:     total pages in PDF

        Returns:
          Tuple of:
            recommended_end: int (1-indexed) — possibly extended
            warnings:        list of dicts describing what was detected:
              {
                "type":        "chapter_cut" | "ok",
                "chapter":     chapter title that gets cut (if any),
                "cut_at":      page where cut happens (1-indexed),
                "chapter_end": natural end of that chapter (1-indexed),
                "extended":    bool — whether we auto-extended
              }
        """
        if not heading_pages:
            # No structure detectable — can't warn about cuts
            return requested_end, []

        # Convert heading pages to 1-indexed chapter boundaries
        # Each heading starts a chapter; it ends when the next heading starts
        chapters_found = []
        sorted_headings = sorted(heading_pages, key=lambda h: h["page_num"])

        # Deduplicate headings on same page (take the largest font size)
        seen_pages: dict[int, dict] = {}
        for h in sorted_headings:
            pn = h["page_num"]
            if pn not in seen_pages or h["font_size"] > seen_pages[pn]["font_size"]:
                seen_pages[pn] = h
        unique_headings = sorted(seen_pages.values(), key=lambda h: h["page_num"])

        for i, heading in enumerate(unique_headings):
            ch_start = heading["page_num"] + 1  # convert to 1-indexed
            if i + 1 < len(unique_headings):
                ch_end = unique_headings[i + 1]["page_num"]  # page before next heading
            else:
                ch_end = total_pages
            chapters_found.append({
                "title":    heading["text"],
                "start":    ch_start,
                "end":      ch_end,
            })

        # Check if requested_end falls mid-chapter
        warnings  = []
        final_end = requested_end

        for ch in chapters_found:
            if ch["start"] > requested_end:
                break  # This chapter starts after our range — not relevant
            if ch["start"] < requested_start:
                continue  # This chapter starts before our range
            if ch["start"] <= requested_end < ch["end"]:
                # Our end page falls inside this chapter
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
                        f"Prevention: auto-extended end page from "
                        f"{requested_end} to {final_end} "
                        f"to complete chapter '{ch['title']}'"
                    )

        return final_end, warnings

    @staticmethod
    def flag_incomplete_pages(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> set[int]:
        """
        Post-hoc verification: identify pages whose final words were NOT
        captured in the assembled chapter text.

        For each chapter, takes the last page in its range (from
        build_chapter_page_map), extracts the last N words from the last
        non-noise block on that page as a fingerprint, then searches for
        those words in the chapter text.

        If found → page was fully captured.
        If not found → page was incomplete (cut off or missing content).

        Returns set of 0-indexed page numbers considered incomplete.
        """
        from text_cleaner import TextCleaner

        if not blocks or not chapters:
            return set()

        page_blocks_map: dict[int, list[dict]] = defaultdict(list)
        for block in blocks:
            page_blocks_map[block.get("page_num", 0)].append(block)

        page_map    = TextAnalyzer.build_chapter_page_map(blocks, chapters)
        incomplete  = set()

        for title, chapter_text in chapters.items():
            if not chapter_text.strip() or title not in page_map:
                continue

            _start_pg, last_pg = page_map[title]
            pg_blocks          = page_blocks_map.get(last_pg, [])

            if not pg_blocks:
                continue

            fingerprint = TextCleaner.get_page_fingerprint(pg_blocks)

            if not fingerprint:
                logger.debug(
                    f"No fingerprint extractable for last page {last_pg+1} "
                    f"of '{title}' — skipping check."
                )
                continue

            found = TextCleaner.fingerprint_in_text(fingerprint, chapter_text)

            if not found:
                incomplete.add(last_pg)
                logger.info(
                    f"Page {last_pg+1} flagged incomplete for chapter '{title}': "
                    f"fingerprint '{fingerprint}' not found in chapter text."
                )
            else:
                logger.debug(
                    f"Page {last_pg+1} verified complete for '{title}': "
                    f"fingerprint '{fingerprint}' confirmed."
                )

        logger.info(
            f"Post-hoc page verification: {len(incomplete)} incomplete pages detected."
        )
        return incomplete

    @staticmethod
    def flag_incomplete_chapters(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> set[str]:
        """
        Post-hoc verification at chapter level.

        A chapter is flagged if:
          1. Its last page failed fingerprint verification, OR
          2. The chapter text ends mid-word (last character is a letter,
             not punctuation — definitively cut off)

        Returns set of chapter title strings considered incomplete.
        """
        if not blocks or not chapters:
            return set()

        incomplete_pages    = TextAnalyzer.flag_incomplete_pages(blocks, chapters)
        page_map            = TextAnalyzer.build_chapter_page_map(blocks, chapters)
        incomplete_chapters = set()

        for title, chapter_text in chapters.items():
            if not chapter_text.strip():
                incomplete_chapters.add(title)
                continue

            # Check if last page of this chapter failed verification
            if title in page_map:
                _start, last_pg = page_map[title]
                if last_pg in incomplete_pages:
                    incomplete_chapters.add(title)
                    logger.info(
                        f"Chapter '{title}' flagged: last page {last_pg+1} "
                        f"failed fingerprint verification."
                    )
                    continue

            # Secondary: does text end mid-word?
            last_char = ""
            for ch in reversed(chapter_text):
                if ch.strip():
                    last_char = ch
                    break
            if last_char and last_char.isalpha():
                incomplete_chapters.add(title)
                logger.info(
                    f"Chapter '{title}' flagged: ends mid-word "
                    f"'...{chapter_text.strip()[-30:]}'"
                )

        logger.info(
            f"Post-hoc chapter verification: {len(incomplete_chapters)} "
            f"incomplete chapters detected."
        )
        return incomplete_chapters

    @staticmethod
    def get_page_range_per_chapter(
        blocks:   list[dict],
        chapters: OrderedDict,
    ) -> dict[str, tuple[int, int]]:
        """
        Returns 1-indexed (first_page, last_page) per chapter for UI display.
        Wraps build_chapter_page_map.
        """
        raw = TextAnalyzer.build_chapter_page_map(blocks, chapters)
        return {title: (s + 1, e + 1) for title, (s, e) in raw.items()}

    # ================================================================== #
    # SECTION 5 — UTILITIES                                               #
    # ================================================================== #

    @staticmethod
    def get_reading_time_estimate(text: str, wpm: int = None) -> str:
        """Estimate audio listening time based on words per minute."""
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
        """
        Split text into sentence-bounded chunks under max_chars.
        Used by AudioGenerator for TTS chunking.
        """
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
            # If a single sentence exceeds max_chars, split at word boundaries
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