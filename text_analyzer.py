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
    # SECTION 3B - EXTENDED DOCUMENT ANALYSIS                           #
    # All methods take plain text strings. No external dependencies.    #
    # ================================================================== #

    @staticmethod
    def compute_readability(text: str) -> dict:
        """
        Compute Flesch Reading Ease and Flesch-Kincaid Grade Level.

        Flesch Reading Ease (0-100):
          90-100 = Very Easy (5th grade)
          60-70  = Standard (8th-9th grade)
          30-50  = Difficult (college)
          0-30   = Very Difficult (professional)

        Flesch-Kincaid Grade Level: US school grade equivalent.

        Both are computed from syllable count, word count, sentence count.
        Syllable counting uses the vowel-group heuristic (fast, ~90% accurate).

        Returns dict with: flesch_ease, flesch_kincaid, grade_label, ease_label
        """
        import re

        def count_syllables(word: str) -> int:
            word = word.lower().strip('.,!?;:\'"')
            if len(word) <= 3:
                return 1
            # Count vowel groups as syllable approximation
            vowels = re.findall(r"[aeiouy]+", word)
            count  = len(vowels)
            # Subtract silent e at end
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words     = re.findall(r"[a-zA-Z]+", text)

        if not sentences or not words:
            return {
                "flesch_ease": 0, "flesch_kincaid": 0,
                "grade_label": "Unknown", "ease_label": "Unknown",
            }

        num_sentences = len(sentences)
        num_words     = len(words)
        num_syllables = sum(count_syllables(w) for w in words)

        # Avoid division by zero
        if num_sentences == 0 or num_words == 0:
            return {
                "flesch_ease": 0, "flesch_kincaid": 0,
                "grade_label": "Unknown", "ease_label": "Unknown",
            }

        asl = num_words / num_sentences         # avg sentence length
        asw = num_syllables / num_words         # avg syllables per word

        flesch_ease     = round(206.835 - 1.015 * asl - 84.6 * asw, 1)
        flesch_ease     = max(0, min(100, flesch_ease))
        flesch_kincaid  = round(0.39 * asl + 11.8 * asw - 15.59, 1)
        flesch_kincaid  = max(0, flesch_kincaid)

        # Labels
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
        Compute detailed text statistics useful for audio listeners.

        Returns dict with:
          total_words, unique_words, vocabulary_richness,
          avg_sentence_length, avg_word_length,
          longest_chapter, shortest_chapter,
          chapter_distribution (list of {title, word_count, pct}),
          total_sentences, paragraphs
        """
        import re
        from collections import OrderedDict as OD

        words     = re.findall(r"[a-zA-Z]+", text.lower())
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        paras     = [p for p in re.split(r'\n\n+', text) if p.strip()]

        total_words   = len(words)
        unique_words  = len(set(words))
        vocab_rich    = round(unique_words / total_words * 100, 1) if total_words > 0 else 0
        avg_sent_len  = round(total_words / len(sentences), 1) if sentences else 0
        avg_word_len  = round(sum(len(w) for w in words) / total_words, 1) if total_words > 0 else 0

        # Chapter distribution
        chapter_dist = []
        for title, ch_text in chapters.items():
            ch_words = len(re.findall(r"[a-zA-Z]+", ch_text))
            pct      = round(ch_words / total_words * 100, 1) if total_words > 0 else 0
            chapter_dist.append({
                "title":      title,
                "word_count": ch_words,
                "pct":        pct,
            })

        # Longest and shortest chapters by word count
        if chapter_dist:
            longest  = max(chapter_dist, key=lambda x: x["word_count"])
            shortest = min(chapter_dist, key=lambda x: x["word_count"])
        else:
            longest = shortest = {"title": "N/A", "word_count": 0, "pct": 0}

        return {
            "total_words":        total_words,
            "unique_words":       unique_words,
            "vocabulary_richness": vocab_rich,
            "avg_sentence_length": avg_sent_len,
            "avg_word_length":    avg_word_len,
            "total_sentences":    len(sentences),
            "paragraphs":         len(paras),
            "longest_chapter":    longest,
            "shortest_chapter":   shortest,
            "chapter_distribution": chapter_dist,
        }

    @staticmethod
    def detect_content_type(text: str, chapters: OrderedDict) -> dict:
        """
        Detect the content type of the document from structural and
        linguistic signals.

        Types: Academic, Report/Analysis, Fiction/Narrative,
               News/Article, Technical, Legal, General

        Returns dict: { type, confidence, signals }
        """
        import re

        full_lower = text.lower()
        scores     = {
            "Academic":           0,
            "Report / Analysis":  0,
            "Fiction / Narrative": 0,
            "News / Article":     0,
            "Technical":          0,
            "Legal":              0,
        }

        # Academic signals
        academic_terms = ["abstract", "methodology", "hypothesis", "conclusion",
                          "references", "et al", "study", "research", "findings",
                          "analysis", "literature", "cited", "peer-reviewed"]
        scores["Academic"] += sum(full_lower.count(t) for t in academic_terms)

        # Report/analysis signals
        report_terms = ["framework", "kpi", "revenue", "market", "strategy",
                        "growth", "forecast", "quarter", "fiscal", "benchmark",
                        "insight", "pattern", "failure", "success", "rate"]
        scores["Report / Analysis"] += sum(full_lower.count(t) for t in report_terms)

        # Fiction signals
        fiction_terms = ["said", "whispered", "shouted", "felt", "thought",
                         "walked", "smiled", "chapter", "protagonist", "character"]
        scores["Fiction / Narrative"] += sum(full_lower.count(t) for t in fiction_terms)
        # Dialogue is a strong fiction signal
        dialogue_count = len(re.findall(r'"[^"]{5,}"', text))
        scores["Fiction / Narrative"] += dialogue_count * 3

        # News/article signals
        news_terms = ["according to", "reported", "announced", "government",
                      "officials", "sources said", "press release", "interview"]
        scores["News / Article"] += sum(full_lower.count(t) for t in news_terms)

        # Technical signals
        tech_terms = ["function", "algorithm", "implementation", "api",
                      "configuration", "parameter", "database", "module",
                      "variable", "syntax", "library", "framework"]
        scores["Technical"] += sum(full_lower.count(t) for t in tech_terms)
        # Code-like patterns
        scores["Technical"] += len(re.findall(r"[a-z_]+\([^\)]*\)", text)) * 2

        # Legal signals
        legal_terms = ["whereas", "hereby", "pursuant", "notwithstanding",
                       "jurisdiction", "liability", "clause", "agreement",
                       "parties", "indemnify", "covenant", "shall"]
        scores["Legal"] += sum(full_lower.count(t) for t in legal_terms)

        # Determine winner
        top_type   = max(scores, key=scores.get)
        top_score  = scores[top_type]
        total      = sum(scores.values()) or 1
        confidence = round(top_score / total * 100, 0)

        if top_score < 5:
            top_type   = "General"
            confidence = 0

        # Top signals that drove the detection
        top_signals = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "type":       top_type,
            "confidence": int(confidence),
            "signals":    top_signals,
            "all_scores": scores,
        }

    @staticmethod
    def compute_topic_density(text: str, keywords: list) -> list:
        """
        Compute how densely each top keyword appears across the document,
        expressed as occurrences per 1,000 words.

        Args:
          text:     full document text
          keywords: list of (word, count) tuples from extract_keywords()

        Returns list of dicts: { word, count, density_per_1k, bar_pct }
        where bar_pct is 0-100 relative to the most frequent term.
        """
        import re
        total_words = len(re.findall(r"\w+", text))
        if not keywords or total_words == 0:
            return []

        max_count  = keywords[0][1] if keywords else 1
        result     = []
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
        Compute readability score per chapter so the user can see which
        sections are hardest to follow while listening.

        Returns list of dicts: { title, flesch_ease, ease_label, word_count }
        sorted from hardest to easiest.
        """
        results = []
        for title, text in chapters.items():
            if not text.strip():
                continue
            r = TextAnalyzer.compute_readability(text)
            import re
            wc = len(re.findall(r"\w+", text))
            results.append({
                "title":       title,
                "flesch_ease": r["flesch_ease"],
                "ease_label":  r["ease_label"],
                "word_count":  wc,
            })
        return sorted(results, key=lambda x: x["flesch_ease"])

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