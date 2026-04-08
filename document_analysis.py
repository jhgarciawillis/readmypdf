"""
document_analysis.py
====================
NLP analysis and extended document KPIs.
All methods are pure functions — no Streamlit, no PDF, no audio.

Sections:
  1 — NLP (keywords, characters, summary, sentiment)
  2 — Extended KPIs (readability, stats, content type, topic density, complexity)
"""

import re
import logging
from collections import Counter, OrderedDict
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class DocumentAnalysis:
    """Static methods for NLP and KPI analysis of text content."""

    # ================================================================== #
    # SECTION 1 — NLP ANALYSIS                                           #
    # ================================================================== #

    @staticmethod
    def extract_keywords(text: str, lang_code: str = "en", top_n: int = 15) -> list:
        """Top keywords by frequency, filtering stopwords. Returns [(word, count)]."""
        stopwords_en = {
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
        stopwords_es = {
            "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
            "al", "en", "y", "o", "pero", "si", "no", "que", "se", "su", "sus",
            "con", "por", "para", "como", "más", "este", "esta", "estos", "estas",
            "ese", "esa", "esos", "esas", "ser", "está", "han", "hay", "ya",
            "son", "fue", "era", "entre", "sobre", "también", "así", "cuando",
            "sin", "dos", "muy", "bien", "porque", "aunque", "desde", "hasta",
        }
        stopwords = stopwords_en | stopwords_es

        words  = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", text.lower())
        counts = Counter(w for w in words if w not in stopwords)
        return counts.most_common(top_n)

    @staticmethod
    def detect_character_names(text: str, lang_code: str = "en") -> list:
        """Recurring capitalized names (entities/characters). Returns list of strings."""
        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        counts     = Counter(candidates)
        non_names  = {
            "The", "This", "That", "These", "Those", "There", "Their",
            "They", "When", "Where", "What", "Which", "While", "With",
            "From", "Have", "Some", "Many", "Most", "More", "Such",
            "Also", "After", "Before", "During", "Under", "Over",
            "Figure", "Table", "Chapter", "Section", "Example",
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
        Skips bylines, headers, short metadata lines.
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
            if s.count("|") >= 1 and len(s) < 100:
                continue
            if re.match(r"^By\s+[A-Z]", s):
                continue
            if re.match(r"^\d+\.\s+[A-Z]", s) and len(s) < 120:
                continue
            content.append(s)
            if len(content) >= num_sentences:
                break
        return " ".join(content)

    @staticmethod
    def sentiment_analysis(text: str, lang_code: str = "en") -> dict:
        """Lexicon-based sentiment. Returns {label, confidence, positive_count, negative_count}."""
        positive = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "outstanding", "success", "successful", "effective", "improve",
            "improvement", "benefit", "positive", "strong", "better", "best",
            "innovative", "opportunity", "growth", "achieve", "achievement",
            "valuable", "significant", "important", "clear", "proven", "trust",
            "bueno", "mejor", "excelente", "éxito", "eficaz", "mejorar",
        }
        negative = {
            "bad", "poor", "terrible", "awful", "failure", "fail", "failed",
            "problem", "issue", "risk", "danger", "loss", "decline", "weak",
            "difficult", "hard", "challenge", "concern", "worry", "threat",
            "ineffective", "wrong", "error", "mistake", "negative", "worse",
            "worst", "crisis", "collapse", "shutdown", "deny", "denial",
            "malo", "peor", "fracaso", "problema", "riesgo", "difícil",
        }
        words = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())
        pos   = sum(1 for w in words if w in positive)
        neg   = sum(1 for w in words if w in negative)
        total = pos + neg

        if total == 0:
            return {"label": "Neutral", "confidence": "low",
                    "positive_count": 0, "negative_count": 0}

        ratio = pos / total
        label = "Positive" if ratio > 0.6 else ("Negative" if ratio < 0.4 else "Neutral")
        diff  = abs(pos - neg)
        conf  = "high" if diff > total * 0.3 else ("medium" if diff > total * 0.1 else "low")

        return {
            "label":          label,
            "confidence":     conf,
            "positive_count": pos,
            "negative_count": neg,
        }

    # ================================================================== #
    # SECTION 2 — EXTENDED KPIs (SEMrush / NinjaSEO inspired)            #
    # ================================================================== #

    @staticmethod
    def compute_readability(text: str) -> dict:
        """
        Flesch Reading Ease + Flesch-Kincaid Grade Level.
        No external deps — pure stdlib regex + arithmetic.

        Flesch Ease scale:
          90–100 Very Easy  |  70–90 Easy  |  60–70 Standard
          50–60 Fairly Difficult  |  30–50 Difficult  |  0–30 Very Difficult

        Returns {flesch_ease, flesch_kincaid, grade_label, ease_label}
        """
        def count_syllables(word: str) -> int:
            word = word.lower().strip(".,!?;:'\"")
            if len(word) <= 3:
                return 1
            vowels = re.findall(r"[aeiouyáéíóúü]+", word)
            count  = len(vowels)
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        sentences     = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        words         = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text)

        if not sentences or not words:
            return {"flesch_ease": 0, "flesch_kincaid": 0,
                    "grade_label": "Unknown", "ease_label": "Unknown"}

        asl = len(words) / len(sentences)
        asw = sum(count_syllables(w) for w in words) / len(words)

        flesch_ease    = round(max(0, min(100, 206.835 - 1.015 * asl - 84.6 * asw)), 1)
        flesch_kincaid = round(max(0, 0.39 * asl + 11.8 * asw - 15.59), 1)

        ease_labels = [
            (90, "Very Easy"), (70, "Easy"), (60, "Standard"),
            (50, "Fairly Difficult"), (30, "Difficult"), (0, "Very Difficult"),
        ]
        ease_label = next(lbl for thresh, lbl in ease_labels if flesch_ease >= thresh)

        grade_labels = [
            (16, "Post-Graduate"), (12, "University"),
            (9, "High School"), (6, "Middle School"), (0, "Elementary"),
        ]
        grade_label = next(lbl for thresh, lbl in grade_labels if flesch_kincaid >= thresh)

        return {
            "flesch_ease":    flesch_ease,
            "flesch_kincaid": flesch_kincaid,
            "grade_label":    grade_label,
            "ease_label":     ease_label,
        }

    @staticmethod
    def compute_text_stats(text: str, chapters: OrderedDict) -> dict:
        """
        Vocabulary richness, sentence/word length stats, chapter distribution.

        KPIs:
          total_words, unique_words, vocabulary_richness (%),
          avg_sentence_length (words), avg_word_length (chars),
          total_sentences, paragraphs,
          longest_chapter, shortest_chapter,
          chapter_distribution: [{title, word_count, pct}]
        """
        words        = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())
        sentences    = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        paras        = [p for p in re.split(r"\n\n+", text) if p.strip()]
        total        = len(words)
        unique       = len(set(words))

        chapter_dist = []
        for title, ch_text in chapters.items():
            wc  = len(re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", ch_text))
            pct = round(wc / total * 100, 1) if total > 0 else 0
            chapter_dist.append({"title": title, "word_count": wc, "pct": pct})

        longest  = max(chapter_dist, key=lambda x: x["word_count"]) if chapter_dist else {"title": "N/A", "word_count": 0}
        shortest = min(chapter_dist, key=lambda x: x["word_count"]) if chapter_dist else {"title": "N/A", "word_count": 0}

        return {
            "total_words":          total,
            "unique_words":         unique,
            "vocabulary_richness":  round(unique / total * 100, 1) if total > 0 else 0,
            "avg_sentence_length":  round(total / len(sentences), 1) if sentences else 0,
            "avg_word_length":      round(sum(len(w) for w in words) / total, 1) if total > 0 else 0,
            "total_sentences":      len(sentences),
            "paragraphs":           len(paras),
            "longest_chapter":      longest,
            "shortest_chapter":     shortest,
            "chapter_distribution": chapter_dist,
        }

    @staticmethod
    def detect_content_type(text: str, chapters: OrderedDict) -> dict:
        """
        Classify document type from linguistic signals.
        Types: Academic | Report/Analysis | Fiction/Narrative |
               News/Article | Technical | Legal | General
        Returns {type, confidence, all_scores}
        """
        fl = text.lower()
        scores = {
            "Academic":            0,
            "Report / Analysis":   0,
            "Fiction / Narrative": 0,
            "News / Article":      0,
            "Technical":           0,
            "Legal":               0,
        }

        # Academic
        for t in ["abstract","methodology","hypothesis","conclusion","references",
                  "et al","study","research","findings","analysis","literature",
                  "ecuación","regresión","estimación","hipótesis","parámetro"]:
            scores["Academic"] += fl.count(t)

        # Report/Analysis
        for t in ["framework","kpi","revenue","market","strategy","growth",
                  "forecast","quarter","fiscal","benchmark","insight","pattern",
                  "failure","rate","modelo","datos","variable","correlación"]:
            scores["Report / Analysis"] += fl.count(t)

        # Fiction
        for t in ["said","whispered","shouted","felt","thought","walked","smiled"]:
            scores["Fiction / Narrative"] += fl.count(t)
        scores["Fiction / Narrative"] += len(re.findall(r'"[^"]{5,}"', text)) * 3

        # News/Article
        for t in ["according to","reported","announced","government","officials",
                  "sources said","press release","según","anunció"]:
            scores["News / Article"] += fl.count(t)

        # Technical
        for t in ["function","algorithm","implementation","api","configuration",
                  "parameter","database","module","variable","syntax"]:
            scores["Technical"] += fl.count(t)
        scores["Technical"] += len(re.findall(r"[a-z_]+\([^)]*\)", text)) * 2

        # Legal
        for t in ["whereas","hereby","pursuant","notwithstanding","jurisdiction",
                  "liability","clause","agreement","parties","indemnify"]:
            scores["Legal"] += fl.count(t)

        top_type  = max(scores, key=scores.get)
        top_score = scores[top_type]
        total     = sum(scores.values()) or 1
        conf      = round(top_score / total * 100)

        if top_score < 5:
            top_type, conf = "General", 0

        return {"type": top_type, "confidence": int(conf), "all_scores": scores}

    @staticmethod
    def compute_topic_density(text: str, keywords: list) -> list:
        """
        Keyword occurrences per 1,000 words.
        Returns [{word, count, density_per_1k, bar_pct}]
        """
        total = len(re.findall(r"\b\w+\b", text))
        if not keywords or total == 0:
            return []
        max_count = keywords[0][1] if keywords else 1
        return [
            {
                "word":           word,
                "count":          count,
                "density_per_1k": round(count / total * 1000, 2),
                "bar_pct":        round(count / max_count * 100),
            }
            for word, count in keywords[:15]
        ]

    @staticmethod
    def detect_language_complexity_by_chapter(chapters: OrderedDict) -> list:
        """
        Flesch Reading Ease per chapter, sorted hardest → easiest.
        Returns [{title, flesch_ease, ease_label, word_count}]
        """
        results = []
        for title, text in chapters.items():
            if not text.strip():
                continue
            r  = DocumentAnalysis.compute_readability(text)
            wc = len(re.findall(r"\b\w+\b", text))
            results.append({
                "title":       title,
                "flesch_ease": r["flesch_ease"],
                "ease_label":  r["ease_label"],
                "word_count":  wc,
            })
        return sorted(results, key=lambda x: x["flesch_ease"])

    @staticmethod
    def compute_lexical_diversity(text: str) -> dict:
        """
        Advanced lexical diversity metrics beyond basic vocabulary richness.

        Type-Token Ratio (TTR): unique/total (sensitive to length)
        Moving Average TTR (MATTR): averaged over 100-word windows, length-independent
        Hapax Ratio: words appearing exactly once / total unique words

        Returns {ttr, mattr, hapax_ratio, hapax_count, avg_word_frequency}
        """
        words = re.findall(r"\b[a-zA-ZÀ-ÿ]+\b", text.lower())
        if not words:
            return {"ttr": 0, "mattr": 0, "hapax_ratio": 0,
                    "hapax_count": 0, "avg_word_frequency": 0}

        from collections import Counter
        freq      = Counter(words)
        total     = len(words)
        unique    = len(freq)
        hapax     = sum(1 for v in freq.values() if v == 1)

        # MATTR with window size 100
        window = 100
        if total >= window:
            ttrs = []
            for i in range(0, total - window + 1, window // 2):
                window_words = words[i:i + window]
                ttrs.append(len(set(window_words)) / window)
            mattr = round(sum(ttrs) / len(ttrs) * 100, 1)
        else:
            mattr = round(unique / total * 100, 1)

        return {
            "ttr":                round(unique / total * 100, 1),
            "mattr":              mattr,
            "hapax_ratio":        round(hapax / unique * 100, 1) if unique > 0 else 0,
            "hapax_count":        hapax,
            "avg_word_frequency": round(total / unique, 1) if unique > 0 else 0,
        }

    @staticmethod
    def compute_sentence_complexity(text: str) -> dict:
        """
        Sentence-level complexity metrics useful for audio comprehension.

        Long sentences (>30 words) are hard to follow while listening.
        Very short sentences (<5 words) can indicate fragment-heavy content.

        Returns {
          long_sentence_count, long_sentence_pct,
          short_sentence_count, avg_sentence_length,
          longest_sentence_words, sentence_length_variance
        }
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]
        if not sentences:
            return {
                "long_sentence_count": 0, "long_sentence_pct": 0,
                "short_sentence_count": 0, "avg_sentence_length": 0,
                "longest_sentence_words": 0, "sentence_length_variance": 0,
            }

        lengths  = [len(s.split()) for s in sentences]
        avg      = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        long_s   = [l for l in lengths if l > 30]
        short_s  = [l for l in lengths if l < 5]

        return {
            "long_sentence_count":    len(long_s),
            "long_sentence_pct":      round(len(long_s) / len(lengths) * 100, 1),
            "short_sentence_count":   len(short_s),
            "avg_sentence_length":    round(avg, 1),
            "longest_sentence_words": max(lengths),
            "sentence_length_variance": round(variance, 1),
        }