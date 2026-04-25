"""
pdf_processor.py
================
Everything PDF. Extracts structured block data from PDF files using
PyMuPDF, with OCR fallback via pytesseract for scanned documents.

Also provides get_heading_pages() for fast pre-scan of chapter
boundaries before full extraction — used by the prevention layer
to detect if a requested page range cuts through a chapter.

Block schema:
  text, x0, y0, x1, y1, font_size, font_name,
  page_num, page_height, page_width, is_heading, heading_level
"""

import io
import logging
import tempfile
from typing import Optional

import pymupdf
import pytesseract
import streamlit as st
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


class PDFProcessor:

    # ================================================================== #
    # SECTION 1 — VALIDATION                                              #
    # ================================================================== #

    @staticmethod
    def validate_pdf(pdf_bytes: bytes) -> tuple[bool, str]:
        """
        Validate raw PDF bytes before any processing.
        Returns (True, "") if valid, (False, error_message) otherwise.
        """
        if not pdf_bytes:
            return False, "The uploaded file is empty."

        if len(pdf_bytes) > Config.MAX_PDF_SIZE_BYTES:
            size_mb = len(pdf_bytes) / (1024 * 1024)
            return False, (
                f"File is {size_mb:.1f} MB — exceeds the "
                f"{Config.MAX_PDF_SIZE_MB} MB limit."
            )

        if not pdf_bytes.startswith(b"%PDF-"):
            return False, "File does not appear to be a valid PDF (missing PDF header)."

        try:
            doc        = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
        except Exception as e:
            return False, f"Could not open PDF: {e}"

        if page_count == 0:
            return False, "PDF has no pages."

        return True, ""

    # ================================================================== #
    # SECTION 2 — FAST PRE-SCAN (for chapter boundary prevention)        #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_heading_pages(pdf_bytes: bytes) -> list[dict]:
        """
        Fast structural pre-scan — extract ONLY heading-candidate spans
        from the full PDF without doing a full extraction.

        Used by TextAnalyzer.get_chapter_aware_page_range() to detect
        chapter boundaries BEFORE the user's page range is applied,
        enabling the prevention layer to warn about mid-chapter cuts.

        Strategy:
          - Open PDF, iterate all pages
          - For each page, get text as dict (spans with font sizes)
          - Collect spans whose font size exceeds median * HEADING_SIZE_RATIO
          - Return minimal list: {text, page_num, font_size}

        This is intentionally fast — no filtering, no OCR, no caching
        of block data. Runs on the full PDF regardless of selected range.

        Args:
          pdf_bytes: raw PDF bytes

        Returns:
          List of dicts: {text: str, page_num: int, font_size: float}
          Empty list on any error or if PDF is scanned (no text).
        """
        heading_candidates = []

        try:
            doc        = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            all_sizes  = []

            # Pass 1: collect all font sizes across entire document
            for page_num in range(len(doc)):
                page      = doc[page_num]
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 0.0)
                            if size > 0:
                                all_sizes.append(size)

            if not all_sizes:
                doc.close()
                return []

            import statistics
            median_size = statistics.median(all_sizes)
            threshold   = median_size * Config.HEADING_SIZE_RATIO

            # Pass 1b: detect cover pages (pages where ≥70% of spans are large font)
            # Cover pages have decorative title text that should not be chapter headings
            page_large: dict[int, int] = {}
            page_body:  dict[int, int] = {}
            for page_num in range(len(doc)):
                page      = doc[page_num]
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            size = span.get("size", 0.0)
                            if not text:
                                continue
                            if size >= threshold:
                                page_large[page_num] = page_large.get(page_num, 0) + 1
                            else:
                                page_body[page_num] = page_body.get(page_num, 0) + 1

            cover_pages: set[int] = set()
            for pn in range(len(doc)):
                large = page_large.get(pn, 0)
                body  = page_body.get(pn, 0)
                total = large + body
                if total > 0 and large / total >= 0.70 and body < 5:
                    cover_pages.add(pn)
                    logger.debug(f"Pre-scan: page {pn+1} identified as cover page")

            # Pass 2: collect spans above threshold (excluding cover pages)
            for page_num in range(len(doc)):
                if page_num in cover_pages:
                    continue
                page      = doc[page_num]
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text      = span.get("text", "").strip()
                            font_size = span.get("size", 0.0)
                            if (
                                font_size >= threshold
                                and Config.MIN_HEADING_CHARS <= len(text) <= Config.MAX_HEADING_CHARS
                                and len(text.split()) >= getattr(Config, "HEADING_MIN_WORDS", 2)
                            ):
                                heading_candidates.append({
                                    "text":      text,
                                    "page_num":  page_num,
                                    "font_size": font_size,
                                })

            doc.close()

        except Exception as e:
            logger.warning(f"get_heading_pages failed: {e}")
            return []

        logger.info(
            f"Pre-scan: found {len(heading_candidates)} heading candidates "
            f"across full PDF."
        )
        return heading_candidates

    # ================================================================== #
    # SECTION 3 — MASTER EXTRACTION                                       #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def extract(
        pdf_bytes:  bytes,
        lang_code:  str          = "en",
        start_page: int          = 1,
        end_page:   Optional[int] = None,
        mode:       str          = "auto",
    ) -> dict:
        """
        Master extraction. Routes to text or OCR mode.

        Returns:
          {blocks, toc, metadata, page_count, mode_used}
        """
        doc_for_count = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages   = len(doc_for_count)
        toc           = doc_for_count.get_toc()
        metadata      = doc_for_count.metadata or {}
        doc_for_count.close()

        start_idx = max(0, start_page - 1)
        if end_page is None or end_page > total_pages:
            end_idx = total_pages
        else:
            end_idx = end_page

        if start_idx >= total_pages:
            start_idx = 0

        if mode == "auto" or mode not in ("text", "ocr"):
            mode = "text"

        logger.info(
            f"Extracting pages {start_idx+1}–{end_idx} of {total_pages} "
            f"in '{mode}' mode, language '{lang_code}'."
        )

        if mode == "text":
            blocks    = PDFProcessor._extract_text_mode(pdf_bytes, start_idx, end_idx)
            mode_used = "text"
        else:
            blocks    = PDFProcessor._extract_ocr_mode(pdf_bytes, lang_code, start_idx, end_idx)
            mode_used = "ocr"

        return {
            "blocks":     blocks,
            "toc":        toc,
            "metadata":   metadata,
            "page_count": total_pages,
            "mode_used":  mode_used,
        }

    # ================================================================== #
    # SECTION 4 — TEXT MODE EXTRACTION                                    #
    # ================================================================== #

    @staticmethod
    def _extract_text_mode(
        pdf_bytes: bytes,
        start_idx: int,
        end_idx:   int,
    ) -> list[dict]:
        """
        Extract text blocks using PyMuPDF native structured text extraction.
        Span-level granularity for accurate font size per text run.
        """
        blocks = []
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(start_idx, end_idx):
                page        = doc[page_num]
                page_width  = page.rect.width
                page_height = page.rect.height
                page_dict   = page.get_text("dict")

                for raw_block in page_dict.get("blocks", []):
                    if raw_block.get("type") != 0:
                        continue
                    for line in raw_block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            bbox = span.get("bbox", (0, 0, 0, 0))
                            blocks.append({
                                "text":          text,
                                "x0":            bbox[0],
                                "y0":            bbox[1],
                                "x1":            bbox[2],
                                "y1":            bbox[3],
                                "font_size":     span.get("size", 0.0),
                                "font_name":     span.get("font", ""),
                                "page_num":      page_num,
                                "page_height":   page_height,
                                "page_width":    page_width,
                                "is_heading":    False,
                                "heading_level": 0,
                            })
            doc.close()
        except Exception as e:
            logger.error(f"Text mode extraction failed: {e}")
            raise

        logger.info(f"Text mode: extracted {len(blocks)} spans from pages {start_idx+1}–{end_idx}.")
        return blocks

    # ================================================================== #
    # SECTION 5 — OCR MODE EXTRACTION                                     #
    # ================================================================== #

    @staticmethod
    def _extract_ocr_mode(
        pdf_bytes: bytes,
        lang_code: str,
        start_idx: int,
        end_idx:   int,
        dpi:       int = 200,
    ) -> list[dict]:
        """
        Extract text from scanned PDFs using pytesseract.
        Returns same block schema as text mode with font_size=0.0.
        """
        tesseract_lang = Config.get_tesseract_code(lang_code)
        blocks         = []
        tess_config    = "--oem 3 --psm 6"
        pt_per_px      = 72.0 / dpi

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(start_idx, end_idx):
                page        = doc[page_num]
                page_width  = page.rect.width
                page_height = page.rect.height

                scale  = dpi / 72.0
                matrix = pymupdf.Matrix(scale, scale)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                img_bytes = pixmap.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                try:
                    tess_data = pytesseract.image_to_data(
                        pil_image,
                        lang=tesseract_lang,
                        config=tess_config,
                        output_type=pytesseract.Output.DICT,
                    )
                except pytesseract.TesseractError as e:
                    logger.warning(f"Tesseract error on page {page_num+1}: {e}. Skipping.")
                    continue

                current_line_num  = None
                current_line_text = []
                current_line_bbox = None
                current_line_conf = []
                n = len(tess_data["text"])

                def _flush_line():
                    if not current_line_text:
                        return
                    text     = " ".join(current_line_text).strip()
                    if not text:
                        return
                    avg_conf = sum(current_line_conf) / len(current_line_conf)
                    if avg_conf < 40:
                        return
                    blocks.append({
                        "text":          text,
                        "x0":            current_line_bbox[0] * pt_per_px,
                        "y0":            current_line_bbox[1] * pt_per_px,
                        "x1":            current_line_bbox[2] * pt_per_px,
                        "y1":            current_line_bbox[3] * pt_per_px,
                        "font_size":     0.0,
                        "font_name":     "",
                        "page_num":      page_num,
                        "page_height":   page_height,
                        "page_width":    page_width,
                        "is_heading":    False,
                        "heading_level": 0,
                    })

                for i in range(n):
                    word      = tess_data["text"][i]
                    conf      = int(tess_data["conf"][i])
                    line_num  = tess_data["line_num"][i]
                    left      = tess_data["left"][i]
                    top       = tess_data["top"][i]
                    width     = tess_data["width"][i]
                    height    = tess_data["height"][i]

                    if not word.strip() or conf == -1:
                        if current_line_num is not None and line_num != current_line_num:
                            _flush_line()
                            current_line_text = []
                            current_line_bbox = None
                            current_line_conf = []
                            current_line_num  = line_num
                        continue

                    if conf < 30:
                        continue

                    x0, y0, x1, y1 = left, top, left + width, top + height

                    if line_num != current_line_num:
                        _flush_line()
                        current_line_text = []
                        current_line_bbox = (x0, y0, x1, y1)
                        current_line_conf = []
                        current_line_num  = line_num

                    current_line_text.append(word)
                    current_line_conf.append(conf)
                    if current_line_bbox is not None:
                        current_line_bbox = (
                            min(current_line_bbox[0], x0),
                            min(current_line_bbox[1], y0),
                            max(current_line_bbox[2], x1),
                            max(current_line_bbox[3], y1),
                        )
                    else:
                        current_line_bbox = (x0, y0, x1, y1)

                _flush_line()

            doc.close()

        except Exception as e:
            logger.error(f"OCR mode extraction failed: {e}")
            raise

        logger.info(f"OCR mode: extracted {len(blocks)} lines from pages {start_idx+1}–{end_idx}.")
        return blocks

    # ================================================================== #
    # SECTION 6 — METADATA & STRUCTURE                                    #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_toc(pdf_bytes: bytes) -> list:
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            toc = doc.get_toc()
            doc.close()
            return toc
        except Exception as e:
            logger.error(f"Error extracting TOC: {e}")
            return []

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_metadata(pdf_bytes: bytes) -> dict:
        try:
            doc      = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            metadata = doc.metadata or {}
            doc.close()
            defaults = {
                "title": "", "author": "", "subject": "",
                "keywords": "", "creator": "", "producer": "",
                "creationDate": "", "modDate": "", "format": "", "encryption": "",
            }
            defaults.update({k: v for k, v in metadata.items() if v})
            return defaults
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_page_count(pdf_bytes: bytes) -> int:
        try:
            doc   = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Error counting pages: {e}")
            return 0

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_page_dimensions(pdf_bytes: bytes, page_num: int = 0) -> tuple[float, float]:
        try:
            doc    = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            page   = doc[page_num]
            w, h   = page.rect.width, page.rect.height
            doc.close()
            return w, h
        except Exception as e:
            logger.error(f"Error getting page dimensions: {e}")
            return 595.0, 842.0

    # ================================================================== #
    # SECTION 7 — PAGE RENDERING                                          #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def pdf_to_page_images(
        pdf_bytes:  bytes,
        start_page: int          = 1,
        end_page:   Optional[int] = None,
        dpi:        int          = 150,
    ) -> list:
        images = []
        try:
            doc         = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            start_idx   = max(0, start_page - 1)
            end_idx     = total_pages if end_page is None or end_page > total_pages else end_page
            scale       = dpi / 72.0
            matrix      = pymupdf.Matrix(scale, scale)
            for page_num in range(start_idx, end_idx):
                page      = doc[page_num]
                pixmap    = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pixmap.tobytes("png")
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(pil_img)
            doc.close()
        except Exception as e:
            logger.error(f"Error rendering PDF pages: {e}")
        return images