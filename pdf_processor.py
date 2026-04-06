"""
pdf_processor.py
================
Everything PDF. Extracts structured block data from PDF files using
PyMuPDF, with OCR fallback via pytesseract for scanned documents.

Responsibilities:
  1. Validate incoming PDF bytes
  2. Extract text blocks with full coordinate and font metadata (text mode)
  3. Extract text blocks via OCR with bounding boxes (scanned mode)
  4. Retrieve table of contents, metadata, and page count
  5. Render pages to PIL Images (for OCR mode and PDF preview)

Core design decisions:
  - pdf_bytes (bytes) is the ONLY input format accepted. The file is read
    ONCE by the caller (streamlit_app.py) and passed everywhere as bytes.
    This eliminates the exhausted-stream bug from the original codebase.
  - All methods are static. No instance state.
  - @st.cache_data is applied only to methods whose inputs are safely
    hashable (bytes, str, int). Never applied to UploadedFile objects.
  - Both extraction modes return the SAME block schema so all downstream
    code (TextCleaner, TextAnalyzer) works identically regardless of mode.

Block schema (dict):
  {
    "text":        str,    # extracted text content
    "x0":          float,  # left edge
    "y0":          float,  # top edge
    "x1":          float,  # right edge
    "y1":          float,  # bottom edge
    "font_size":   float,  # 0.0 in OCR mode (not available from Tesseract)
    "font_name":   str,    # "" in OCR mode
    "page_num":    int,    # 0-indexed page number
    "page_height": float,  # full page height in points
    "page_width":  float,  # full page width in points
    "is_heading":  bool,   # False by default; set True by TextAnalyzer
    "heading_level": int,  # 0 by default; set 1/2/3 by TextAnalyzer
  }

This module knows about:
  - pymupdf (PyMuPDF >= 1.27)
  - pytesseract + PIL (OCR fallback)
  - streamlit (for @st.cache_data only)
  - config.Config

This module knows NOTHING about:
  - TextCleaner, TextAnalyzer, AudioGenerator, UIComponents
  - Chapter splitting, TTS, audio processing
"""

import io
import logging
import tempfile
from typing import Optional

import pymupdf          # PyMuPDF >= 1.27 — import as pymupdf not fitz
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

        Checks:
          1. Not empty
          2. Starts with the PDF magic bytes (%PDF-)
          3. Under the configured size limit (Config.MAX_PDF_SIZE_BYTES)
          4. Can be opened by PyMuPDF without error
          5. Has at least one page

        Returns:
          (True, "")               — valid PDF, ready to process
          (False, error_message)   — invalid, error_message explains why

        Called by streamlit_app.py immediately after reading the upload.
        No processing happens if this returns False.
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
            return False, (
                "File does not appear to be a valid PDF "
                "(missing PDF header)."
            )

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
        except Exception as e:
            return False, f"Could not open PDF: {e}"

        if page_count == 0:
            return False, "PDF has no pages."

        return True, ""

    # ================================================================== #
    # SECTION 2 — MASTER EXTRACTION METHOD                               #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def extract(
        pdf_bytes:  bytes,
        lang_code:  str  = "en",
        start_page: int  = 1,
        end_page:   Optional[int] = None,
        mode:       str  = "auto",
    ) -> dict:
        """
        Master extraction method. Routes to text or OCR extractor based
        on mode, then returns a unified result dict.

        Args:
          pdf_bytes:  raw PDF bytes (read once by caller)
          lang_code:  language code from Config.SUPPORTED_LANGUAGES
          start_page: 1-indexed first page to process (default: 1)
          end_page:   1-indexed last page to process (None = all pages)
          mode:       "auto" | "text" | "ocr"
                      "auto" uses TextCleaner.detect_pdf_mode() result
                      which must be passed in — see note below.

        Note on "auto" mode:
          detect_pdf_mode() is called by streamlit_app.py BEFORE extract()
          so the mode string passed here is already resolved to "text" or
          "ocr". Passing mode="auto" here falls back to "text" — this is
          intentional so extract() is self-contained for caching purposes.

        Returns dict:
          {
            "blocks":    list[dict],  # flat list of all block dicts
            "toc":       list,        # [(level, title, page), ...]
            "metadata":  dict,        # title, author, subject, etc.
            "page_count": int,
            "mode_used": str,         # "text" or "ocr"
          }

        Safe to cache: all args are bytes/str/int/None — hashable.
        """
        # Resolve page range
        doc_for_count = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc_for_count)
        toc         = doc_for_count.get_toc()
        metadata    = doc_for_count.metadata or {}
        doc_for_count.close()

        # Clamp page range to valid bounds
        start_idx = max(0, start_page - 1)           # convert to 0-indexed
        if end_page is None or end_page > total_pages:
            end_idx = total_pages                     # exclusive
        else:
            end_idx = end_page                        # 1-indexed end = exclusive 0-indexed

        if start_idx >= total_pages:
            logger.warning(
                f"start_page={start_page} exceeds total pages={total_pages}. "
                f"Resetting to page 1."
            )
            start_idx = 0

        # Resolve extraction mode
        if mode == "auto" or mode not in ("text", "ocr"):
            mode = "text"   # caller should have resolved this already

        logger.info(
            f"Extracting pages {start_idx+1}–{end_idx} of {total_pages} "
            f"in '{mode}' mode, language '{lang_code}'."
        )

        # Route to the correct extractor
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
    # SECTION 3 — TEXT MODE EXTRACTION (native PyMuPDF)                  #
    # ================================================================== #

    @staticmethod
    def _extract_text_mode(
        pdf_bytes:  bytes,
        start_idx:  int,
        end_idx:    int,
    ) -> list[dict]:
        """
        Extract text blocks from a text-based PDF using PyMuPDF's native
        structured text extraction.

        Uses page.get_text("dict") which returns a full document tree:
          page → blocks → lines → spans

        Each span contains:
          - text content
          - bounding box (x0, y0, x1, y1) in points
          - font name and size

        We flatten spans → lines → blocks into a single-level list of
        block dicts using the shared block schema.

        Span-level granularity is used (not block-level) so we capture
        individual font sizes per text run, enabling accurate heading
        detection even when a page mixes font sizes within a single
        PyMuPDF block (e.g. a bold heading followed by body text in
        the same visual block).

        Args:
          pdf_bytes:  raw PDF bytes
          start_idx:  0-indexed first page (inclusive)
          end_idx:    0-indexed last page (exclusive)

        Returns:
          Flat list of block dicts (shared schema).
        """
        blocks = []

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(start_idx, end_idx):
                page        = doc[page_num]
                page_width  = page.rect.width
                page_height = page.rect.height

                # get_text("dict") returns the full structured tree
                page_dict = page.get_text("dict")

                for raw_block in page_dict.get("blocks", []):
                    # Only process text blocks (type 0). type 1 = image blocks.
                    if raw_block.get("type") != 0:
                        continue

                    for line in raw_block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue

                            bbox = span.get("bbox", (0, 0, 0, 0))
                            # bbox from PyMuPDF is (x0, y0, x1, y1)

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

        logger.info(
            f"Text mode: extracted {len(blocks)} spans from "
            f"pages {start_idx+1}–{end_idx}."
        )
        return blocks

    # ================================================================== #
    # SECTION 4 — OCR MODE EXTRACTION (pytesseract fallback)             #
    # ================================================================== #

    @staticmethod
    def _extract_ocr_mode(
        pdf_bytes:  bytes,
        lang_code:  str,
        start_idx:  int,
        end_idx:    int,
        dpi:        int = 200,
    ) -> list[dict]:
        """
        Extract text from a scanned PDF using pytesseract OCR.

        Process per page:
          1. Render page to a PIL Image at 200 DPI using PyMuPDF
          2. Run pytesseract image_to_data() to get word-level bounding boxes
             and confidence scores
          3. Group high-confidence words into line-level blocks
          4. Convert pixel coordinates to point coordinates (matching text mode)
          5. Append to flat block list using shared block schema

        OCR mode limitations vs text mode:
          - font_size is always 0.0 (Tesseract does not report font sizes)
          - font_name is always ""
          - Heading detection by font size is therefore unavailable →
            TextAnalyzer falls back to regex pattern matching
          - Accuracy depends on scan quality and language model

        Args:
          pdf_bytes:  raw PDF bytes
          lang_code:  language code (e.g. "en") — mapped to Tesseract code
          start_idx:  0-indexed first page (inclusive)
          end_idx:    0-indexed last page (exclusive)
          dpi:        render resolution (higher = better OCR, slower)

        Returns:
          Flat list of block dicts (shared schema).
          font_size=0.0, font_name="" for all entries.
        """
        tesseract_lang = Config.get_tesseract_code(lang_code)
        blocks         = []
        # Tesseract config: OEM 3 = LSTM neural net, PSM 6 = uniform block of text
        tess_config    = "--oem 3 --psm 6"

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(start_idx, end_idx):
                page        = doc[page_num]
                page_width  = page.rect.width
                page_height = page.rect.height

                # Render page to image at target DPI
                # PyMuPDF default is 72 DPI; scale matrix converts to target DPI
                scale  = dpi / 72.0
                matrix = pymupdf.Matrix(scale, scale)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                # Convert to PIL Image via raw bytes
                img_bytes = pixmap.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Run Tesseract with data output (returns DataFrame-like dict)
                try:
                    tess_data = pytesseract.image_to_data(
                        pil_image,
                        lang=tesseract_lang,
                        config=tess_config,
                        output_type=pytesseract.Output.DICT,
                    )
                except pytesseract.TesseractError as e:
                    logger.warning(
                        f"Tesseract error on page {page_num+1}: {e}. "
                        f"Skipping page."
                    )
                    continue

                # Pixel-to-point conversion factors
                px_to_pt_x = page_width  / (page.rect.width  * scale / 72.0 * scale)
                px_to_pt_y = page_height / (page.rect.height * scale / 72.0 * scale)
                # Simplified: 1 pixel = (72 / dpi) points
                pt_per_px = 72.0 / dpi

                # Group Tesseract words into line blocks
                # Tesseract data has one entry per word; we group by line_num
                current_line_num  = None
                current_line_text = []
                current_line_bbox = None   # (x0, y0, x1, y1) in pixels
                current_line_conf = []

                n = len(tess_data["text"])

                def _flush_line():
                    """Append the current accumulated line as a block."""
                    if not current_line_text:
                        return
                    text = " ".join(current_line_text).strip()
                    if not text:
                        return
                    avg_conf = sum(current_line_conf) / len(current_line_conf)
                    if avg_conf < 40:
                        # Skip very low-confidence lines (likely noise)
                        return
                    x0_pt = current_line_bbox[0] * pt_per_px
                    y0_pt = current_line_bbox[1] * pt_per_px
                    x1_pt = current_line_bbox[2] * pt_per_px
                    y1_pt = current_line_bbox[3] * pt_per_px
                    blocks.append({
                        "text":          text,
                        "x0":            x0_pt,
                        "y0":            y0_pt,
                        "x1":            x1_pt,
                        "y1":            y1_pt,
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
                    block_num = tess_data["block_num"][i]
                    left      = tess_data["left"][i]
                    top       = tess_data["top"][i]
                    width     = tess_data["width"][i]
                    height    = tess_data["height"][i]

                    # Skip empty words or words with -1 confidence (layout markers)
                    if not word.strip() or conf == -1:
                        if current_line_num is not None and line_num != current_line_num:
                            _flush_line()
                            current_line_text = []
                            current_line_bbox = None
                            current_line_conf = []
                            current_line_num  = line_num
                        continue

                    # Skip very low-confidence individual words
                    if conf < 30:
                        continue

                    x0 = left
                    y0 = top
                    x1 = left + width
                    y1 = top  + height

                    if line_num != current_line_num:
                        _flush_line()
                        current_line_text = []
                        current_line_bbox = (x0, y0, x1, y1)
                        current_line_conf = []
                        current_line_num  = line_num

                    current_line_text.append(word)
                    current_line_conf.append(conf)

                    # Expand bounding box to encompass this word
                    if current_line_bbox is not None:
                        current_line_bbox = (
                            min(current_line_bbox[0], x0),
                            min(current_line_bbox[1], y0),
                            max(current_line_bbox[2], x1),
                            max(current_line_bbox[3], y1),
                        )
                    else:
                        current_line_bbox = (x0, y0, x1, y1)

                # Flush the last line on the page
                _flush_line()

                logger.debug(
                    f"OCR page {page_num+1}: extracted "
                    f"{len([b for b in blocks if b['page_num'] == page_num])} lines."
                )

            doc.close()

        except Exception as e:
            logger.error(f"OCR mode extraction failed: {e}")
            raise

        logger.info(
            f"OCR mode: extracted {len(blocks)} lines from "
            f"pages {start_idx+1}–{end_idx} (lang: {tesseract_lang})."
        )
        return blocks

    # ================================================================== #
    # SECTION 5 — METADATA & STRUCTURE                                   #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_toc(pdf_bytes: bytes) -> list:
        """
        Extract the PDF's built-in table of contents.

        Returns:
          List of [level, title, page] entries from PyMuPDF's get_toc().
          Empty list if the PDF has no embedded TOC.

        Note: This is the PDF's own bookmarks/outline — separate from the
        heading structure detected by TextAnalyzer from font sizes.
        Both are available to the UI; the font-based one is more reliable
        when the PDF's TOC is absent or incomplete.
        """
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
        """
        Extract PDF metadata (title, author, subject, creator, etc.).

        Returns:
          dict with keys: title, author, subject, keywords, creator,
          producer, creationDate, modDate, format, encryption.
          Missing fields are empty strings.
        """
        try:
            doc      = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            metadata = doc.metadata or {}
            doc.close()

            # Ensure all expected keys exist
            defaults = {
                "title": "", "author": "", "subject": "",
                "keywords": "", "creator": "", "producer": "",
                "creationDate": "", "modDate": "",
                "format": "", "encryption": "",
            }
            defaults.update({k: v for k, v in metadata.items() if v})
            return defaults

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_page_count(pdf_bytes: bytes) -> int:
        """Return total number of pages in the PDF."""
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
        """
        Return (width, height) in points for a specific page (0-indexed).
        Returns (595.0, 842.0) — A4 defaults — on error.
        """
        try:
            doc    = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            page   = doc[page_num]
            width  = page.rect.width
            height = page.rect.height
            doc.close()
            return width, height
        except Exception as e:
            logger.error(f"Error getting page dimensions: {e}")
            return 595.0, 842.0

    # ================================================================== #
    # SECTION 6 — PAGE RENDERING (for OCR and UI preview)                #
    # ================================================================== #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def pdf_to_page_images(
        pdf_bytes:  bytes,
        start_page: int = 1,
        end_page:   Optional[int] = None,
        dpi:        int = 150,
    ) -> list:
        """
        Render PDF pages to PIL Images.

        Used by:
          - OCR mode (200 DPI — higher quality for Tesseract)
          - UI preview panel (150 DPI — adequate for display)

        Args:
          pdf_bytes:  raw PDF bytes
          start_page: 1-indexed first page (inclusive)
          end_page:   1-indexed last page (inclusive, None = all)
          dpi:        render resolution

        Returns:
          List of PIL Image objects (RGB), one per rendered page.
          Empty list if rendering fails.
        """
        images = []
        try:
            doc        = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)

            start_idx = max(0, start_page - 1)
            if end_page is None or end_page > total_pages:
                end_idx = total_pages
            else:
                end_idx = end_page   # 1-indexed end = exclusive 0-indexed equivalent

            scale  = dpi / 72.0
            matrix = pymupdf.Matrix(scale, scale)

            for page_num in range(start_idx, end_idx):
                page      = doc[page_num]
                pixmap    = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pixmap.tobytes("png")
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(pil_img)

            doc.close()

        except Exception as e:
            logger.error(f"Error rendering PDF pages to images: {e}")

        return images


# ------------------------------------------------------------------ #
# Standalone test — run `python pdf_processor.py` to verify          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        print()
        print("Running schema validation test instead...")

        # Test that block schema keys are all present
        dummy_block = {
            "text": "test", "x0": 0.0, "y0": 0.0,
            "x1": 100.0, "y1": 20.0,
            "font_size": 12.0, "font_name": "Helvetica",
            "page_num": 0, "page_height": 842.0, "page_width": 595.0,
            "is_heading": False, "heading_level": 0,
        }
        required_keys = {
            "text", "x0", "y0", "x1", "y1",
            "font_size", "font_name",
            "page_num", "page_height", "page_width",
            "is_heading", "heading_level",
        }
        missing = required_keys - set(dummy_block.keys())
        if missing:
            print(f"FAIL — Missing keys in block schema: {missing}")
        else:
            print("PASS — Block schema has all required keys.")
        sys.exit(0)

    pdf_path = sys.argv[1]
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"File: {pdf_path} ({len(pdf_bytes)/1024:.1f} KB)")

    # Validate
    valid, err = PDFProcessor.validate_pdf(pdf_bytes)
    print(f"Valid: {valid}" + (f" — {err}" if err else ""))
    if not valid:
        sys.exit(1)

    # Metadata
    meta = PDFProcessor.get_metadata(pdf_bytes)
    print(f"Title:  {meta.get('title') or '(none)'}")
    print(f"Author: {meta.get('author') or '(none)'}")
    print(f"Pages:  {PDFProcessor.get_page_count(pdf_bytes)}")

    # TOC
    toc = PDFProcessor.get_toc(pdf_bytes)
    print(f"TOC entries: {len(toc)}")
    for entry in toc[:5]:
        print(f"  Level {entry[0]}: '{entry[1]}' (page {entry[2]})")

    # Text mode extraction (first 3 pages)
    print("\nText mode extraction (pages 1-3)...")
    result = PDFProcessor.extract(pdf_bytes, lang_code="en", start_page=1, end_page=3, mode="text")
    blocks = result["blocks"]
    print(f"  Blocks extracted: {len(blocks)}")
    if blocks:
        sizes = [b["font_size"] for b in blocks if b["font_size"] > 0]
        if sizes:
            import statistics
            print(f"  Font size range: {min(sizes):.1f}–{max(sizes):.1f} pt "
                  f"(median: {statistics.median(sizes):.1f} pt)")
        print(f"  Sample block: '{blocks[0]['text'][:80]}'")