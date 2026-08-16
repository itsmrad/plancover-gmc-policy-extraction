"""PDF -> normalised in-memory document.

Produces a :class:`PolicyDocument` whose ``text`` is one continuous, cleaned character
stream with an offset->page index. Downstream extractors work on character offsets rather
than page objects, which is what lets a cue window span a line break or a table row without
special-casing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from . import layout, ocr

LOGGER = logging.getLogger(__name__)

_PAGE_MARK = "\n\n===== PAGE {n} =====\n"


@dataclass
class Page:
    number: int  # 1-based
    text: str  # layout-sorted + scrubbed
    table_text: str = ""
    used_ocr: bool = False

    @property
    def search_text(self) -> str:
        """Text plus pipe-rendered tables: the surface all extractors search."""
        if self.table_text:
            return f"{self.text}\n\n[TABLES]\n{self.table_text}"
        return self.text


@dataclass
class PolicyDocument:
    path: str
    file_name: str
    sha256: str
    pages: List[Page] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Built in __post_init__-style by :func:`load_document`.
    text: str = ""
    _page_spans: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_pages(self) -> List[int]:
        return [p.number for p in self.pages if p.used_ocr]

    @property
    def has_text_layer(self) -> bool:
        return any(not p.used_ocr and len(p.text.strip()) >= ocr.MIN_CHARS_FOR_TEXT_LAYER
                   for p in self.pages)

    def page_at(self, offset: int) -> Optional[int]:
        """Map a character offset in ``self.text`` back to a 1-based page number."""
        for start, end, number in self._page_spans:
            if start <= offset < end:
                return number
        return self.pages[-1].number if self.pages else None

    def page_text(self, number: int) -> str:
        for page in self.pages:
            if page.number == number:
                return page.search_text
        return ""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_tables(path: str, page_index: int) -> str:
    """Best-effort table extraction. Never fatal: tables are an accuracy bonus, not a
    requirement, and pdfplumber can legitimately fail on unusual page structures."""
    try:
        import pdfplumber

        with pdfplumber.open(path, pages=[page_index + 1]) as pdf:
            if not pdf.pages:
                return ""
            return layout.tables_to_text(pdf.pages[0].extract_tables())
    except Exception as exc:  # pragma: no cover - depends on PDF internals
        LOGGER.debug("table extraction failed on page %s of %s: %s", page_index + 1, path, exc)
        return ""


def load_document(path: str, *, extract_tables: bool = True,
                  allow_ocr: bool = True) -> PolicyDocument:
    """Load and normalise a policy PDF."""
    document = PolicyDocument(
        path=path,
        file_name=os.path.basename(path),
        sha256=_sha256(path),
    )

    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf):
            # sort=True is the single most important accuracy decision in ingestion:
            # it reorders spans by geometry so labels sit next to their values.
            raw = page.get_text("text", sort=True)
            text = layout.scrub(raw)
            used_ocr = False

            if len(text.strip()) < ocr.MIN_CHARS_FOR_TEXT_LAYER:
                # Retry in reading order before paying for OCR -- occasionally sort=True
                # loses text on pages with unusual span geometry.
                fallback = layout.scrub(page.get_text("text"))
                if len(fallback.strip()) > len(text.strip()):
                    text = fallback
            if len(text.strip()) < ocr.MIN_CHARS_FOR_TEXT_LAYER and allow_ocr:
                if ocr.ocr_available():
                    ocr_text = layout.scrub(ocr.ocr_page(page))
                    if ocr_text.strip():
                        text, used_ocr = ocr_text, True
                else:
                    document.warnings.append(
                        f"page {index + 1} has no usable text layer and OCR is unavailable "
                        "(install pytesseract + the tesseract binary to enable it)"
                    )

            table_text = _extract_tables(path, index) if extract_tables else ""
            document.pages.append(
                Page(number=index + 1, text=text, table_text=table_text, used_ocr=used_ocr)
            )

    # Build the continuous stream and the offset -> page index.
    buffer: List[str] = []
    cursor = 0
    for page in document.pages:
        marker = _PAGE_MARK.format(n=page.number)
        body = page.search_text + "\n"
        buffer.append(marker)
        buffer.append(body)
        start = cursor + len(marker)
        cursor = start + len(body)
        document._page_spans.append((start, cursor, page.number))
    document.text = "".join(buffer)

    if not document.has_text_layer:
        document.warnings.append(
            "document appears to be image-only; extraction quality will be limited"
        )
    return document


def discover_pdfs(input_path: str) -> List[str]:
    """Return the PDFs to process: a single file, or every PDF in a directory."""
    if os.path.isfile(input_path):
        return [input_path]
    found: List[str] = []
    for root, _dirs, files in os.walk(input_path):
        for name in sorted(files):
            if name.lower().endswith(".pdf") and not name.startswith("."):
                found.append(os.path.join(root, name))
    return found
