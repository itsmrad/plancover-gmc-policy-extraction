"""Text normalisation for policy PDFs.

Two problems this module exists to solve, both discovered by inspecting the sample set:

1. **Scrambled reading order.** PyMuPDF's default reading order on the Niva Bupa schedule
   emits every label first and every value ~40 lines later, so ``Policy number`` and
   ``00600900202301`` are nowhere near each other. Layout-sorted extraction
   (``page.get_text("text", sort=True)``) restores ``label ....... value`` rows, which is
   what makes label-proximity extraction viable at all.

2. **PostScript operator leakage.** The Liberty PDFs dump raw content-stream operators
   (``313 292 m``, ``0.1 0 0 0.1 9 0 cm``, ``308 214 170 -164 re``) into the text layer.
   Left in place this noise pollutes both regex windows and LLM context.
"""

from __future__ import annotations

import re
from typing import List

# A content-stream operator line: optional numeric operands followed by a PostScript /
# PDF graphics operator and nothing else.
_PS_OPERATORS = (
    "m l c v y h re f f* F B B* b b* S s n W W* q Q cm w J j M d i gs g G rg RG k K "
    "sh BT ET Tf Td TD Tj TJ T* cs CS sc scn SC SCN ri ro"
).split()
_PS_LINE = re.compile(
    r"^\s*(?:[-+]?\d*\.?\d+\s+)*(?:%s)\s*$" % "|".join(re.escape(op) for op in _PS_OPERATORS)
)
_PS_ARRAY = re.compile(r"^\s*\[\s*\]\s*\d*\s*[dD]\s*$")
_PS_COMMENT = re.compile(r"^\s*%\s*\S*$")

# The Care Health PDFs render the rupee glyph as a standalone backtick in a symbol font.
_LONE_BACKTICK = re.compile(r"^\s*[`\u00b0]+\s*$")

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTISPACE = re.compile(r"[ \t\u00a0]{2,}")
_BLANK_RUN = re.compile(r"\n{3,}")


def is_noise_line(line: str) -> bool:
    """True when a line carries no policy information."""
    stripped = line.strip()
    if not stripped:
        return False  # blank lines are structural, handled separately
    if _LONE_BACKTICK.match(stripped):
        return True
    if _PS_ARRAY.match(stripped) or _PS_COMMENT.match(stripped):
        return True
    # Guard: only treat as an operator line when it is short. Real prose is never this
    # shape, but this keeps a stray "1 M" style false positive from eating a long line.
    if len(stripped) <= 60 and _PS_LINE.match(stripped):
        return True
    return False


def scrub(text: str) -> str:
    """Drop noise lines and normalise whitespace, preserving line structure."""
    text = _ZERO_WIDTH.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    kept: List[str] = []
    for line in text.split("\n"):
        if is_noise_line(line):
            continue
        line = line.replace("\u00a0", " ").replace("`", " ")
        # Collapse the wide gutters that layout-sorted extraction inserts between a label
        # and its value into a single separator, so windows stay compact.
        line = _MULTISPACE.sub("  ", line).rstrip()
        kept.append(line)
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


def normalise_for_matching(text: str) -> str:
    """Lower-cased, punctuation-tolerant form used for cue matching.

    Insurance documents split terms inconsistently ("Waitin g Period" appears verbatim in
    the Care Health PDFs because of glyph spacing, "Pre-Existing" vs "Pre Existing" vs
    "PreExisting"). Collapsing separators makes a single cue list match all variants.
    """
    text = text.lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9%₹.,:/()+&'\-\n ]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def tables_to_text(tables: List[List[List[object]]]) -> str:
    """Render extracted tables as pipe-delimited rows.

    Pipe rows keep a cell's row-mates adjacent in the character stream, which is exactly
    what cue-window extraction needs: a label cell and its value cell end up within a few
    characters of each other even when they were centimetres apart on the page.
    """
    chunks: List[str] = []
    for table in tables:
        rows: List[str] = []
        for row in table:
            cells = [
                _MULTISPACE.sub(" ", str(cell).replace("\n", " ").strip())
                for cell in row
                if cell is not None and str(cell).strip()
            ]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            chunks.append("\n".join(rows))
    return "\n\n".join(chunks)
