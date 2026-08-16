"""Generic "read the cell next to the label" resolution.

Policy schedules put a label's value in one of a small number of places, and *which* place
varies by insurer. Rather than encode per-insurer positions, this module enumerates the
handful of possibilities and lets the caller score them:

* **right of the label, same row** -- Niva Bupa: ``ICU  2%``
* **the column beneath the label** -- Care Health: a header row
  ``Sum Insured | ...Normal Hospitalization | ...ICU Hospitalization`` over data rows
  ``Rs. 300,000 | 2 % of Sum Insured per day | 4 % of Sum Insured per day``
* **left of the label, same row** -- prose form: ``Rs. 75,000 for Normal``
* **the next non-empty row** -- vertical label/value stacks

Cells are split on runs of two or more spaces (the gutter width that layout-sorted
extraction produces) or on explicit pipes from rendered tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

_CELL_SPLIT = re.compile(r"\s{2,}|\s*\|\s*")


@dataclass
class Cell:
    text: str
    #: Character offset of the cell within the page text.
    start: int
    #: Position of the cell within its row.
    index: int


@dataclass
class Row:
    cells: List[Cell]
    line_no: int
    start: int
    raw: str

    @property
    def is_multi_cell(self) -> bool:
        return len(self.cells) > 1


def build_rows(text: str) -> List[Row]:
    """Split page text into rows of cells, tracking character offsets."""
    rows: List[Row] = []
    offset = 0
    for line_no, raw in enumerate(text.split("\n")):
        cells: List[Cell] = []
        cursor = 0
        for index, piece in enumerate(_CELL_SPLIT.split(raw)):
            stripped = piece.strip()
            if stripped:
                position = raw.find(stripped, cursor)
                if position < 0:
                    position = cursor
                cells.append(Cell(text=stripped, start=offset + position, index=len(cells)))
                cursor = position + len(stripped)
        if cells:
            # Re-index after dropping blanks so column positions stay contiguous.
            for index, cell in enumerate(cells):
                cell.index = index
            rows.append(Row(cells=cells, line_no=line_no, start=offset, raw=raw))
        offset += len(raw) + 1
    return rows


@dataclass
class Neighbour:
    """A candidate value string near a label, with how it was found."""

    text: str
    strategy: str
    #: Character distance from the label occurrence -- smaller is better.
    distance: int
    offset: int


def find_label_hits(rows: List[Row], cue: str) -> List[Tuple[int, int, int]]:
    """``(row_index, cell_index, position_within_cell)`` for every occurrence of ``cue``."""
    cue_lower = cue.lower()
    hits: List[Tuple[int, int, int]] = []
    for row_index, row in enumerate(rows):
        for cell in row.cells:
            position = cell.text.lower().find(cue_lower)
            if position >= 0:
                hits.append((row_index, cell.index, position))
    return hits


def neighbours(rows: List[Row], row_index: int, cell_index: int, position: int, cue: str,
               *, column_depth: int = 4) -> List[Neighbour]:
    """Every plausible value string for a label found at the given location."""
    row = rows[row_index]
    cell = row.cells[cell_index]
    cue_end = position + len(cue)
    results: List[Neighbour] = []

    # 1. Remainder of the label's own cell, to the right of the cue.
    inside_right = cell.text[cue_end:].strip(" :-|\t")
    if inside_right:
        results.append(Neighbour(inside_right, "same_cell_right", 1, cell.start + cue_end))

    # 2. Cells to the right on the same row -- the most common schedule layout.
    for offset, other in enumerate(row.cells[cell_index + 1:], start=1):
        results.append(Neighbour(other.text, "row_right", offset * 2, other.start))
        if offset >= 3:
            break

    # 3. The column beneath the label (header row over data rows).
    depth = 0
    for below in rows[row_index + 1:]:
        if depth >= column_depth:
            break
        if not below.is_multi_cell and len(below.cells) <= cell_index:
            continue
        if cell_index < len(below.cells):
            depth += 1
            results.append(Neighbour(below.cells[cell_index].text, "column_below",
                                     3 + depth, below.cells[cell_index].start))

    # 4. Text to the left inside the same cell -- "Rs. 75,000 for Normal".
    inside_left = cell.text[:position].strip(" :-|\t")
    if inside_left:
        results.append(Neighbour(inside_left, "same_cell_left", 2, cell.start))

    # 5. Cells to the left on the same row.
    for offset, other in enumerate(reversed(row.cells[:cell_index]), start=1):
        results.append(Neighbour(other.text, "row_left", 3 + offset * 2, other.start))
        if offset >= 2:
            break

    # 6. The next rows wholesale -- vertical label/value stacks.
    for offset, below in enumerate(rows[row_index + 1:row_index + 3], start=1):
        results.append(Neighbour(below.raw.strip(), "next_row", 6 + offset, below.start))

    return results


def region_text(text: str, centre: int, before: int, after: int) -> Tuple[str, int]:
    """A bounded window around ``centre``; returns ``(window, window_start_offset)``."""
    start = max(0, centre - before)
    end = min(len(text), centre + after)
    return text[start:end], start


#: A physical line this long that does not end with terminal punctuation is a *wrapped*
#: continuation of a sentence, not a complete row. Below this length it is far more likely
#: to be a short table cell pair ("Pre-Existing Disease (PED)  Waived Off").
_WRAP_MIN_LENGTH = 70
_TERMINATORS = ('.', ';', ':', '!', '?', '"', "'", '\u201d', '\u2019', ',')


def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    offset = 0
    for line in text.split("\n"):
        spans.append((offset, offset + len(line), line))
        offset += len(line) + 1
    return spans


def _continues(line: str) -> bool:
    """True when ``line`` looks like it was wrapped mid-sentence."""
    stripped = line.rstrip()
    if len(stripped) < _WRAP_MIN_LENGTH:
        return False
    return not stripped.endswith(_TERMINATORS)


def sentence_at(text: str, index: int, max_len: int = 460, max_extend: int = 3) -> str:
    """The *logical* line containing ``index``.

    PDF text is hard-wrapped, so a single ``\\n`` is usually not a sentence boundary. Naively
    stopping at the newline truncated
    ``"Infertility & related ailments ... are outside the scope of this policy."`` right
    before the words that carry the verdict, which silently turned an exclusion into a
    not-found. Wrapped lines are therefore re-joined, while short label/value rows are left
    alone so that three consecutive ``"... Waived Off"`` rows do not merge into one.
    """
    spans = _line_spans(text)
    if not spans:
        return ""
    position = 0
    for i, (start, end, _line) in enumerate(spans):
        if start <= index <= end:
            position = i
            break
    else:
        position = len(spans) - 1

    first = position
    for _ in range(max_extend):
        if first == 0:
            break
        if _continues(spans[first - 1][2]):
            first -= 1
        else:
            break

    last = position
    for _ in range(max_extend):
        if last + 1 >= len(spans):
            break
        if _continues(spans[last][2]):
            last += 1
        else:
            break

    joined = " ".join(spans[i][2].strip() for i in range(first, last + 1) if spans[i][2].strip())
    return re.sub(r"\s{2,}", "  ", joined).strip()[:max_len]
