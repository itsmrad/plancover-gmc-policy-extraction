"""Deterministic, evidence-carrying extractor.

Reads a :class:`~gmc_extract.ingestion.PolicyDocument` against the declarative
:mod:`.field_specs` catalogue and returns the best-scoring candidate per field.

The scoring model, in one place so it can be argued with:

===========================  =========================================================
Term                         Rationale
===========================  =========================================================
cue specificity              Earlier cues in a spec are more specific; a match on
                             "maximum eligibility for icu hospitalization" should beat a
                             bare "icu".
anchor quality               A cue found near a *label-shaped* anchor (first cell of its
                             row) outranks one found near the same words buried in a
                             prose paragraph. This is what stops "Room Rent actually
                             incurred" mid-sentence from dragging in the maternity
                             "Normal 25,000" figure.
label shape                  Same idea for the cue itself: schedule labels start their
                             cell.
strategy                     Value to the right of a label is more likely than value to
                             the left, which is more likely than two rows down.
distance                     Nearer is better, both from the cue and from the anchor.
===========================  =========================================================

Weights are coarse on purpose. Tuning them precisely against five documents would be
overfitting; what matters is that the ordering is right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from ..ingestion import PolicyDocument
from ..schema import FieldStatus, ProductType, ValueUnit
from . import tabular
from .field_specs import ALL_SPECS, TEXTUAL_LIMITS, FieldSpec, ValueKind
from .parsers import (
    all_days,
    all_money,
    parse_basis,
    parse_date,
    parse_months,
    parse_percent,
    parse_status,
)

_NULLISH = {"na", "n/a", "nil", "none", "not specified", "-", "--", "nan", "not applicable"}

#: A count cell must be essentially just a number. Without this guard, prose such as
#: "Dependents of Primary members shall be declared at the inception of the Policy" was
#: yielding a headcount from an unrelated digit.
_COUNT_CELL = re.compile(
    r"^(?:nos?\.?\s*|total\s*)?(\d{1,6})"
    r"(?:\s*(?:lives|members|employees|persons|nos?\.?|numbers?|adults|children))?$",
    re.IGNORECASE,
)

#: A percentage sitting next to cost-sharing wording is a co-pay, not a benefit limit.
_COPAY_CONTEXT = re.compile(r"co\s*-?\s*pay(?:ment)?|deductible|co\s*-?\s*insurance",
                            re.IGNORECASE)

_STRATEGY_BONUS = {
    "row_right": 1.5,
    "same_cell_right": 1.2,
    "column_below": 1.0,
    "same_cell_left": 0.8,
    "row_left": 0.4,
    "next_row": 0.3,
}

#: Cap anchors per cue so a word like "maternity" appearing 20 times cannot explode the
#: candidate set. Anchors are already ordered by specificity.
_MAX_ANCHORS_PER_CUE = 6


@dataclass
class Candidate:
    """A scored extraction result with its provenance."""

    status: FieldStatus
    value: Optional[Union[float, str]] = None
    unit: Optional[ValueUnit] = None
    basis: Optional[str] = None
    raw_text: str = ""
    page: Optional[int] = None
    score: float = 0.0
    strategy: str = ""
    cue: str = ""
    needs_review: bool = False
    note: Optional[str] = None


# --------------------------------------------------------------------------------------
# Value parsing per kind
# --------------------------------------------------------------------------------------
#: Canonical forms for textual limits, so "covered upto sum insured" and "upto Sum Insured"
#: land on the same QMS cell value instead of two near-duplicate strings.
_LIMIT_CANONICAL = {
    "covered upto sum insured": "up to sum insured",
    "upto sum insured": "up to sum insured",
    "up to sum insured": "up to sum insured",
    "upto si": "up to sum insured",
    "as per sum insured": "up to sum insured",
    "sum insured": "up to sum insured",
    "no capping": "no limit",
    "no sub-limit": "no limit",
    "no sublimit": "no limit",
    "no restriction": "no limit",
    "actuals": "at actuals",
}


def _textual_limit(text: str) -> Optional[str]:
    lowered = " ".join(text.lower().split())
    for phrase in TEXTUAL_LIMITS:
        if phrase in lowered:
            return _LIMIT_CANONICAL.get(phrase, phrase)
    return None


def _is_nullish(text: str) -> bool:
    """Whether the cell *is* an explicit null token.

    Matched exactly rather than after stripping punctuation. Stripping ``"."`` used to
    reduce a stray full stop to the empty string, which then matched the null set and beat
    the real value with a high score -- the maternity limits came back as "not specified"
    while ``Rs. 50,000`` sat one cell away.
    """
    return " ".join(text.lower().split()) in _NULLISH


def _pick_money(text: str, *, prefer_last: bool, require_currency: bool
                ) -> Optional[Tuple[float, str]]:
    """Choose which amount in ``text`` is the value.

    ``prefer_last`` is used when the label sits to the *right* of its value, as in
    "Rs. 75,000 for Normal": the nearest amount is then the last one, not the first.
    Getting this backwards silently swaps Normal and C-Section limits when they differ.
    """
    amounts = all_money(text, require_currency=require_currency)
    if not amounts and require_currency:
        amounts = all_money(text, require_currency=False)
    if not amounts:
        return None
    return amounts[-1] if prefer_last else amounts[0]


def _parse_value(text: str, spec: FieldSpec, *, prefer_last: bool) -> Optional[Candidate]:
    """Turn a candidate value string into a typed :class:`Candidate` (unscored)."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()

    if _is_nullish(cleaned):
        return Candidate(status=FieldStatus.NOT_SPECIFIED, raw_text=cleaned,
                         note="document states the value as NA/nil")

    kind = spec.kind

    if kind is ValueKind.MONEY:
        picked = _pick_money(cleaned, prefer_last=prefer_last,
                             require_currency=spec.require_currency)
        if picked:
            return Candidate(status=FieldStatus.PRESENT, value=picked[0], unit=ValueUnit.INR,
                             basis=parse_basis(cleaned), raw_text=cleaned)
        limit = _textual_limit(cleaned)
        if limit:
            return Candidate(status=FieldStatus.COVERED, value=limit, unit=ValueUnit.TEXT,
                             raw_text=cleaned)
        return None

    if kind is ValueKind.PERCENT_OR_MONEY:
        percent = parse_percent(cleaned)
        if percent:
            value, of_si, _raw = percent
            if not of_si and spec.percent_defaults_to_sum_insured:
                of_si = True
            return Candidate(
                status=FieldStatus.COVERED, value=value,
                unit=ValueUnit.PERCENT_OF_SUM_INSURED if of_si else ValueUnit.PERCENT,
                basis=parse_basis(cleaned), raw_text=cleaned,
            )
        picked = _pick_money(cleaned, prefer_last=prefer_last, require_currency=True)
        if picked:
            return Candidate(status=FieldStatus.COVERED, value=picked[0], unit=ValueUnit.INR,
                             basis=parse_basis(cleaned), raw_text=cleaned)
        limit = _textual_limit(cleaned)
        if limit:
            return Candidate(status=FieldStatus.COVERED, value=limit, unit=ValueUnit.TEXT,
                             raw_text=cleaned)
        return None

    if kind is ValueKind.PERCENT:
        percent = parse_percent(cleaned)
        if percent:
            value, of_si, _raw = percent
            return Candidate(
                status=FieldStatus.APPLIED, value=value,
                unit=ValueUnit.PERCENT_OF_SUM_INSURED if of_si else ValueUnit.PERCENT,
                basis=parse_basis(cleaned), raw_text=cleaned,
            )
        return None

    if kind is ValueKind.DAYS:
        days = all_days(cleaned)
        if days:
            index = min(spec.duration_index, len(days) - 1)
            value, _raw = days[index]
            return Candidate(status=FieldStatus.COVERED, value=float(value),
                             unit=ValueUnit.DAYS, raw_text=cleaned)
        months = parse_months(cleaned)
        if months:
            return Candidate(status=FieldStatus.COVERED, value=float(months[0]),
                             unit=ValueUnit.MONTHS, raw_text=cleaned)
        return None

    if kind is ValueKind.MONTHS:
        months = parse_months(cleaned)
        if months:
            return Candidate(status=FieldStatus.APPLIED, value=float(months[0]),
                             unit=ValueUnit.MONTHS, raw_text=cleaned)
        return None

    if kind is ValueKind.COUNT:
        match = _COUNT_CELL.match(cleaned)
        if match:
            return Candidate(status=FieldStatus.PRESENT, value=float(match.group(1)),
                             unit=ValueUnit.COUNT, raw_text=cleaned)
        return None

    if kind is ValueKind.DATE:
        parsed = parse_date(cleaned)
        if parsed:
            return Candidate(status=FieldStatus.PRESENT, value=parsed[0].isoformat(),
                             unit=ValueUnit.TEXT, raw_text=cleaned)
        return None

    # TEXT
    compact = " ".join(cleaned.split())
    # Reject separator-only cells (":", "|", "-"). Layout-sorted rows often place a lone
    # colon between a label and its value, and it would otherwise win on proximity.
    if not any(char.isalnum() for char in compact):
        return None
    compact = _ENUM_PREFIX.sub("", compact)
    compact = _trim_address_noise(compact) if spec.path.endswith("policyholder_name") else compact
    if not compact:
        return None
    return Candidate(status=FieldStatus.PRESENT, value=compact, unit=ValueUnit.TEXT,
                     raw_text=cleaned)


_ENUM_PREFIX = re.compile(r"^\d{1,2}\s*[.)]\s*")


_DIGIT_TOKEN = re.compile(r"\s\S*\d\S*")
_CAPS_TOKEN = re.compile(r"^[A-Z&.,'/()-]{2,}$")


def _trim_address_noise(text: str) -> str:
    """Trim a company name that layout-sorted extraction glued to its address.

    Two independent signals, applied in order:

    1. An ALL-CAPS company name followed by a mixed-case token is a name/address join
       ("MUKUNDA FOODS PVT LTD ground and 1st floor no s 13/5").
    2. Otherwise cut at the first digit-bearing token -- company names rarely contain
       digits, street addresses almost always do ("AAYUV TECHNOLOGIES D35, MADHURA NAGAR").
    """
    tokens = text.split()
    caps_run = 0
    for token in tokens:
        if _CAPS_TOKEN.match(token):
            caps_run += 1
        else:
            break
    if caps_run >= 2 and caps_run < len(tokens):
        return " ".join(tokens[:caps_run]).strip(" ,;-")

    match = _DIGIT_TOKEN.search(text)
    return text[:match.start()].strip(" ,;-") if match else text


def _status_candidate(page_text: str, offset: int, spec: FieldSpec) -> Optional[Candidate]:
    """Resolve a coverage verdict (and any stated limit) from the cue's sentence."""
    sentence = tabular.sentence_at(page_text, offset)
    if not sentence:
        return None

    status, phrase = parse_status(sentence, spec.status_mode)

    # Look one line further only when that line is a bare verdict ("Waived Off",
    # "Not Covered"). An earlier, looser version of this rule read the *next paragraph* and
    # turned "Day Care Treatment: list attached" into NOT_COVERED by picking up the words
    # "List of Expenses Generally Excluded" from an unrelated line.
    if status is FieldStatus.NOT_FOUND:
        newline = page_text.find("\n", offset)
        if newline >= 0:
            follow = tabular.sentence_at(page_text, newline + 1)
            if follow and len(follow) <= 45:
                status, phrase = parse_status(follow, spec.status_mode)
                if status is not FieldStatus.NOT_FOUND:
                    sentence = f"{sentence} | {follow}"

    candidate = Candidate(status=status, raw_text=sentence, note=None)

    if spec.kind is ValueKind.STATUS_WITH_LIMIT:
        percent = parse_percent(sentence)
        money = _pick_money(sentence, prefer_last=False, require_currency=True)
        copay_context = bool(_COPAY_CONTEXT.search(sentence))
        if percent and not copay_context:
            value, of_si, _raw = percent
            candidate.value = value
            candidate.unit = (ValueUnit.PERCENT_OF_SUM_INSURED if of_si
                              else ValueUnit.PERCENT)
            candidate.basis = parse_basis(sentence)
        elif money:
            candidate.value = money[0]
            candidate.unit = ValueUnit.INR
            candidate.basis = parse_basis(sentence)
        else:
            limit = _textual_limit(sentence)
            if limit:
                candidate.value = limit
                candidate.unit = ValueUnit.TEXT
        if percent and copay_context:
            # Record it, but do not present a cost-sharing percentage as a benefit limit.
            candidate.note = (f"a {percent[0]:g}% co-payment/cost-sharing term applies to "
                              "this benefit")
        # A stated limit implies the benefit exists, even without a "covered" verb
        # ("Emergency Ambulance  INR 1000 per hospitalization").
        if candidate.value is not None and candidate.status is FieldStatus.NOT_FOUND:
            candidate.status = FieldStatus.COVERED
            candidate.note = "coverage inferred from a stated limit"

    if candidate.status is FieldStatus.NOT_FOUND and spec.presence_implies_covered:
        candidate.status = (FieldStatus.APPLIED if spec.presence_status == "applied"
                            else FieldStatus.COVERED)
        candidate.needs_review = True
        candidate.note = candidate.note or (
            "benefit named but no explicit coverage wording found nearby; status inferred "
            "from the presence of the clause"
        )

    if candidate.status is FieldStatus.NOT_FOUND:
        return None
    if phrase:
        candidate.note = candidate.note or f"matched polarity phrase: '{phrase}'"
    return candidate


def _block_at(page_text: str, offset: int) -> str:
    """The blank-line-delimited paragraph containing ``offset``.

    Used for long enumerations such as Niva Bupa's disease-wise capping schedule, where the
    value spans four wrapped lines and truncating to one line would lose most of it.
    """
    start = page_text.rfind("\n\n", 0, offset)
    end = page_text.find("\n\n", offset)
    start = 0 if start < 0 else start + 2
    end = len(page_text) if end < 0 else end
    return " ".join(page_text[start:end].split())


# --------------------------------------------------------------------------------------
# Region (anchor) resolution
# --------------------------------------------------------------------------------------
def _anchor_regions(page_text: str, rows: List[tabular.Row], spec: FieldSpec
                    ) -> List[Tuple[int, int, float, int]]:
    """``(start, end, anchor_bonus, anchor_offset)`` windows to search within."""
    if not spec.anchor_cues:
        return [(0, len(page_text), 0.0, -1)]

    regions: List[Tuple[int, int, float, int]] = []
    for cue_index, anchor in enumerate(spec.anchor_cues):
        hits = tabular.find_label_hits(rows, anchor)[:_MAX_ANCHORS_PER_CUE]
        for row_index, cell_index, position in hits:
            cell = rows[row_index].cells[cell_index]
            offset = cell.start + position
            # A label-shaped anchor (starts the first cell of its row) is far stronger
            # evidence of a section heading than the same words inside a paragraph.
            if position == 0 and cell_index == 0:
                bonus = 3.0
            elif position == 0:
                bonus = 1.5
            else:
                bonus = 0.0
            bonus -= cue_index * 0.3
            start = max(0, offset - spec.window_before)
            end = min(len(page_text), offset + spec.window_after)
            regions.append((start, end, bonus, offset))
    return regions or [(0, len(page_text), 0.0, -1)]


def _has_negative_cue(page_text: str, offset: int, spec: FieldSpec, span: int = 130) -> bool:
    if not spec.negative_cues:
        return False
    window = page_text[max(0, offset - span):offset + span].lower()
    return any(cue in window for cue in spec.negative_cues)


_ENUMERATION = re.compile(r"^\s*\(?\d{1,2}\s*[.)]\s*$")
_PROSE_TAIL = re.compile(r"[A-Za-z]\s*$")


def _prefix_bonus(prefix: str, cell_index: int) -> float:
    """Score a cue that is not at the start of its cell, based on what precedes it.

    A cue preceded by nothing but an enumeration marker ("2. Family Structure") is a label.
    A cue preceded by a word is a *prose mention* ("...to be covered under Family Structure,
    then the same needs to be declared..."), which should lose to the real label. Without
    this penalty the prose mention on a later page won the family-structure field purely by
    sitting closer to some text.
    """
    stripped = prefix.strip()
    if not stripped:
        return 1.5 if cell_index == 0 else 0.8
    if _ENUMERATION.match(prefix):
        return 1.5 if cell_index == 0 else 0.8
    if _PROSE_TAIL.search(prefix):
        return -1.0
    return 0.0


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
def _candidates_for_spec(document: PolicyDocument, spec: FieldSpec) -> List[Candidate]:
    found: List[Candidate] = []

    for page in document.pages:
        page_text = page.search_text
        if not page_text.strip():
            continue
        rows = tabular.build_rows(page_text)
        regions = _anchor_regions(page_text, rows, spec)

        for cue_index, cue in enumerate(spec.cues):
            hits = tabular.find_label_hits(rows, cue)
            if not hits:
                continue
            cue_score = 4.0 - cue_index * 0.45

            for row_index, cell_index, position in hits:
                cell = rows[row_index].cells[cell_index]
                offset = cell.start + position
                if _has_negative_cue(page_text, offset, spec):
                    continue

                containing = [r for r in regions if r[0] <= offset < r[1]]
                if not containing:
                    continue
                # Use the strongest region that contains this hit.
                start, end, anchor_bonus, anchor_offset = max(containing, key=lambda r: r[2])
                anchor_penalty = (abs(offset - anchor_offset) / 300.0
                                  if anchor_offset >= 0 else 0.0)

                if position == 0 and cell_index == 0:
                    label_bonus = 1.5
                elif position == 0:
                    label_bonus = 0.8
                else:
                    label_bonus = _prefix_bonus(cell.text[:position], cell_index)

                base = cue_score + anchor_bonus + label_bonus - anchor_penalty

                # Status-style fields are resolved from the sentence, not from a cell.
                if spec.kind in (ValueKind.STATUS, ValueKind.STATUS_WITH_LIMIT):
                    candidate = _status_candidate(page_text, offset, spec)
                    if candidate:
                        candidate.score = base + 1.0
                        candidate.page = page.number
                        candidate.cue = cue
                        candidate.strategy = "sentence"
                        found.append(candidate)
                    continue

                if spec.capture_block or spec.capture_sentence:
                    captured = (_block_at(page_text, offset) if spec.capture_block
                                else tabular.sentence_at(page_text, offset))
                    if captured:
                        candidate = _parse_value(captured, spec, prefer_last=False)
                        if candidate:
                            candidate.score = base + 1.0
                            candidate.page = page.number
                            candidate.cue = cue
                            candidate.strategy = ("block" if spec.capture_block
                                                  else "sentence")
                            found.append(candidate)
                    continue

                region_basis = (parse_basis(page_text[start:end])
                                if spec.kind is ValueKind.PERCENT_OR_MONEY else None)

                for neighbour in tabular.neighbours(rows, row_index, cell_index, position,
                                                   cue):
                    prefer_last = neighbour.strategy in ("same_cell_left", "row_left")
                    candidate = _parse_value(neighbour.text, spec, prefer_last=prefer_last)
                    if not candidate:
                        continue
                    if candidate.basis is None and region_basis:
                        # "Room rent/day & ICU/day" states the basis on the label, not in
                        # the value cell.
                        candidate.basis = region_basis
                    candidate.score = (base
                                       + _STRATEGY_BONUS.get(neighbour.strategy, 0.0)
                                       - neighbour.distance * 0.15)
                    candidate.page = page.number
                    candidate.cue = cue
                    candidate.strategy = neighbour.strategy
                    found.append(candidate)
    return found


def extract_fields(document: PolicyDocument, product_type: ProductType
                   ) -> Dict[str, Candidate]:
    """Best rule-based candidate per field path, for the applicable specs only."""
    results: Dict[str, Candidate] = {}
    for spec in ALL_SPECS:
        if product_type not in spec.products:
            continue
        candidates = _candidates_for_spec(document, spec)
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.score)
        # Flag genuine ambiguity: two candidates scoring nearly the same with different
        # values means a human should look, rather than us picking arbitrarily.
        rivals = [c for c in candidates
                  if c.value != best.value and c.score > best.score - 0.5]
        if rivals:
            best.needs_review = True
            rival = max(rivals, key=lambda c: c.score)
            note = f"competing candidate: {rival.value!r} from {rival.strategy}"
            best.note = f"{best.note}; {note}" if best.note else note
        results[spec.path] = best
    return results
