"""Reconcile the rule and LLM extractors into one QMS field, with a confidence grade.

The reconciliation policy, and the reasoning behind it:

=====================================  ==================  ====================================
Situation                              Result              Why
=====================================  ==================  ====================================
Both agree                             ``rule+llm``, HIGH  Two independent methods reaching the
                                                           same value is the strongest signal
                                                           available without a human.
Both found it, values differ           LLM wins, review    The LLM is better at nested
                                                           conditions and unusual phrasing; the
                                                           rule value is kept as ``alternate``
                                                           so nothing is lost.
Both found it, statuses differ         LLM wins, LOW,      A polarity disagreement is the most
                                       review              consequential kind, so it is graded
                                                           down and flagged.
Rule only                              ``rule``, MEDIUM    Deterministic and evidenced, but
                                       (HIGH if the        unconfirmed.
                                       match was strong)
LLM only                               ``llm``, MEDIUM     Evidence-verified, but a phrasing the
                                                           deterministic layer did not know.
Neither                                ``not_found``       Reported explicitly, never dropped.
=====================================  ==================  ====================================

Flagging a disagreement is the point. A confidently wrong number is a bug; an honest
``needs_review: true`` is a working quality gate.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..schema import Confidence, ExtractionSource, FieldStatus, QMSField, ValueUnit
from .field_specs import SPECS_BY_PATH, FieldSpec
from .llm_extractor import LLMFieldResult
from .parsers import display_for
from .rule_extractor import Candidate

#: A rule candidate scoring at least this well, with no internal ambiguity, is treated as
#: reliable on its own. Calibrated against the sample set: clean label/value hits land
#: around 6-9, while speculative fallbacks land near 3.
_STRONG_RULE_SCORE = 5.5

#: Relative tolerance for "the two extractors agree" on a numeric value.
_NUMERIC_TOLERANCE = 0.01


def _values_agree(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return abs(float(left) - float(right)) / scale <= _NUMERIC_TOLERANCE
    first = " ".join(str(left).lower().split())
    second = " ".join(str(right).lower().split())
    if first == second:
        return True
    # For free-text conditions the two extractors legitimately quote different spans of the
    # same clause -- the rule layer takes the whole sentence, the LLM often the operative
    # phrase. One containing the other is agreement, not a conflict worth flagging.
    if len(first) >= 12 and len(second) >= 12:
        if first in second or second in first:
            return True
    # Long enumerations (disease-wise capping schedules) are quoted with small differences:
    # the rule layer picks up the label glued into the middle of the block by the page
    # layout. Near-identical token sets are the same answer, so this avoids a review flag
    # that a human would immediately dismiss.
    if len(first) >= 200 and len(second) >= 200:
        tokens_a, tokens_b = set(first.split()), set(second.split())
        overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
        if overlap >= 0.9:
            return True
    return False


_VERDICTS = (FieldStatus.COVERED, FieldStatus.NOT_COVERED, FieldStatus.WAIVED_OFF,
             FieldStatus.APPLIED)


def _status_label(status) -> str:
    text = status if isinstance(status, str) else status.value
    return text.replace("_", " ").title()


def _render(value, unit: Optional[ValueUnit], basis: Optional[str],
            status=None) -> Optional[str]:
    """The human-facing cell value.

    A numeric limit speaks for itself ("Rs. 75,000", "2% of sum insured per day"), but a
    textual limit does not: a cell reading just "up to sum insured" loses the verdict, so the
    status is prefixed. A cell with no value at all renders as the verdict alone.
    """
    if value is None:
        return _status_label(status) if status is not None else None
    if unit is ValueUnit.TEXT and status in _VERDICTS:
        return f"{_status_label(status)} - {value}"
    return display_for(value, unit, basis)


def _join_notes(*parts: Optional[str]) -> Optional[str]:
    kept = [part.strip() for part in parts if part and part.strip()]
    return "; ".join(dict.fromkeys(kept)) or None


#: Strategies that mean the rule extractor read a value sitting next to an actual label,
#: rather than inferring one from a sentence.
_LABELLED_STRATEGIES = frozenset({
    "row_right", "column_below", "same_cell_right", "row_left", "same_cell_left",
})


def _is_strong_labelled(rule: Candidate) -> bool:
    """Whether the rule candidate came from an explicitly labelled schedule field.

    This decides who wins a genuine conflict. Niva Bupa's schedule has a labelled
    ``Co-payment  NA`` field, while a Special Conditions paragraph elsewhere mentions a 50%
    co-pay for specific procedures. The LLM reads the paragraph, the rule layer reads the
    labelled field -- and for a QMS column named "Co-payment", the labelled field is the
    answer. The paragraph finding is kept as ``alternate`` and the conflict is flagged.
    """
    return (rule.score >= _STRONG_RULE_SCORE
            and not rule.needs_review
            and rule.strategy in _LABELLED_STRATEGIES)


def merge_field(spec: FieldSpec, rule: Optional[Candidate],
                llm: Optional[LLMFieldResult]) -> QMSField:
    """Combine one field's two candidate answers."""
    if rule is None and llm is None:
        return QMSField.missing(spec.notes)

    if rule is not None and llm is not None:
        status_match = rule.status == llm.status
        value_match = _values_agree(rule.value, llm.value)

        if status_match and value_match:
            return QMSField(
                status=llm.status,
                value=llm.value if llm.value is not None else rule.value,
                unit=llm.unit or rule.unit,
                basis=llm.basis or rule.basis,
                display=_render(llm.value if llm.value is not None else rule.value,
                                llm.unit or rule.unit, llm.basis or rule.basis, llm.status),
                raw_text=llm.evidence or rule.raw_text,
                page=llm.page or rule.page,
                source=ExtractionSource.RULE_AND_LLM,
                confidence=Confidence.HIGH,
                needs_review=rule.needs_review,
                notes=_join_notes(spec.notes, rule.note, llm.notes),
            )

        # Same verdict, but only one extractor volunteered a limit. That is not a conflict --
        # take the more informative answer rather than discarding the limit.
        if status_match and (rule.value is None) != (llm.value is None):
            richer = rule if rule.value is not None else llm
            value = richer.value
            unit = richer.unit
            basis = richer.basis
            return QMSField(
                status=rule.status,
                value=value,
                unit=unit,
                basis=basis,
                display=_render(value, unit, basis, rule.status),
                raw_text=(rule.raw_text if richer is rule else llm.evidence),
                page=(rule.page if richer is rule else llm.page),
                source=ExtractionSource.RULE_AND_LLM,
                confidence=Confidence.MEDIUM,
                needs_review=rule.needs_review,
                notes=_join_notes(spec.notes, rule.note, llm.notes,
                                  "both extractors agreed on the status; the limit was "
                                  "stated by only one of them"),
            )

        # A genuine conflict.
        prefer_rule = _is_strong_labelled(rule)
        if prefer_rule:
            value, unit, basis = rule.value, rule.unit, rule.basis
            status = rule.status
            raw_text, page = rule.raw_text, rule.page
            alternate = _render(llm.value, llm.unit, llm.basis, llm.status)
            alternate = f"LLM extractor: {alternate}"
            reason = ("kept the value from the explicitly labelled schedule field; the LLM "
                      "read a different figure elsewhere in the document")
        else:
            value, unit, basis = llm.value, llm.unit, llm.basis
            status = llm.status
            raw_text, page = (llm.evidence or rule.raw_text), (llm.page or rule.page)
            alternate = _render(rule.value, rule.unit, rule.basis, rule.status)
            alternate = f"rule extractor: {alternate}"
            reason = "preferred the LLM value; it handles nested conditions better"

        return QMSField(
            status=status,
            value=value,
            unit=unit,
            basis=basis,
            display=_render(value, unit, basis, status),
            raw_text=raw_text,
            page=page,
            source=ExtractionSource.RULE_AND_LLM,
            confidence=Confidence.LOW if not status_match else Confidence.MEDIUM,
            needs_review=True,
            alternate=alternate,
            notes=_join_notes(
                spec.notes, llm.notes, rule.note,
                "rule and LLM extractors disagreed"
                + ("" if status_match else
                   f" on status (rule said '{rule.status.value}', "
                   f"LLM said '{llm.status.value}')"),
                reason,
            ),
        )

    if rule is not None:
        strong = rule.score >= _STRONG_RULE_SCORE and not rule.needs_review
        return QMSField(
            status=rule.status,
            value=rule.value,
            unit=rule.unit,
            basis=rule.basis,
            display=_render(rule.value, rule.unit, rule.basis, rule.status),
            raw_text=rule.raw_text,
            page=rule.page,
            source=ExtractionSource.RULE,
            confidence=Confidence.HIGH if strong else Confidence.MEDIUM,
            needs_review=rule.needs_review,
            notes=_join_notes(spec.notes, rule.note),
        )

    assert llm is not None
    return QMSField(
        status=llm.status,
        value=llm.value,
        unit=llm.unit,
        basis=llm.basis,
        display=_render(llm.value, llm.unit, llm.basis, llm.status),
        raw_text=llm.evidence,
        page=llm.page,
        source=ExtractionSource.LLM,
        confidence=Confidence.MEDIUM,
        needs_review=False,
        notes=_join_notes(spec.notes, llm.notes,
                          "found only by the LLM extractor; evidence verified against the "
                          "document"),
    )


def merge_all(rule_results: Dict[str, Candidate],
              llm_results: Dict[str, LLMFieldResult]) -> Dict[str, QMSField]:
    """Merge every field either extractor produced."""
    merged: Dict[str, QMSField] = {}
    for path in set(rule_results) | set(llm_results):
        spec = SPECS_BY_PATH.get(path)
        if spec is None:
            continue
        merged[path] = merge_field(spec, rule_results.get(path), llm_results.get(path))
    return merged
