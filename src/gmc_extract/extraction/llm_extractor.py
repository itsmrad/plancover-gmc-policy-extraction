"""LLM extraction layer: retrieval + one strict-JSON call per field group.

**Why per group and not one big prompt.** Eight focused prompts of ~10 fields each are
measurably more reliable than one prompt of ~50 fields: the model keeps the whole instruction
in view, the retrieved context is on-topic for every field being asked about, the prompt fits
any context window regardless of document length, and a malformed response degrades one
section rather than the entire document.

**Why the LLM is the second opinion, not the only one.** It generalises to phrasing the cue
lists have never seen, which is exactly the adaptability requirement. But it is
non-deterministic and will confidently invent a plausible limit, so its output is
cross-checked against the rule layer rather than trusted outright (see :mod:`.merge`).

The prompt forbids inference and requires a verbatim ``evidence`` quote for every populated
field. That quote is checked against the document, and a value whose evidence cannot be found
is discarded -- a cheap, effective anti-hallucination guard.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMSettings
from ..ingestion import PolicyDocument
from ..llm import complete_json
from ..schema import FieldStatus, ProductType, ValueUnit
from .field_specs import GROUPS, FieldSpec, ValueKind
from .parsers import StatusMode
from .parsers import all_percent
from .retrieval import retrieve, specs_for_prompt


def _values_close(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return abs(left - right) < 0.01
    return str(left).strip().lower() == str(right).strip().lower()


def _percent_from_evidence(evidence, value):
    """Re-derive a percentage from the model's own verbatim quote.

    gpt-4o-mini returned ``0.5`` for a clause reading "upto 50% of the Sum Insured" -- it
    converted the percentage to a fraction. The quote is copied verbatim from the document, so
    when it contains exactly one percentage that figure is authoritative and the model's
    arithmetic is not. Restricted to the single-percentage case so it cannot pick the wrong
    number out of a clause that mentions several.
    """
    if not evidence or not isinstance(value, float):
        return None
    found = all_percent(evidence)
    if len(found) != 1:
        return None
    quoted = found[0][0]
    if abs(quoted - value) < 0.01:
        return None
    return quoted


#: Basis qualifiers we are willing to publish. A model asked for a "basis" will happily return
#: a fragment of the clause ("as per terms and conditions"), which then pollutes the display
#: string; only recognised qualifiers are kept.
_VALID_BASES = frozenset({
    "per day", "per claim", "per hospitalization", "per family", "per person",
    "per policy period", "per eye", "per week",
})


def _repair_from_evidence(spec: FieldSpec, evidence: Optional[str]):
    """Parse the model's verbatim quote with the rule layer's own parsers.

    Used when the model supplies a good quote but an unusable value or a missing unit. Since
    the quote is copied from the document, running the deterministic parser over it recovers
    exactly what the rule layer would have produced -- which is both more reliable than the
    model's transcription and consistent with the other extractor by construction.
    """
    if not evidence:
        return None
    # Imported here to keep the module-level dependency graph one-directional.
    from .rule_extractor import _parse_value, limit_from_text

    if spec.kind in (ValueKind.STATUS, ValueKind.STATUS_WITH_LIMIT):
        value, unit, basis, _note = limit_from_text(spec, evidence)
        if value is None:
            return None
        return value, unit, basis

    parsed = _parse_value(evidence, spec, prefer_last=False)
    if parsed is None or parsed.value is None:
        return None
    return parsed.value, parsed.unit, parsed.basis

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract structured facts from Indian group insurance policy \
documents for a Quality Management System.

Absolute rules:
1. Use ONLY the supplied document text. Never use outside knowledge or typical market values.
2. If a field is not stated in the supplied text, return status "not_found" and value null. \
Do not guess, and do not infer a value from a similar field.
3. If the document explicitly says NA / Nil / not specified, return status "not_specified".
4. For every field where you return a value or a status other than "not_found", include \
"evidence": a VERBATIM substring copied exactly from the supplied text that justifies it.
5. Normalise amounts to whole rupees as a number: "Rs. 5 Lakh" -> 500000, "5,00,000" -> \
500000, "INR 1000" -> 1000. Never return the digits with commas.
6. Distinguish a benefit LIMIT from a CO-PAYMENT. A co-pay percentage is not a benefit limit; \
mention it in "notes" instead.
7. Reply with a single JSON object and nothing else.

Choosing "status" - each field below lists the only statuses allowed for it:
- "covered"       : the benefit IS provided. Use this even when a limit is stated. A benefit \
with a limit of Rs. 75,000 is "covered" with value 75000, NOT "applied".
- "not_covered"   : the benefit is excluded, not payable, or outside the scope of the policy.
- "waived_off"    : a WAITING PERIOD is waived, or the document says the condition does not \
apply to insured members. "Pre-existing diseases are covered from day one" means the PED \
waiting period is "waived_off".
- "applied"       : a WAITING PERIOD or a cost-sharing condition (co-payment, deductible) DOES \
apply. Never use "applied" for a benefit or for a factual value.
- "present"       : informational, non-coverage fields - policy number, policyholder name, \
premium amounts, dates, head counts, family structure, and free-text conditions. Use this \
whenever the field is a fact rather than a coverage decision.
- "not_specified" : the field appears but the document gives NA / Nil / blank.
- "not_found"     : the field does not appear in the supplied text."""

_STATUS_VALUES = {
    "covered", "not_covered", "waived_off", "applied", "present", "not_specified",
    "not_found",
}
_UNIT_VALUES = {unit.value for unit in ValueUnit}

_KIND_HINTS = {
    ValueKind.MONEY: 'a rupee amount; unit "INR"',
    ValueKind.PERCENT_OR_MONEY: ('either a percentage (unit "percent_of_sum_insured") or a '
                                 'rupee cap (unit "INR") or a textual limit such as '
                                 '"No Limit" / "Sum Insured" (unit "text")'),
    ValueKind.PERCENT: 'a percentage; unit "percent"',
    ValueKind.DAYS: 'a number of days; unit "days"',
    ValueKind.MONTHS: 'a number of months; unit "months"',
    ValueKind.STATUS: 'no value, status only',
    ValueKind.STATUS_WITH_LIMIT: 'status, plus a limit value with its unit if one is stated',
    ValueKind.COUNT: 'a whole-number count; unit "count"',
    ValueKind.DATE: 'a date in YYYY-MM-DD form; unit "text"',
    ValueKind.TEXT: 'a short verbatim string; unit "text"',
}


@dataclass
class LLMFieldResult:
    status: FieldStatus
    value: Optional[Any] = None
    unit: Optional[ValueUnit] = None
    basis: Optional[str] = None
    evidence: Optional[str] = None
    notes: Optional[str] = None
    page: Optional[int] = None


def _field_key(spec: FieldSpec) -> str:
    return spec.path.rsplit(".", 1)[-1]


#: Statuses each field kind may legitimately return, and the status to fall back to when a
#: value was produced but the model chose a word outside that set.
#:
#: Constraining the vocabulary *per field* rather than globally is what stopped the model
#: labelling every extracted figure "applied" -- values such as 86 employees, 30 days and
#: Rs. 75,000 were all correct, but an unconstrained status turned each one into a spurious
#: disagreement with the rule extractor.
_INFORMATIONAL = (FieldStatus.PRESENT, FieldStatus.NOT_SPECIFIED, FieldStatus.NOT_FOUND)
_BENEFIT = (FieldStatus.COVERED, FieldStatus.NOT_COVERED, FieldStatus.NOT_SPECIFIED,
            FieldStatus.NOT_FOUND)
_WAITING = (FieldStatus.WAIVED_OFF, FieldStatus.APPLIED, FieldStatus.NOT_COVERED,
            FieldStatus.NOT_SPECIFIED, FieldStatus.NOT_FOUND)
_CONDITION = (FieldStatus.APPLIED, FieldStatus.NOT_SPECIFIED, FieldStatus.NOT_FOUND)


def allowed_statuses(spec: FieldSpec) -> Tuple[FieldStatus, ...]:
    """The statuses that make sense for a given field."""
    if spec.status_mode is StatusMode.WAITING_PERIOD:
        return _WAITING
    if spec.kind in (ValueKind.STATUS, ValueKind.STATUS_WITH_LIMIT,
                     ValueKind.PERCENT_OR_MONEY):
        return _BENEFIT
    if spec.kind is ValueKind.PERCENT:
        return _CONDITION
    if spec.kind is ValueKind.DAYS:
        # Pre/post hospitalisation is a covered benefit expressed as a duration.
        return _BENEFIT
    return _INFORMATIONAL


def _fallback_status(spec: FieldSpec) -> FieldStatus:
    """The status to use when a value was extracted but the label was out of vocabulary."""
    statuses = allowed_statuses(spec)
    if FieldStatus.COVERED in statuses:
        return FieldStatus.COVERED
    if FieldStatus.PRESENT in statuses:
        return FieldStatus.PRESENT
    return statuses[0]


def _build_prompt(specs: List[FieldSpec], snippets) -> str:
    lines = ["DOCUMENT TEXT (each block tagged with its page):", ""]
    for snippet in snippets:
        lines.append(f"--- page {snippet.page} ---")
        lines.append(snippet.text)
        lines.append("")
    lines.append("FIELDS TO EXTRACT (JSON keys):")
    for spec in specs:
        hint = _KIND_HINTS.get(spec.kind, "")
        permitted = " | ".join(status.value for status in allowed_statuses(spec))
        lines.append(f'- "{_field_key(spec)}": {spec.label}. Expect {hint}. '
                     f'Allowed status: {permitted}.')
    lines.append("")
    lines.append(
        'Reply with: {"<field key>": {"status": ..., "value": ..., "unit": ..., '
        '"basis": ..., "evidence": ..., "notes": ...}, ...} for every field key listed.'
    )
    return "\n".join(lines)


def _normalise_number(value: Any) -> Any:
    """Accept a number, or a numeric string the model wrote with commas or a currency mark."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        if cleaned not in ("", "-", ".", "-."):
            try:
                return float(cleaned)
            except ValueError:
                return value
    return value


def _evidence_present(document: PolicyDocument, evidence: Optional[str]) -> bool:
    """Whether a quoted evidence string really occurs in the document.

    Exact (whitespace-normalised) containment is the fast path. The fallback is deliberately
    *not* a prefix match: an earlier version accepted a quote whose first six words appeared
    in the document, which let ``"Ambulance charges payable up to a maximum of Rs. 9,99,999
    per trip"`` pass against a document that says ``Rs. 1,000`` -- precisely the hallucination
    the guard exists to stop.

    Instead the fallback requires that **every numeric token** in the quote occurs in the
    document (a fabricated limit always carries a number that is not there) and that the bulk
    of the remaining tokens do too, which tolerates a model eliding a middle clause.
    """
    if not evidence or len(evidence.strip()) < 6:
        return False
    haystack = re.sub(r"\s+", " ", document.text).lower()
    needle = re.sub(r"\s+", " ", evidence).strip().lower()
    if needle in haystack:
        return True

    tokens = re.findall(r"[a-z0-9][a-z0-9.,/%()-]*", needle)
    if len(tokens) < 4:
        return False
    numeric = [token for token in tokens if any(char.isdigit() for char in token)]
    for token in numeric:
        if token.strip(".,/()-") and token not in haystack:
            return False
    present = sum(1 for token in tokens if token in haystack)
    return present / len(tokens) >= 0.85


def _page_of_evidence(document: PolicyDocument, evidence: Optional[str]) -> Optional[int]:
    if not evidence:
        return None
    needle = " ".join(evidence.split()[:8]).lower()
    for page in document.pages:
        if needle and needle in " ".join(page.search_text.split()).lower():
            return page.number
    return None


def _parse_group_response(document: PolicyDocument, specs: List[FieldSpec],
                          payload: Dict[str, Any]) -> Dict[str, LLMFieldResult]:
    results: Dict[str, LLMFieldResult] = {}
    for spec in specs:
        raw = payload.get(_field_key(spec))
        if not isinstance(raw, dict):
            continue

        status_text = str(raw.get("status", "")).strip().lower().replace(" ", "_")
        if status_text not in _STATUS_VALUES:
            continue
        status = FieldStatus(status_text)
        if status is FieldStatus.NOT_FOUND:
            continue

        evidence = raw.get("evidence")
        evidence = evidence.strip() if isinstance(evidence, str) else None
        if not _evidence_present(document, evidence):
            LOGGER.debug("dropping unverifiable LLM value for %s", spec.path)
            continue

        value = raw.get("value")
        unit_text = str(raw.get("unit") or "").strip()
        unit = ValueUnit(unit_text) if unit_text in _UNIT_VALUES else None

        # Normalise integers to float up front. A subtle trap: ``isinstance(0, float)`` is
        # False, so an integer value silently bypassed the evidence-repair branch below and a
        # model that answered ``0`` had its answer published verbatim.
        if isinstance(value, bool):
            value = None
        elif isinstance(value, int):
            value = float(value)

        if unit in (ValueUnit.INR, ValueUnit.PERCENT, ValueUnit.PERCENT_OF_SUM_INSURED,
                    ValueUnit.DAYS, ValueUnit.MONTHS, ValueUnit.COUNT):
            value = _normalise_number(value)
            if not isinstance(value, float):
                unit = ValueUnit.TEXT if isinstance(value, str) else None
        elif isinstance(value, str):
            value = " ".join(value.split()) or None

        notes_extra = None
        if unit in (ValueUnit.PERCENT, ValueUnit.PERCENT_OF_SUM_INSURED):
            corrected = _percent_from_evidence(evidence, value)
            if corrected is not None:
                notes_extra = (f"percentage corrected to {corrected:g}% from the quoted "
                               f"evidence (model returned {value})")
                value = corrected
        elif unit is None and value is not None:
            # No usable unit: re-parse the model's own quote deterministically.
            repaired = _repair_from_evidence(spec, evidence)
            if repaired is not None:
                new_value, new_unit, new_basis = repaired
                if not _values_close(new_value, value):
                    notes_extra = (f"value re-derived as {new_value!r} by parsing the quoted "
                                   f"evidence (model returned {value!r} with no unit)")
                value, unit = new_value, new_unit
                if new_basis:
                    raw["basis"] = new_basis

        basis = raw.get("basis")
        if isinstance(basis, str) and basis.strip().lower() not in _VALID_BASES:
            basis = None

        # Coerce an out-of-vocabulary status onto the field's own vocabulary. A model that
        # says "applied" for "86 employees" has extracted the right fact and mislabelled it;
        # dropping the fact would waste it, and keeping the label would manufacture a
        # disagreement with the rule extractor.
        permitted = allowed_statuses(spec)
        if status not in permitted:
            if value is None:
                LOGGER.debug("dropping %s: status %s not valid for this field and no value",
                             spec.path, status.value)
                continue
            status = _fallback_status(spec)

        notes = raw.get("notes")
        note_text = notes.strip() if isinstance(notes, str) and notes.strip() else None
        if notes_extra:
            note_text = f"{note_text}; {notes_extra}" if note_text else notes_extra
        results[spec.path] = LLMFieldResult(
            status=status,
            value=value,
            unit=unit,
            basis=basis.strip() if isinstance(basis, str) and basis.strip() else None,
            evidence=evidence,
            notes=note_text,
            page=_page_of_evidence(document, evidence),
        )
    return results


def extract_with_llm(document: PolicyDocument, product_type: ProductType,
                     settings: LLMSettings) -> Dict[str, LLMFieldResult]:
    """Run one JSON-mode call per field group. Returns ``{}`` when the LLM is unavailable."""
    if not settings.enabled:
        return {}

    results: Dict[str, LLMFieldResult] = {}
    for group in GROUPS:
        specs = specs_for_prompt(group, product_type)
        if not specs:
            continue
        snippets = retrieve(document, group, limit=settings.snippets_per_group)
        if not snippets:
            continue
        payload = complete_json(settings, SYSTEM_PROMPT, _build_prompt(specs, snippets))
        if not payload:
            continue
        results.update(_parse_group_response(document, specs, payload))
    return results
