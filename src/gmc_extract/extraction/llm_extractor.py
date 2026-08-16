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
from typing import Any, Dict, List, Optional

from ..config import LLMSettings
from ..ingestion import PolicyDocument
from ..llm import complete_json
from ..schema import FieldStatus, ProductType, ValueUnit
from .field_specs import GROUPS, FieldSpec, ValueKind
from .retrieval import retrieve, specs_for_prompt

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
6. A waiting period that is waived is status "waived_off". A waiting period that applies is \
status "applied". A benefit that is excluded is status "not_covered".
7. Distinguish a benefit LIMIT from a CO-PAYMENT. A co-pay percentage is not a benefit limit; \
mention it in "notes" instead.
8. Reply with a single JSON object and nothing else."""

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


def _build_prompt(specs: List[FieldSpec], snippets) -> str:
    lines = ["DOCUMENT TEXT (each block tagged with its page):", ""]
    for snippet in snippets:
        lines.append(f"--- page {snippet.page} ---")
        lines.append(snippet.text)
        lines.append("")
    lines.append("FIELDS TO EXTRACT (JSON keys):")
    for spec in specs:
        hint = _KIND_HINTS.get(spec.kind, "")
        lines.append(f'- "{_field_key(spec)}": {spec.label}. Expect {hint}.')
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

    Whitespace-insensitive, because a model will normalise the spacing of a quote it copied
    correctly. Values whose evidence cannot be located are dropped as likely fabrications.
    """
    if not evidence or len(evidence.strip()) < 6:
        return False
    haystack = re.sub(r"\s+", " ", document.text).lower()
    needle = re.sub(r"\s+", " ", evidence).strip().lower()
    if needle in haystack:
        return True
    # Fall back to a distinctive fragment: models sometimes elide a middle clause.
    words = needle.split()
    if len(words) >= 6:
        return " ".join(words[:6]) in haystack or " ".join(words[-6:]) in haystack
    return False


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

        if unit in (ValueUnit.INR, ValueUnit.PERCENT, ValueUnit.PERCENT_OF_SUM_INSURED,
                    ValueUnit.DAYS, ValueUnit.MONTHS, ValueUnit.COUNT):
            value = _normalise_number(value)
            if not isinstance(value, float):
                unit = ValueUnit.TEXT if isinstance(value, str) else None
        elif isinstance(value, str):
            value = " ".join(value.split()) or None

        basis = raw.get("basis")
        notes = raw.get("notes")
        results[spec.path] = LLMFieldResult(
            status=status,
            value=value,
            unit=unit,
            basis=basis.strip() if isinstance(basis, str) and basis.strip() else None,
            evidence=evidence,
            notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
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
