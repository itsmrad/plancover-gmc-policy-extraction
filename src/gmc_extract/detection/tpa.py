"""TPA (Third Party Administrator) detection.

Three tiers, tried in order of reliability:

1. **Known-TPA lexicon** -- catches the common case by name (Medi Assist, Paramount, ...).
2. **Label proximity** -- reads whatever follows "Existing TPA" / "Claims Administrator" /
   "Third Party Administrator", so a TPA absent from the lexicon is still found.
3. **Generic shape match** -- ``<Something> [Insurance] TPA [Services] [Pvt] [Ltd]``.

Then a resolution step, which is the part that actually matters on the sample set: if the
claims administrator resolves to the *insurer itself*, the answer is not "no TPA found", it
is ``in_house_insurer_administered``. All three sample insurers administer claims in-house,
so a detector without this tier would report a null for every document and look broken.

**Broker guard.** Both the Niva Bupa and Liberty schedules name "Hii Insurance Broking
Services Private Limited" as agent/intermediary. A broker is not a TPA, and it sits right
next to the fields a naive proximity search would read. Broker-shaped names are therefore
explicitly rejected.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..ingestion import PolicyDocument
from ..ingestion.layout import normalise_for_matching
from ..schema import Confidence, DetectionEvidence, TPADetection, TPAMode
from .insurer import insurer_name_forms
from .registry import IN_HOUSE_MARKERS, KNOWN_TPAS, TPA_LABELS

#: Entities that are emphatically *not* TPAs, however close they sit to a TPA label.
_BROKER_MARKERS = (
    "broking", "broker", "insurance brokers", "intermediary", "agent", "web aggregator",
    "corporate agent",
)
_NULL_VALUES = {"", "na", "n/a", "nil", "none", "not applicable", "-", "--", "not covered"}

_GENERIC_TPA = re.compile(
    r"([A-Z][A-Za-z&.,'\-]*(?:\s+[A-Z][A-Za-z&.,'\-]*){0,4}\s+"
    r"(?:Insurance\s+|Health\s+|Healthcare\s+)?TPA"
    r"(?:\s+Services)?(?:\s+(?:Private|Pvt)\.?)?(?:\s+(?:Limited|Ltd)\.?)?)"
)


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" :;|.-\t")
    return value


def _looks_like_broker(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _BROKER_MARKERS)


def _matches_insurer(name: str, insurer_key: Optional[str]) -> bool:
    lowered = name.lower()
    for form in insurer_name_forms(insurer_key):
        if form and form in lowered:
            return True
    return False


def _label_values(document: PolicyDocument) -> List[Tuple[str, str, Optional[int]]]:
    """Pull ``(label, value, page)`` triples for every TPA-ish label in the document."""
    results: List[Tuple[str, str, Optional[int]]] = []
    for page in document.pages:
        for line in page.search_text.split("\n"):
            lowered = line.lower()
            for label in TPA_LABELS:
                position = lowered.find(label)
                if position < 0:
                    continue
                tail = line[position + len(label):]
                value = _clean_value(tail)
                if value:
                    results.append((label, value, page.number))
                break
    return results


def _label_windows(document: PolicyDocument,
                   window: int = 350) -> List[Tuple[str, str, Optional[int]]]:
    """``(label, following_text, page)`` for each TPA label.

    Needed because label and value are not always on the same line. In the Care Health
    schedule the servicing entity sits in a table *below* the "Claims Servicing Team"
    heading and is itself split across two lines ("Care Health" / "Insurance Ltd"), so only
    a whitespace-collapsed window over the following text can see it.
    """
    results: List[Tuple[str, str, Optional[int]]] = []
    for page in document.pages:
        text = page.search_text
        lowered = text.lower()
        for label in TPA_LABELS:
            start = 0
            while True:
                position = lowered.find(label, start)
                if position < 0:
                    break
                begin = position + len(label)
                results.append((label, _clean_value(text[begin:begin + window]), page.number))
                start = position + 1
    return results


def detect_tpa(document: PolicyDocument, insurer_key: Optional[str] = None) -> TPADetection:
    norm_text = normalise_for_matching(document.text)
    evidence: List[DetectionEvidence] = []

    # --- Tier 1: known TPA by name -------------------------------------------------------
    for canonical, aliases in KNOWN_TPAS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in norm_text:
                index = norm_text.find(alias)
                evidence.append(DetectionEvidence(
                    signal="known_tpa_alias",
                    matched_text=re.sub(r"\s+", " ", norm_text[max(0, index - 40):index + 80]),
                    page=None,
                    weight=3.0,
                ))
                return TPADetection(name=canonical, mode=TPAMode.EXTERNAL,
                                    confidence=Confidence.HIGH, evidence=evidence)

    # --- Tier 2: whatever a TPA/claims-administrator label points at ---------------------
    in_house_hit: Optional[Tuple[str, str, Optional[int]]] = None
    for label, value, page in _label_values(document):
        if value.lower() in _NULL_VALUES:
            evidence.append(DetectionEvidence(signal="label_null_value",
                                              matched_text=f"{label}: {value}",
                                              page=page, weight=1.0))
            continue
        if _looks_like_broker(value):
            # Recorded, not returned: a broker next to a TPA label is a known trap.
            evidence.append(DetectionEvidence(signal="rejected_broker",
                                              matched_text=f"{label}: {value}",
                                              page=page, weight=0.0))
            continue
        if _matches_insurer(value, insurer_key):
            in_house_hit = (label, value, page)
            continue
        if "tpa" in value.lower() or "administrator" in label:
            evidence.append(DetectionEvidence(signal="label_proximity",
                                              matched_text=f"{label}: {value}",
                                              page=page, weight=2.5))
            return TPADetection(name=_clean_value(value), mode=TPAMode.EXTERNAL,
                                confidence=Confidence.MEDIUM, evidence=evidence)

    # --- Tier 2b: label window -- value may sit on following lines, not the same line ----
    for label, window, page in _label_windows(document):
        compact = normalise_for_matching(window)
        for canonical, aliases in KNOWN_TPAS.items():
            for alias in aliases:
                if alias in compact:
                    evidence.append(DetectionEvidence(
                        signal="label_window_known_tpa", matched_text=f"{label} -> {alias}",
                        page=page, weight=3.0))
                    return TPADetection(name=canonical, mode=TPAMode.EXTERNAL,
                                        confidence=Confidence.HIGH, evidence=evidence)
        if in_house_hit is None and _matches_insurer(compact, insurer_key):
            in_house_hit = (label, re.sub(r"\s+", " ", window[:120]), page)

    # --- Tier 3: generic "<Name> TPA Pvt Ltd" shape anywhere in the document -------------
    for match in _GENERIC_TPA.finditer(document.text):
        candidate = _clean_value(match.group(1))
        if _looks_like_broker(candidate) or _matches_insurer(candidate, insurer_key):
            continue
        if len(candidate) < 8:
            continue
        evidence.append(DetectionEvidence(
            signal="generic_tpa_pattern",
            matched_text=re.sub(r"\s+", " ", document.text[max(0, match.start() - 40):
                                                           match.end() + 40]),
            page=document.page_at(match.start()),
            weight=2.0,
        ))
        return TPADetection(name=candidate, mode=TPAMode.EXTERNAL,
                            confidence=Confidence.MEDIUM, evidence=evidence)

    # --- Resolution: in-house servicing --------------------------------------------------
    if in_house_hit:
        label, value, page = in_house_hit
        evidence.append(DetectionEvidence(signal="claims_administrator_is_insurer",
                                          matched_text=f"{label}: {value}",
                                          page=page, weight=3.0))
        return TPADetection(name=None, mode=TPAMode.IN_HOUSE,
                            confidence=Confidence.HIGH, evidence=evidence)

    for marker in IN_HOUSE_MARKERS:
        if marker in norm_text:
            index = norm_text.find(marker)
            evidence.append(DetectionEvidence(
                signal="in_house_marker",
                matched_text=re.sub(r"\s+", " ", norm_text[max(0, index - 40):index + 80]),
                page=None, weight=2.0,
            ))
            return TPADetection(name=None, mode=TPAMode.IN_HOUSE,
                                confidence=Confidence.MEDIUM, evidence=evidence)

    # No external TPA named anywhere and no claims-administrator field: for a group policy
    # this most often means the insurer services claims itself, but we report it as a
    # low-confidence inference rather than asserting it.
    if any(marker in norm_text for marker in ("claims servicing", "claims@", "for claims")):
        evidence.append(DetectionEvidence(
            signal="insurer_claims_contact_only",
            matched_text="document lists insurer claims contact and names no TPA",
            weight=1.0,
        ))
        return TPADetection(name=None, mode=TPAMode.IN_HOUSE,
                            confidence=Confidence.LOW, evidence=evidence)

    return TPADetection(name=None, mode=TPAMode.UNKNOWN, confidence=Confidence.LOW,
                        evidence=evidence)
