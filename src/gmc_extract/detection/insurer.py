"""Insurer identification by weighted multi-signal evidence scoring."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..ingestion import PolicyDocument
from ..ingestion.layout import normalise_for_matching
from ..schema import Confidence, DetectionEvidence, InsurerDetection, ProductType
from .registry import (
    INSURERS,
    PRODUCT_CUES,
    WEIGHT_ALIAS,
    WEIGHT_BRAND,
    WEIGHT_CIN,
    WEIGHT_DOMAIN,
    WEIGHT_IRDAI,
    WEIGHT_LEGAL_NAME,
    InsurerSignature,
)

#: "IRDA Registration Number: 148", "IRDAI Registration No. 145", "IRDA regn no 150".
_IRDAI_PATTERN = re.compile(
    r"irda(?:i)?\s*(?:of\s+india)?\s*(?:regist(?:ration|ered)?|regn\.?|reg\.?)\s*"
    r"(?:number|no\.?|#)?\s*[:\-]?\s*(\d{2,4})",
    re.IGNORECASE,
)
#: Indian Corporate Identity Number.
_CIN_PATTERN = re.compile(r"\b([UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b")

_CONFIDENCE_STRONG = 5.0
_CONFIDENCE_WEAK = 2.5


def _page_of(document: PolicyDocument, needle: str) -> Optional[int]:
    """Locate ``needle`` in the raw document text to attribute evidence to a page.

    Matching is done on the normalised text but page attribution needs raw offsets, so we
    re-find the needle in a lightly-normalised copy of the raw text. Being a few characters
    out is harmless here -- the only consumer is a page label in the evidence trail.
    """
    if not needle:
        return None
    lowered = document.text.lower()
    index = lowered.find(needle.lower())
    if index < 0:
        # The needle may only exist post-normalisation (e.g. collapsed whitespace).
        compact = re.sub(r"\s+", " ", needle.lower())
        index = re.sub(r"\s+", " ", lowered).find(compact)
        if index < 0:
            return None
    return document.page_at(index)


def _context(text: str, index: int, span: int = 90) -> str:
    start = max(0, index - span // 3)
    return re.sub(r"\s+", " ", text[start:index + span]).strip()


def _score_insurer(
    signature: InsurerSignature,
    norm_text: str,
    irdai_numbers: List[str],
    cins: List[str],
    document: PolicyDocument,
) -> Tuple[float, List[DetectionEvidence]]:
    """Sum the weights of every fingerprint that hits.

    Each *signal type* contributes at most once. A footer repeated on six pages should not
    outweigh a genuinely richer match elsewhere -- we are measuring breadth of evidence, not
    frequency.
    """
    score = 0.0
    evidence: List[DetectionEvidence] = []

    def add(signal: str, weight: float, matched: str, index: int) -> None:
        nonlocal score
        score += weight
        evidence.append(
            DetectionEvidence(
                signal=signal,
                matched_text=_context(norm_text, index) if index >= 0 else matched,
                page=_page_of(document, matched),
                weight=weight,
            )
        )

    for name in signature.legal_names:
        idx = norm_text.find(name)
        if idx >= 0:
            add("legal_name", WEIGHT_LEGAL_NAME, name, idx)
            break

    if signature.cin and signature.cin in cins:
        add("cin", WEIGHT_CIN, signature.cin, norm_text.find(signature.cin.lower()))

    if signature.irdai_registration_no and signature.irdai_registration_no in irdai_numbers:
        match = _IRDAI_PATTERN.search(norm_text)
        add("irdai_registration_no", WEIGHT_IRDAI, signature.irdai_registration_no,
            match.start() if match else -1)

    for domain in signature.domains:
        idx = norm_text.find(domain)
        if idx >= 0:
            add("website_domain", WEIGHT_DOMAIN, domain, idx)
            break

    if not any(e.signal == "legal_name" for e in evidence):
        for alias in signature.aliases:
            idx = norm_text.find(alias)
            if idx >= 0:
                add("alias", WEIGHT_ALIAS, alias, idx)
                break

    for token in signature.brand_tokens:
        idx = norm_text.find(token)
        if idx >= 0:
            add("brand_token", WEIGHT_BRAND, token, idx)
            break

    return score, evidence


def detect_insurer(document: PolicyDocument) -> InsurerDetection:
    norm_text = normalise_for_matching(document.text)
    irdai_numbers = [m.group(1) for m in _IRDAI_PATTERN.finditer(norm_text)]
    cins = [m.group(1).upper() for m in _CIN_PATTERN.finditer(document.text)]

    scored: List[Tuple[float, InsurerSignature, List[DetectionEvidence]]] = []
    for signature in INSURERS:
        score, evidence = _score_insurer(signature, norm_text, irdai_numbers, cins, document)
        if score > 0:
            scored.append((score, signature, evidence))

    if not scored:
        return InsurerDetection(confidence=Confidence.LOW)

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best, best_evidence = scored[0]
    runner_up = None
    if len(scored) > 1:
        runner_up = f"{scored[1][1].display_name} (score {round(scored[1][0], 2)})"

    if best_score >= _CONFIDENCE_STRONG:
        confidence = Confidence.HIGH
    elif best_score >= _CONFIDENCE_WEAK:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    return InsurerDetection(
        name=best.display_name,
        canonical_key=best.key,
        irdai_registration_no=best.irdai_registration_no
        if best.irdai_registration_no in irdai_numbers else None,
        cin=best.cin if best.cin and best.cin in cins else None,
        confidence=confidence,
        score=round(best_score, 2),
        evidence=best_evidence,
        runner_up=runner_up,
    )


def detect_product_type(document: PolicyDocument) -> Tuple[ProductType, Dict[str, float]]:
    """Classify the product so inapplicable benefit groups are not hallucinated.

    Two of the five sample documents are Group *Personal Accident* schedules, not medical
    cover. Reporting maternity limits for an accident policy would be a fabrication, so the
    pipeline marks those groups ``not_applicable`` instead.
    """
    norm_text = normalise_for_matching(document.text)
    scores: Dict[str, float] = {}
    for product, cues in PRODUCT_CUES.items():
        total = 0.0
        for cue, weight in cues.items():
            count = norm_text.count(cue)
            if count:
                # Damped repetition: more mentions is more evidence, but sub-linearly.
                total += weight * min(1 + 0.15 * (count - 1), 2.0)
        scores[product.value] = round(total, 2)

    best_key = max(scores, key=lambda k: scores[k])
    if scores[best_key] < 3.0:
        return ProductType.UNKNOWN, scores
    return ProductType(best_key), scores


def insurer_name_forms(key: Optional[str]) -> List[str]:
    """All name forms for an insurer -- used by TPA detection to spot in-house servicing."""
    for signature in INSURERS:
        if signature.key == key:
            return [signature.display_name.lower(), *signature.legal_names,
                    *signature.aliases, *signature.brand_tokens]
    return []
