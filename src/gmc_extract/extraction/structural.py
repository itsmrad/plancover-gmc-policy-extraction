"""Structural extraction that does not fit the one-label-one-value model.

Three things need bespoke handling:

* **Family structure** -- a single phrase ("Self + Spouse + 2 Dependent children",
  "Employee, Spouse and Kids") that has to be decomposed into the QMS's boolean columns
  plus a child count.
* **Sum insured tiers** -- a *set* of values scattered across a benefit grid and a premium
  rate table, which must be separated from the aggregate (group-level) sum insured and from
  per-life premium figures that live in the same tables.
* **Product name** -- a document title rather than a labelled field.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..ingestion import PolicyDocument
from ..schema import Confidence, ExtractionSource, FamilyStructure
from .parsers import all_money

# --------------------------------------------------------------------------------------
# Family structure
# --------------------------------------------------------------------------------------
_EMPLOYEE_TOKENS = ("employee", "self", "insured person", "primary member",
                    "primary insured", "member")
_SPOUSE_TOKENS = ("spouse", "husband", "wife", "partner")
_CHILD_TOKENS = ("child", "children", "kid", "kids", "son", "daughter", "dependent children")
_PARENT_TOKENS = ("parent", "mother", "father")
_IN_LAW_PATTERNS = (
    "parents in law", "parents-in-law", "parent in law", "parent-in-law", "in-laws",
    "in laws", "parents inlaw",
)
_CHILD_COUNT = re.compile(
    r"(\d+)\s*(?:dependent\s*)?(?:children|child|kids?|dependants?|dependents?)",
    re.IGNORECASE,
)


def parse_family_structure(text: str) -> dict:
    """Decompose a family-structure phrase into QMS boolean columns.

    In-law wording is removed *before* looking for "parent", otherwise
    "Parents-in-law" would also set the ``parents`` flag -- a subtle double count, since
    parents and parents-in-law are separately priced categories in group medical cover.
    """
    lowered = " " + " ".join(text.lower().split()) + " "

    parents_in_law = any(pattern in lowered for pattern in _IN_LAW_PATTERNS)
    without_in_laws = lowered
    for pattern in _IN_LAW_PATTERNS:
        without_in_laws = without_in_laws.replace(pattern, " ")

    count_match = _CHILD_COUNT.search(lowered)
    return {
        "employee": any(token in lowered for token in _EMPLOYEE_TOKENS),
        "spouse": any(token in lowered for token in _SPOUSE_TOKENS),
        "children": any(token in lowered for token in _CHILD_TOKENS),
        "parents": any(token in without_in_laws for token in _PARENT_TOKENS),
        "parents_in_law": parents_in_law,
        "max_children": int(count_match.group(1)) if count_match else None,
    }


def build_family_structure(raw: Optional[str], page: Optional[int],
                           cover_type: Optional[str]) -> FamilyStructure:
    if not raw:
        return FamilyStructure(
            cover_type=cover_type,
            notes="no family-structure field found in the document",
        )
    parsed = parse_family_structure(raw)
    parts = [name.replace("_", " ").title()
             for name in ("employee", "spouse", "children", "parents", "parents_in_law")
             if parsed[name]]
    return FamilyStructure(
        raw_text=raw,
        display=" + ".join(parts) if parts else None,
        employee=parsed["employee"],
        spouse=parsed["spouse"],
        children=parsed["children"],
        parents=parsed["parents"],
        parents_in_law=parsed["parents_in_law"],
        max_children=parsed["max_children"],
        cover_type=cover_type,
        source=ExtractionSource.RULE,
        confidence=Confidence.HIGH if parts else Confidence.LOW,
        page=page,
    )


# --------------------------------------------------------------------------------------
# Sum insured
# --------------------------------------------------------------------------------------
#: Strong tier cues: rate-table headers and explicit per-member sum insured labels. Their
#: whole neighbourhood is trusted.
_STRONG_TIER_CUES = (
    "age band/si", "age band / si", "age band/ si", "age band/si", "graded sum insured",
    "flat sum insured", "inpatient care - sum insured", "inpatient care \u2013 sum insured",
    "capital sum insured", "per employee csi", "sum insured per", "si band",
    "premium per life", "sum insured band",
)
#: Weak cues: the bare phrase appears constantly ("Covered upto Sum Insured",
#: "Restricted to 50% of the Sum Insured"). Only an amount *immediately* after it counts,
#: otherwise the maternity and day-care limits nearby get mistaken for sum insured tiers.
_WEAK_TIER_CUES = ("sum insured", "sum assured", "si")
_WEAK_TIER_SPAN = 40

#: Cues for the group-level aggregate, which must NOT be reported as a tier.
_AGGREGATE_CUES = (
    "total sum insured", "aggregate sum insured", "overall sum insured", "total si",
    "total sum assured",
)
#: Plausible per-member sum insured band. Below this a number is a premium or a day count;
#: above it, it is a group aggregate rather than an individual tier.
_TIER_MIN = 25_000.0
_TIER_MAX = 10_000_000.0
#: Per-member sum insured is always a round figure in the Indian group market (50k, 1L,
#: 1.5L, 2L, 3L, 5L, 7.5L, 10L...). This single constraint removes per-life premium values
#: such as 25,505.46 that sit in the very same rate table.
_TIER_ROUNDING = 5_000.0


def _windows(document: PolicyDocument, cues: Tuple[str, ...], before: int, after: int
             ) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for page in document.pages:
        text = page.search_text
        lowered = text.lower()
        for cue in cues:
            start = 0
            while True:
                index = lowered.find(cue, start)
                if index < 0:
                    break
                begin = max(0, index - before)
                found.append((text[begin:index + after], page.number))
                start = index + 1
    return found


def _is_tier(value: float, aggregate: Optional[float]) -> bool:
    return (_TIER_MIN <= value <= _TIER_MAX
            and value != aggregate
            and value % _TIER_ROUNDING == 0)


def extract_sum_insured(document: PolicyDocument
                        ) -> Tuple[List[float], Optional[str], Optional[float],
                                   Optional[str], Optional[int]]:
    """``(tiers, basis, aggregate, evidence, page)``.

    The aggregate is read as the *first* plausible amount after its label, not the largest in
    the neighbourhood. Taking the largest looked reasonable until the intermediary's phone
    number (``9901679750``), printed two rows below "Total Sum Insured", was reported as a
    17-million-rupee group aggregate.
    """
    aggregate: Optional[float] = None
    evidence: Optional[str] = None
    aggregate_page: Optional[int] = None
    for window, page in _windows(document, _AGGREGATE_CUES, 10, 160):
        for value, _raw in all_money(window):
            if value >= _TIER_MIN:
                aggregate = value
                evidence = " ".join(window.split())[:200]
                aggregate_page = page
                break
        if aggregate is not None:
            break

    tiers: List[float] = []

    def offer(value: float) -> None:
        if _is_tier(value, aggregate) and value not in tiers:
            tiers.append(value)

    for window, _page in _windows(document, _STRONG_TIER_CUES, 60, 300):
        for value, _raw in all_money(window):
            offer(value)

    for page in document.pages:
        text = page.search_text
        lowered = text.lower()
        for cue in _WEAK_TIER_CUES:
            # Word-boundary matching, not substring: an earlier version searched for the
            # bare cue "si", which matched inside "Insured", "Basic" and "consider" and
            # dragged the maternity C-Section limit in as a sum insured tier.
            for match in re.finditer(r"\b%s\b" % re.escape(cue), lowered):
                tail = text[match.end():match.end() + _WEAK_TIER_SPAN]
                for value, _raw in all_money(tail):
                    offer(value)

    tiers.sort()

    basis: Optional[str] = None
    lowered_all = document.text.lower()
    if "graded sum insured" in lowered_all:
        basis = "graded"
    elif "flat sum insured" in lowered_all:
        basis = "flat"
    elif tiers:
        basis = "graded" if len(tiers) > 1 else "flat"

    return tiers, basis, aggregate, evidence, aggregate_page


# --------------------------------------------------------------------------------------
# Product name
# --------------------------------------------------------------------------------------
_TITLE_PATTERN = re.compile(
    r"policy\s+(?:certificate|document|schedule)\s*[-–:]\s*([^\n]{3,60})", re.IGNORECASE
)
_POLICY_LINE = re.compile(r"^[^\n]{6,70}policy[^\n]{0,20}$", re.IGNORECASE | re.MULTILINE)


def extract_product_name(document: PolicyDocument) -> Tuple[Optional[str], Optional[int]]:
    """The product/plan name, taken from the document title.

    The explicit "Policy Document - <name>" pattern is searched across *all* pages: Niva
    Bupa prints it on the schedule page (page 3), behind two pages of covering letter, and
    limiting the search to the first two pages returned a sentence from the letter instead.
    """
    for page in document.pages:
        match = _TITLE_PATTERN.search(page.text)
        if match:
            return _clean_title(match.group(1)), page.number

    for page in document.pages[:2]:
        for line in page.text.split("\n")[:14]:
            stripped = line.strip()
            if not stripped or len(stripped) > 70 or stripped.endswith("."):
                continue
            lowered = stripped.lower()
            if "policy" not in lowered:
                continue
            if lowered.startswith(("policy no", "policy number", "policy period",
                                   "policy issuing", "policy servicing", "policy type",
                                   "policy tenure", "this policy", "the policy")):
                continue
            letters = [c for c in stripped if c.isalpha()]
            # A product title is typographically distinct: predominantly upper case.
            if letters and sum(c.isupper() for c in letters) / len(letters) < 0.5:
                continue
            cleaned = _clean_title(stripped)
            if cleaned and len(cleaned) > 8:
                return cleaned, page.number
    return None, None


def _clean_title(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = re.sub(r"[^\w\s&./()-]+$", "", cleaned).strip(" -–:")
    cleaned = re.sub(r"\s*-\s*policy\s+schedule$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
