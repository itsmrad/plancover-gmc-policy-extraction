"""Value normalisers: the layer that turns Indian insurance prose into QMS numbers.

Every literal handled here was taken from the sample documents. The point of this module is
that a QMS needs ``500000``, while a human auditor needs to see ``"Rs. 5 LAKH"`` -- so each
parser returns a normalised value *and* the caller keeps the raw span.

The subtle part is **status polarity**. "Pre-existing diseases are covered" and
"Pre-Existing Disease (PED): Waived Off" mean the same thing (no PED waiting period), but
the words are opposites. Conversely "Domiciliary Hospitalization is specifically excluded"
and "Corporate Floater: Not Covered" are both exclusions expressed differently. Naive
keyword presence gets these backwards, so status parsing is polarity-aware and
context-mode-aware (see :class:`StatusMode`).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Tuple

from ..schema import FieldStatus, ValueUnit


class StatusMode(str, Enum):
    """How to interpret coverage words for a given field.

    ``BENEFIT``
        "covered" -> COVERED, "not covered"/"excluded" -> NOT_COVERED.
    ``WAITING_PERIOD``
        A waiting period that is *waived* is good news; a disease being "covered" from day
        one is the same statement. So "waived"/"covered"/"nil" -> WAIVED_OFF, while
        "applicable"/"applies"/"not covered" -> APPLIED.
    """

    BENEFIT = "benefit"
    WAITING_PERIOD = "waiting_period"


# --------------------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------------------
_MULTIPLIERS = {
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "lakh(s)": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "mn": 1_000_000,
}

_CURRENCY = r"(?:rs\.?|inr|₹|rupees)"
#: Handles both Indian ("5,00,000") and international ("51,900,000") comma grouping, plus
#: bare decimals ("4500000.00") and word multipliers ("5 Lakhs").
#:
#: The comma-grouped branch *requires* a comma. Without that requirement the branch would
#: match only the first three digits of an ungrouped number, silently turning "4500000.00"
#: into 450 and "Rs.9600.00" into 960 -- a bug worth calling out, because it produces a
#: plausible-looking wrong number rather than an obvious failure.
_MONEY = re.compile(
    r"(?:(?P<cur>%s)\s*)?"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<mult>lakhs?|lacs?|crores?|cr\b|thousand|million|mn\b|k\b)?"
    % _CURRENCY,
    re.IGNORECASE,
)
_WORDS_IN_LAKHS = re.compile(r"\b(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)\b", re.IGNORECASE)


def parse_money(text: str, *, require_currency: bool = False) -> Optional[Tuple[float, str]]:
    """First monetary amount in ``text``, normalised to whole rupees.

    Returns ``(amount, raw_span)``. ``require_currency=True`` insists on a currency marker
    or a lakh/crore multiplier, which suppresses false positives from clause numbers,
    percentages and phone numbers.
    """
    for match in _MONEY.finditer(text):
        raw = match.group(0).strip()
        number = match.group("num").replace(",", "")
        multiplier_word = (match.group("mult") or "").lower().strip()
        has_currency = bool(match.group("cur"))

        if require_currency and not has_currency and not multiplier_word:
            continue
        try:
            value = float(number)
        except ValueError:
            continue
        if multiplier_word:
            for key, factor in _MULTIPLIERS.items():
                if multiplier_word.startswith(key.rstrip("s")):
                    value *= factor
                    break
        # A bare small integer with no currency marker is almost always a list index or a
        # day count, not an amount.
        if not has_currency and not multiplier_word:
            if value < 1000:
                continue
            # A bare 4-digit number in the 1900-2100 range is a year far more often than a
            # limit (policy documents are dense with dates).
            if 1900 <= value <= 2100 and float(value).is_integer():
                continue
        return value, raw
    return None


def all_money(text: str, *, require_currency: bool = False) -> List[Tuple[float, str]]:
    """Every monetary amount in ``text``, in order of appearance."""
    found: List[Tuple[float, str]] = []
    for match in _MONEY.finditer(text):
        segment = match.group(0)
        parsed = parse_money(segment, require_currency=require_currency)
        if parsed:
            found.append((parsed[0], segment.strip()))
    return found


# --------------------------------------------------------------------------------------
# Percentages
# --------------------------------------------------------------------------------------
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_OF_SUM_INSURED = re.compile(
    r"(?:of\s+(?:the\s+)?)?(?:sum\s*insured|si\b|aggregate\s+sum\s+insured)", re.IGNORECASE
)
_BASIS = [
    (re.compile(r"per\s*day|/\s*day\b", re.IGNORECASE), "per day"),
    (re.compile(r"per\s*hospitali[sz]ation", re.IGNORECASE), "per hospitalization"),
    (re.compile(r"per\s*claim", re.IGNORECASE), "per claim"),
    (re.compile(r"per\s*family", re.IGNORECASE), "per family"),
    (re.compile(r"per\s*(?:insured\s+)?(?:person|member|life)", re.IGNORECASE), "per person"),
    (re.compile(r"per\s*policy\s*(?:period|year)", re.IGNORECASE), "per policy period"),
    (re.compile(r"per\s*eye", re.IGNORECASE), "per eye"),
    (re.compile(r"per\s*week", re.IGNORECASE), "per week"),
]


def parse_percent(text: str) -> Optional[Tuple[float, bool, str]]:
    """First percentage in ``text`` -> ``(value, is_of_sum_insured, raw_span)``."""
    match = _PERCENT.search(text)
    if not match:
        return None
    tail = text[match.end():match.end() + 60]
    of_si = bool(_OF_SUM_INSURED.search(tail))
    return float(match.group(1)), of_si, match.group(0).strip()


def parse_basis(text: str) -> Optional[str]:
    """The qualifier attached to a limit: per day / per claim / per family / ..."""
    for pattern, label in _BASIS:
        if pattern.search(text):
            return label
    return None


# --------------------------------------------------------------------------------------
# Durations
# --------------------------------------------------------------------------------------
_DAYS = re.compile(r"(\d{1,4})\s*(?:calendar\s*)?days?\b", re.IGNORECASE)
_MONTHS = re.compile(r"(\d{1,3})\s*months?\b", re.IGNORECASE)
_YEARS = re.compile(r"(\d{1,2})\s*(?:year|yr)s?\b", re.IGNORECASE)
_ORDINAL_YEARS = re.compile(
    r"\b(?:1st|first)\s*(?:(?:and|&|/|,)\s*(?:2nd|second)\s*)?year\b", re.IGNORECASE
)


def parse_days(text: str) -> Optional[Tuple[int, str]]:
    match = _DAYS.search(text)
    if match:
        return int(match.group(1)), match.group(0).strip()
    return None


def all_days(text: str) -> List[Tuple[int, str]]:
    return [(int(m.group(1)), m.group(0).strip()) for m in _DAYS.finditer(text)]


def parse_months(text: str) -> Optional[Tuple[int, str]]:
    match = _MONTHS.search(text)
    if match:
        return int(match.group(1)), match.group(0).strip()
    return None


def parse_years(text: str) -> Optional[Tuple[int, str]]:
    match = _YEARS.search(text)
    if match:
        return int(match.group(1)), match.group(0).strip()
    return None


# --------------------------------------------------------------------------------------
# Status / polarity
# --------------------------------------------------------------------------------------
#: Ordered longest-first within each bucket so "not covered" is tested before "covered".
_WAIVED_PHRASES = (
    "waived off", "waived-off", "waived for", "is waived", "are waived", "waived",
    "waiver of", "waiver applicable", "waiver", "condition is waived",
)
_NOT_COVERED_PHRASES = (
    "not covered", "not payable", "not applicable", "not available", "specifically excluded",
    "stands excluded", "is excluded", "are excluded", "excluded from the scope",
    "outside the scope", "excluded", "no cover", "nil cover", "not included",
)
_COVERED_PHRASES = (
    "covered upto", "covered up to", "is covered", "are covered", "shall be covered",
    "covered under", "we will cover", "we shall reimburse", "shall reimburse", "is payable",
    "are payable", "payable up to", "payable upto", "is available", "available", "reimbursed",
    "included", "provided", "extended under the policy", "covered", "applicable up to",
)
_APPLIED_PHRASES = (
    "shall apply", "will apply", "is applicable", "are applicable", "applicable", "applies",
    "as per policy terms", "standard waiting period", "applied",
)
_NULLISH = ("na", "n/a", "nil", "not specified", "none", "-", "--")


def _first_phrase(text: str, phrases: Tuple[str, ...]) -> Optional[str]:
    lowered = text.lower()
    best: Optional[Tuple[int, str]] = None
    for phrase in phrases:
        index = lowered.find(phrase)
        if index >= 0 and (best is None or index < best[0]):
            best = (index, phrase)
    return best[1] if best else None


def parse_status(text: str, mode: StatusMode = StatusMode.BENEFIT
                 ) -> Tuple[FieldStatus, Optional[str]]:
    """Resolve a coverage verdict from a text window.

    Returns ``(status, matched_phrase)``. Negative phrases are tested before positive ones
    because they contain them as substrings ("not covered" contains "covered").
    """
    stripped = text.strip().lower()
    if stripped in _NULLISH:
        return FieldStatus.NOT_SPECIFIED, stripped

    waived = _first_phrase(text, _WAIVED_PHRASES)
    negative = _first_phrase(text, _NOT_COVERED_PHRASES)
    positive = _first_phrase(text, _COVERED_PHRASES)
    applied = _first_phrase(text, _APPLIED_PHRASES)

    if mode is StatusMode.WAITING_PERIOD:
        # A waived waiting period is the headline answer whenever it is stated.
        if waived:
            return FieldStatus.WAIVED_OFF, waived
        # "Pre-existing diseases are covered" == the PED waiting period does not apply.
        if positive and not negative:
            return FieldStatus.WAIVED_OFF, positive
        if applied or negative:
            return FieldStatus.APPLIED, applied or negative
        return FieldStatus.NOT_FOUND, None

    if waived:
        return FieldStatus.WAIVED_OFF, waived
    if negative:
        # A later "covered" does not rescue an explicit exclusion for the same subject.
        return FieldStatus.NOT_COVERED, negative
    if positive:
        return FieldStatus.COVERED, positive
    if applied:
        return FieldStatus.APPLIED, applied
    return FieldStatus.NOT_FOUND, None


# --------------------------------------------------------------------------------------
# Counts and dates
# --------------------------------------------------------------------------------------
_COUNT = re.compile(r"\b(\d{1,6})\b")


def parse_count(text: str) -> Optional[Tuple[int, str]]:
    match = _COUNT.search(text)
    if match:
        return int(match.group(1)), match.group(0)
    return None


_MONTH_NAMES = (
    "january february march april may june july august september october november december"
).split()
_MONTH_LOOKUP = {name[:3]: index + 1 for index, name in enumerate(_MONTH_NAMES)}
_MONTH_LOOKUP.update({name: index + 1 for index, name in enumerate(_MONTH_NAMES)})

_DATE_PATTERNS = (
    # 02-Apr-2022 / 01-August-2023 / 26 April 2024 / 26th April 2024
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s\-/]([A-Za-z]{3,9})[\s\-/](\d{4})\b"),
    # 02/06/2022 / 17-03-2022  (day first: Indian convention)
    re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"),
    # 2022-06-02
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),
)


def parse_date(text: str) -> Optional[Tuple[date, str]]:
    """First date in ``text``, assuming day-first ordering (Indian convention)."""
    for index, pattern in enumerate(_DATE_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        try:
            if index == 0:
                day, month_name, year = match.groups()
                month = _MONTH_LOOKUP.get(month_name.lower()[:3])
                if not month:
                    continue
                return date(int(year), month, int(day)), match.group(0)
            if index == 1:
                day, month, year = (int(g) for g in match.groups())
                if month > 12:  # tolerate an accidental month-first document
                    day, month = month, day
                return date(year, month, day), match.group(0)
            year, month, day = (int(g) for g in match.groups())
            return date(year, month, day), match.group(0)
        except ValueError:
            continue
    return None


def all_dates(text: str) -> List[Tuple[date, str]]:
    found: List[Tuple[date, str]] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = parse_date(match.group(0))
            if parsed:
                found.append(parsed)
    # Preserve document order, drop duplicates.
    seen = set()
    ordered: List[Tuple[date, str]] = []
    for value, raw in sorted(found, key=lambda item: text.find(item[1])):
        if value not in seen:
            seen.add(value)
            ordered.append((value, raw))
    return ordered


def tenure_between(start: Optional[date], end: Optional[date]
                   ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """``(days, months, display)`` for a policy period, e.g. ``(365, 12, "12 months")``."""
    if not start or not end:
        return None, None, None
    days = (end - start).days + 1
    months = round(days / 30.44)
    display = f"{months} months" if months % 12 else f"{months // 12} year(s)"
    return days, months, display


# --------------------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------------------
def format_inr(value: float) -> str:
    """Render an amount with Indian digit grouping, e.g. 500000 -> "Rs. 5,00,000"."""
    whole = int(round(value))
    text = str(abs(whole))
    if len(text) > 3:
        head, tail = text[:-3], text[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        text = f"{head},{tail}"
    return f"Rs. {'-' if whole < 0 else ''}{text}"


def display_for(value, unit: Optional[ValueUnit], basis: Optional[str] = None) -> str:
    """Human-readable rendering of a normalised value."""
    if unit is ValueUnit.INR and isinstance(value, (int, float)):
        rendered = format_inr(value)
    elif unit is ValueUnit.PERCENT_OF_SUM_INSURED:
        rendered = f"{_trim(value)}% of sum insured"
    elif unit is ValueUnit.PERCENT:
        rendered = f"{_trim(value)}%"
    elif unit is ValueUnit.DAYS:
        rendered = f"{_trim(value)} days"
    elif unit is ValueUnit.MONTHS:
        rendered = f"{_trim(value)} months"
    elif unit is ValueUnit.YEARS:
        rendered = f"{_trim(value)} year(s)"
    elif unit is ValueUnit.COUNT:
        rendered = str(_trim(value))
    else:
        rendered = str(value)
    return f"{rendered} {basis}" if basis else rendered


def _trim(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_iso_or_none(text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
