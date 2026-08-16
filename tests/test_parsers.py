"""Parser tests.

Every literal here was copied out of the five sample policies. Cases marked "regression" are
defects the first implementation actually had, each of which produced a *plausible wrong
number* rather than a visible failure -- the exact class of bug the accuracy metric punishes.
"""

from datetime import date

import pytest

from gmc_extract.extraction.parsers import (
    StatusMode,
    all_days,
    all_money,
    format_inr,
    parse_basis,
    parse_date,
    parse_money,
    parse_months,
    parse_percent,
    parse_status,
    tenure_between,
)
from gmc_extract.schema import FieldStatus


@pytest.mark.parametrize("text,expected", [
    ("Rs. 200,000", 200_000),
    ("Rs. 300,000", 300_000),
    ("INR 5 Lakhs", 500_000),
    ("Rs. 5 LAKH", 500_000),
    ("\u20b975,000/-", 75_000),
    ("5,00,000", 500_000),          # Indian lakh grouping
    ("51,900,000", 51_900_000),     # international grouping
    ("17,200,000", 17_200_000),
    ("2 Lacs", 200_000),
    ("1 crore", 10_000_000),
    ("Rs. 1,000/- per claim", 1_000),
    ("Cholecystectomy- 35000", 35_000),
    ("Rs.1,04,635/-", 104_635),
])
def test_parse_money(text, expected):
    parsed = parse_money(text)
    assert parsed is not None, text
    assert parsed[0] == expected


@pytest.mark.parametrize("text,expected", [
    ("4500000.00", 4_500_000),                      # was parsed as 450
    ("Rs.9600.00", 9_600),                          # was parsed as 960
    ("Upto Rs. 1200 or 2 PPE kit per day", 1_200),  # was parsed as 120
])
def test_parse_money_ungrouped_regression(text, expected):
    """Regression: the comma-grouped regex branch used to truncate ungrouped numbers."""
    parsed = parse_money(text)
    assert parsed is not None and parsed[0] == expected


@pytest.mark.parametrize("text", ["clause 17 refers", "period 2022 to 2023", "0-35", ""])
def test_parse_money_rejects_non_amounts(text):
    assert parse_money(text) is None


def test_parse_money_require_currency():
    assert parse_money("Total 48 Total 24000000", require_currency=True) is None
    assert parse_money("Rs. 9600", require_currency=True)[0] == 9600


def test_all_money_preserves_order():
    assert [v for v, _ in all_money("Normal 25,000 C-Section 35,000")] == [25_000, 35_000]


@pytest.mark.parametrize("text,value,of_si", [
    ("2 % of Sum Insured per day", 2.0, True),
    ("4 % of Sum Insured per day", 4.0, True),
    ("50% of the Sum Insured", 50.0, True),
    ("1%", 1.0, False),
    ("10% Co-pay shall apply", 10.0, False),
])
def test_parse_percent(text, value, of_si):
    parsed = parse_percent(text)
    assert parsed is not None
    assert parsed[0] == value and parsed[1] is of_si


@pytest.mark.parametrize("text,expected", [
    ("2 % of Sum Insured per day", "per day"),
    ("Room rent/day & ICU/day", "per day"),
    ("INR 1000 per hospitalization", "per hospitalization"),
    ("Rs. 1,000/- per claim", "per claim"),
    ("up to 50k per family", "per family"),
])
def test_parse_basis(text, expected):
    assert parse_basis(text) == expected


def test_pre_and_post_days_are_ordered():
    text = "Pre & Post Hospitalization is covered for 30 days and 60 days respectively."
    assert [d for d, _ in all_days(text)] == [30, 60]


def test_parse_months():
    assert parse_months("9 month waiting period")[0] == 9


@pytest.mark.parametrize("text,expected", [
    ("Pre-existing diseases are covered for existing members", FieldStatus.WAIVED_OFF),
    ("Pre-Existing Disease (PED)   Waived Off", FieldStatus.WAIVED_OFF),
    ("30 Days Wait Period condition is waived off", FieldStatus.WAIVED_OFF),
    ("Initial Waiting Period  Waived Off", FieldStatus.WAIVED_OFF),
    ("Standard waiting period is applicable", FieldStatus.APPLIED),
])
def test_waiting_period_polarity(text, expected):
    """A waived waiting period and a disease "covered from day one" are the same statement."""
    assert parse_status(text, StatusMode.WAITING_PERIOD)[0] is expected


@pytest.mark.parametrize("text,expected", [
    ("Domiciliary Hospitalization is specifically excluded", FieldStatus.NOT_COVERED),
    ("OPD Coverage  Not Covered", FieldStatus.NOT_COVERED),
    ("Corporate Floater  Not Covered", FieldStatus.NOT_COVERED),
    ("holter monitoring are outside the scope of this policy", FieldStatus.NOT_COVERED),
    ("Terrorism cover extended under the policy", FieldStatus.COVERED),
    ("Ambulance charges payable up to a maximum amount of Rs. 1,000", FieldStatus.COVERED),
    ("Listed Day Care Treatment  Covered upto Sum Insured", FieldStatus.COVERED),
    ("NA", FieldStatus.NOT_SPECIFIED),
])
def test_benefit_polarity(text, expected):
    assert parse_status(text, StatusMode.BENEFIT)[0] is expected


def test_negation_beats_the_substring_it_contains():
    """"not covered" contains "covered"; the negative must be tested first."""
    assert parse_status("This benefit is not covered")[0] is FieldStatus.NOT_COVERED


@pytest.mark.parametrize("text,expected", [
    ("02-Apr-2022", date(2022, 4, 2)),
    ("00:00 hrs 17-Mar-2022", date(2022, 3, 17)),
    ("02/06/2022", date(2022, 6, 2)),        # day-first, Indian convention
    ("01-August-2023", date(2023, 8, 1)),
    ("31-July-2024", date(2024, 7, 31)),
    ("26th April 2024", date(2024, 4, 26)),
    ("Midnight 01-Apr-2023", date(2023, 4, 1)),
])
def test_parse_date(text, expected):
    parsed = parse_date(text)
    assert parsed is not None and parsed[0] == expected


def test_tenure_between():
    days, months, display = tenure_between(date(2022, 4, 2), date(2023, 4, 1))
    assert (days, months, display) == (365, 12, "1 year(s)")


@pytest.mark.parametrize("value,expected", [
    (500_000, "Rs. 5,00,000"),
    (51_900_000, "Rs. 5,19,00,000"),
    (1_000, "Rs. 1,000"),
])
def test_format_inr_uses_indian_grouping(value, expected):
    assert format_inr(value) == expected
