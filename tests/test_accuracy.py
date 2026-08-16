"""Ground-truth accuracy check.

The expected values below were read **by hand** out of the five sample PDFs. They live here,
in the test suite, and are never consulted by the pipeline -- the brief forbids hardcoding
expected answers into the *output*, which is a different thing from asserting them in tests.

This file is the evidence behind the accuracy claim in the README. Running it prints a
per-document score, so the number in the README can be regenerated rather than trusted.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Tuple

import pytest

from gmc_extract.config import LLMSettings
from gmc_extract.pipeline import PipelineOptions, process_document
from gmc_extract.schema import QMSField

CARE_1 = "1.Policy Copy.pdf"
CARE_2 = "GHI Policy.pdf"
NIVA = "olj4KTUo9B1546-1692687606_925469 - 00 GMC Renewal Policy 00.pdf"
LIBERTY_1 = "Net Catalyst - GPA - Policy Copy - 2022-23.pdf"
LIBERTY_2 = "Policy liberty 2022-2023.pdf"


def V(value: Any) -> Tuple[str, Any]:
    """Expect this normalised field value."""
    return ("value", value)


def S(status: str) -> Tuple[str, Any]:
    """Expect this coverage status."""
    return ("status", status)


def A(value: Any) -> Tuple[str, Any]:
    """Expect this plain (non-QMSField) attribute."""
    return ("attr", value)


#: path -> expectation, per document. Verified against the PDFs by reading them.
GROUND_TRUTH: Dict[str, Dict[str, Tuple[str, Any]]] = {
    CARE_1: {
        "insurer.name": A("Care Health Insurance Ltd."),
        "insurer.irdai_registration_no": A("148"),
        "policy.product_type": A("group_medical_cover"),
        "policy.product_name": V("Group Care 360"),
        "policy.policy_number": V("41201895"),
        "policy.policyholder_name": V("AAYUV TECHNOLOGIES PRIVATE LIMITED"),
        "policy.previous_year_policy_period.inception_date": A(date(2022, 4, 2)),
        "policy.previous_year_policy_period.expiry_date": A(date(2023, 4, 1)),
        "policy.previous_year_policy_period.tenure_months": A(12),
        "policy.previous_year_premium.net_premium": V(349_469),
        "policy.previous_year_premium.gross_premium": V(412_373),
        "structure.sum_insured_tiers": A([300_000.0, 500_000.0]),
        "structure.sum_insured_basis": A("graded"),
        "structure.aggregate_sum_insured": V(51_900_000),
        "structure.family_structure.max_children": A(2),
        "structure.family_structure.spouse": A(True),
        "demographics.employees": V(115),
        "demographics.dependents_total": V(32),
        "demographics.total_lives": V(147),
        "benefits.room_and_hospitalisation.room_rent": V(2.0),
        "benefits.room_and_hospitalisation.icu_charges": V(4.0),
        "benefits.room_and_hospitalisation.pre_hospitalization": V(30),
        "benefits.room_and_hospitalisation.post_hospitalization": V(60),
        "benefits.maternity.nine_month_waiting_period": S("waived_off"),
        "benefits.maternity.normal_delivery_metro": V(50_000),
        "benefits.maternity.c_section_metro": V(50_000),
        "benefits.maternity.pre_post_natal_expenses": S("not_covered"),
        "benefits.maternity.baby_day_one_cover": S("covered"),
        "benefits.waiting_periods.thirty_day_waiting_period": S("waived_off"),
        "benefits.waiting_periods.first_and_second_year_waiting_period": S("waived_off"),
        "benefits.waiting_periods.pre_existing_diseases": S("waived_off"),
        "benefits.other_benefits.domiciliary_hospitalization": S("not_covered"),
        "benefits.other_benefits.modern_treatment": V(50.0),
        "benefits.other_benefits.psychiatric_treatment": S("covered"),
        "benefits.other_benefits.bariatric_treatment": S("covered"),
        "benefits.other_benefits.teleconsultation": S("covered"),
        "benefits.infertility_and_ambulance.infertility_treatment": S("not_covered"),
        "benefits.infertility_and_ambulance.ambulance_charges": V(2_000),
    },
    CARE_2: {
        "insurer.name": A("Care Health Insurance Ltd."),
        "policy.product_type": A("group_medical_cover"),
        "policy.policy_number": V("39899400"),
        "policy.policyholder_name": V("MUKUNDA FOODS PVT LTD"),
        "policy.previous_year_policy_period.inception_date": A(date(2022, 3, 17)),
        "policy.previous_year_policy_period.expiry_date": A(date(2023, 3, 16)),
        "policy.previous_year_premium.net_premium": V(461_699),
        "policy.previous_year_premium.gross_premium": V(544_804),
        "structure.sum_insured_tiers": A([200_000.0]),
        "structure.sum_insured_basis": A("flat"),
        "structure.aggregate_sum_insured": V(17_200_000),
        "structure.family_structure.max_children": A(4),
        "demographics.employees": V(86),
        "demographics.dependents_total": V(66),
        "demographics.total_lives": V(152),
        # This policy imposes no room-rent cap at all; "No Limit" is the correct answer and
        # a blank cell would be a miss.
        "benefits.room_and_hospitalisation.room_rent": V("no limit"),
        "benefits.room_and_hospitalisation.icu_charges": V("no limit"),
        "benefits.room_and_hospitalisation.pre_hospitalization": V(30),
        "benefits.room_and_hospitalisation.post_hospitalization": V(60),
        "benefits.maternity.nine_month_waiting_period": S("waived_off"),
        "benefits.maternity.normal_delivery_metro": V(75_000),
        "benefits.maternity.c_section_metro": V(75_000),
        "benefits.maternity.pre_post_natal_expenses": S("not_covered"),
        "benefits.waiting_periods.thirty_day_waiting_period": S("waived_off"),
        "benefits.waiting_periods.first_and_second_year_waiting_period": S("waived_off"),
        "benefits.waiting_periods.pre_existing_diseases": S("waived_off"),
        "benefits.other_benefits.domiciliary_hospitalization": S("not_covered"),
        "benefits.infertility_and_ambulance.infertility_treatment": S("not_covered"),
        "benefits.infertility_and_ambulance.ambulance_charges": V(1_000),
        "benefits.buffer_and_waivers.corporate_buffer": S("covered"),
        "benefits.buffer_and_waivers.corporate_buffer_limit": V(500_000),
        "benefits.buffer_and_waivers.co_payment": V(10.0),
    },
    NIVA: {
        "insurer.name": A("Niva Bupa Health Insurance Company Ltd."),
        "insurer.irdai_registration_no": A("145"),
        "policy.product_type": A("group_medical_cover"),
        "policy.product_name": V("Health Plus"),
        "policy.policy_number": V("00600900202301"),
        "policy.policyholder_name": V("Myndloop Tech Private Limited"),
        "policy.previous_year_policy_period.inception_date": A(date(2023, 8, 1)),
        "policy.previous_year_policy_period.expiry_date": A(date(2024, 7, 31)),
        "policy.previous_year_policy_period.first_policy_inception_date": A(date(2023, 8, 1)),
        "policy.previous_year_premium.net_premium": V(83_454),
        # The schedule states 83,454 net + 15,022 IGST = 98,476 gross. Page 6 is a separate
        # *premium receipt* for Rs. 1,04,635. I originally recorded the receipt figure here;
        # the LLM extractor disagreed with the rule extractor, and reviewing the conflict
        # showed my ground truth was the wrong one. Left documented because catching my
        # error is exactly what the second extractor is for.
        "policy.previous_year_premium.gross_premium": V(98_476),
        "structure.sum_insured_tiers": A([500_000.0]),
        "structure.aggregate_sum_insured": V(4_500_000),
        "structure.family_structure.employee": A(True),
        "structure.family_structure.spouse": A(True),
        "structure.family_structure.children": A(True),
        "structure.family_structure.cover_type": A("Floater"),
        "demographics.employees": V(9),
        "demographics.total_lives": V(20),
        "benefits.room_and_hospitalisation.room_rent": V(1.0),
        "benefits.room_and_hospitalisation.icu_charges": V(2.0),
        "benefits.room_and_hospitalisation.pre_hospitalization": V(30),
        "benefits.room_and_hospitalisation.post_hospitalization": V(60),
        "benefits.maternity.normal_delivery_metro": V(25_000),
        "benefits.maternity.c_section_metro": V(35_000),
        "benefits.maternity.pre_post_natal_expenses": V(5_000),
        "benefits.maternity.baby_day_one_cover": S("covered"),
        "benefits.waiting_periods.thirty_day_waiting_period": S("waived_off"),
        "benefits.waiting_periods.first_and_second_year_waiting_period": S("waived_off"),
        "benefits.waiting_periods.pre_existing_diseases": S("waived_off"),
        "benefits.other_benefits.opd_benefit": S("not_covered"),
        "benefits.other_benefits.day_care_expenses": S("covered"),
        "benefits.other_benefits.psychiatric_treatment": S("covered"),
        "benefits.infertility_and_ambulance.ambulance_charges": V(1_000),
        "benefits.buffer_and_waivers.corporate_buffer": S("not_covered"),
        "benefits.buffer_and_waivers.corporate_buffer_limit": S("not_specified"),
        "benefits.buffer_and_waivers.co_payment": S("not_specified"),
        "benefits.buffer_and_waivers.disease_wise_capping": S("present"),
    },
    LIBERTY_1: {
        "insurer.name": A("Liberty General Insurance Ltd."),
        "insurer.irdai_registration_no": A("150"),
        "policy.product_type": A("group_personal_accident"),
        "policy.policy_number": V("4112-200101-22-7001375-00-000"),
        "policy.policyholder_name": V("Net Catalysts"),
        "policy.previous_year_policy_period.inception_date": A(date(2022, 6, 2)),
        "policy.previous_year_policy_period.expiry_date": A(date(2023, 6, 1)),
        "policy.previous_year_premium.net_premium": V(9_600),
        "policy.previous_year_premium.gross_premium": V(11_328),
        "structure.aggregate_sum_insured": V(24_000_000),
        "demographics.total_lives": V(48),
        # A medical benefit on an accident policy must be explicitly inapplicable, never
        # invented and never silently blank.
        "benefits.maternity.normal_delivery_metro": S("not_applicable"),
        "benefits.room_and_hospitalisation.room_rent": S("not_applicable"),
        "benefits.waiting_periods.pre_existing_diseases": S("not_applicable"),
    },
}
GROUND_TRUTH[LIBERTY_2] = dict(GROUND_TRUTH[LIBERTY_1])


def _resolve(record, path: str):
    target = record
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _check(record, path: str, expectation: Tuple[str, Any]) -> Tuple[bool, Any]:
    kind, expected = expectation
    target = _resolve(record, path)

    if kind == "attr":
        actual = target
        if isinstance(actual, QMSField):
            actual = actual.value
        return actual == expected, actual

    assert isinstance(target, QMSField), f"{path} is not a QMSField"
    if kind == "status":
        actual = target.status if isinstance(target.status, str) else target.status.value
        return actual == expected, actual

    actual = target.value
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 0.01, actual
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower(), actual
    return actual == expected, actual


@pytest.fixture(scope="module")
def records(documents):
    """Records under test.

    Rule-only by default, so the suite needs no API key and no network. Set
    ``GMC_TEST_HYBRID=1`` (with a provider configured in ``.env``) to run the same
    ground-truth table against the hybrid pipeline and prove the LLM layer does not regress
    any verified field.
    """
    if os.getenv("GMC_TEST_HYBRID") == "1":
        settings = LLMSettings.from_env()
        if not settings.enabled:
            pytest.skip("GMC_TEST_HYBRID=1 but no LLM provider is configured")
    else:
        settings = LLMSettings(provider="none")
    options = PipelineOptions(llm=settings)
    return {name: process_document(document, options)
            for name, document in documents.items()}


@pytest.mark.parametrize("file_name", list(GROUND_TRUTH))
def test_document_matches_ground_truth(file_name, records):
    record = records[file_name]
    failures: List[str] = []
    for path, expectation in GROUND_TRUTH[file_name].items():
        ok, actual = _check(record, path, expectation)
        if not ok:
            failures.append(f"{path}: expected {expectation[0]}={expectation[1]!r}, "
                            f"got {actual!r}")
    assert not failures, f"{file_name}\n  " + "\n  ".join(failures)


def test_overall_accuracy_and_report(records, capsys):
    """Print the score used in the README, and hold the line at 100%."""
    total = 0
    correct = 0
    lines: List[str] = []
    for file_name, expectations in GROUND_TRUTH.items():
        hits = 0
        for path, expectation in expectations.items():
            ok, _actual = _check(records[file_name], path, expectation)
            hits += int(ok)
        total += len(expectations)
        correct += hits
        lines.append(f"  {file_name[:52]:54} {hits:3d}/{len(expectations):<3d} "
                     f"{100.0 * hits / len(expectations):5.1f}%")

    mode = "hybrid" if os.getenv("GMC_TEST_HYBRID") == "1" else "rule_only"
    report = (f"\nGround-truth accuracy [{mode}] ({len(GROUND_TRUTH)} documents, "
              f"{total} verified fields)\n" + "\n".join(lines)
              + f"\n  {'TOTAL':54} {correct:3d}/{total:<3d} "
                f"{100.0 * correct / total:5.1f}%\n")
    with capsys.disabled():
        print(report)

    assert correct == total, f"{total - correct} ground-truth mismatches"


def test_every_record_has_the_identical_key_set(records):
    """The QMS contract: same keys for every document, whatever the insurer or product."""
    key_sets = {name: set(_flatten(record.model_dump(mode="json")))
                for name, record in records.items()}
    reference = next(iter(key_sets.values()))
    for name, keys in key_sets.items():
        assert keys == reference, f"{name} has a different key set"


def _flatten(payload, prefix: str = ""):
    keys = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            keys.append(path)
            keys.extend(_flatten(value, path))
    return keys
