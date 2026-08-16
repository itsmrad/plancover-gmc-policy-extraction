"""Hybrid-path tests using a stub provider.

No API key is required. A stub stands in for the HTTP call so the LLM parsing, the
evidence-verification guard and the whole reconciliation matrix are exercised in CI, and so
the hybrid code path is not shipped untested just because a key was unavailable at build
time.
"""

from __future__ import annotations

import pytest  # noqa: F401

from gmc_extract.config import LLMSettings
from gmc_extract.extraction import llm_extractor
from gmc_extract.extraction.field_specs import SPECS_BY_PATH
from gmc_extract.extraction.llm_extractor import LLMFieldResult, extract_with_llm
from gmc_extract.extraction.merge import merge_field
from gmc_extract.extraction.rule_extractor import Candidate
from gmc_extract.pipeline import PipelineOptions, process_document
from gmc_extract.schema import Confidence, ExtractionSource, FieldStatus, ValueUnit

ROOM_RENT = "benefits.room_and_hospitalisation.room_rent"
PED = "benefits.waiting_periods.pre_existing_diseases"


def _rule(value, unit=ValueUnit.INR, status=FieldStatus.COVERED, score=7.0, review=False):
    return Candidate(status=status, value=value, unit=unit, raw_text="rule evidence",
                     page=2, score=score, needs_review=review)


def _llm(value, unit=ValueUnit.INR, status=FieldStatus.COVERED, evidence="llm evidence"):
    return LLMFieldResult(status=status, value=value, unit=unit, evidence=evidence, page=2)


# --------------------------------------------------------------------------------------
# Reconciliation matrix
# --------------------------------------------------------------------------------------
def test_agreement_yields_high_confidence_and_dual_source():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(75_000), _llm(75_000))
    assert field.source == ExtractionSource.RULE_AND_LLM
    assert field.confidence == Confidence.HIGH
    assert field.needs_review is False
    assert field.value == 75_000


def test_numeric_agreement_tolerates_rounding():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(500_000.0), _llm(500_000))
    assert field.confidence == Confidence.HIGH


def test_value_disagreement_prefers_llm_but_keeps_the_rule_answer():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(50_000), _llm(75_000))
    assert field.value == 75_000
    assert field.needs_review is True
    assert field.alternate and "50,000" in field.alternate
    assert field.confidence == Confidence.MEDIUM


def test_status_disagreement_is_graded_down_to_low():
    """A polarity conflict is the most consequential kind, so it must not read as reliable."""
    rule = _rule(None, unit=None, status=FieldStatus.NOT_COVERED)
    field = merge_field(SPECS_BY_PATH[PED], rule,
                        _llm(None, unit=None, status=FieldStatus.WAIVED_OFF))
    assert field.status == FieldStatus.WAIVED_OFF
    assert field.confidence == Confidence.LOW
    assert field.needs_review is True
    assert "disagreed" in (field.notes or "")


def test_rule_only_is_high_when_the_match_was_strong_and_unambiguous():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(75_000, score=8.0), None)
    assert field.source == ExtractionSource.RULE
    assert field.confidence == Confidence.HIGH


def test_rule_only_is_medium_when_the_match_was_weak_or_ambiguous():
    assert merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(75_000, score=3.0),
                       None).confidence == Confidence.MEDIUM
    assert merge_field(SPECS_BY_PATH[ROOM_RENT], _rule(75_000, review=True),
                       None).confidence == Confidence.MEDIUM


def test_llm_only_is_medium_and_says_so():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], None, _llm(75_000))
    assert field.source == ExtractionSource.LLM
    assert field.confidence == Confidence.MEDIUM
    assert "only by the LLM" in (field.notes or "")


def test_neither_extractor_yields_not_found():
    field = merge_field(SPECS_BY_PATH[ROOM_RENT], None, None)
    assert field.status == FieldStatus.NOT_FOUND
    assert field.is_populated is False


# --------------------------------------------------------------------------------------
# Response parsing and the anti-hallucination guard
# --------------------------------------------------------------------------------------
def test_unverifiable_llm_values_are_discarded(documents, monkeypatch):
    """A value whose quoted evidence is not in the document is treated as fabricated."""
    document = documents["GHI Policy.pdf"]
    specs = [SPECS_BY_PATH["benefits.infertility_and_ambulance.ambulance_charges"]]

    payload = {
        "ambulance_charges": {
            "status": "covered",
            "value": 999_999,
            "unit": "INR",
            "evidence": "Ambulance charges payable up to a maximum of Rs. 9,99,999 per trip",
        }
    }
    parsed = llm_extractor._parse_group_response(document, specs, payload)
    assert parsed == {}, "invented evidence must be rejected"


def test_verifiable_llm_values_are_kept(documents):
    document = documents["GHI Policy.pdf"]
    specs = [SPECS_BY_PATH["benefits.infertility_and_ambulance.ambulance_charges"]]
    payload = {
        "ambulance_charges": {
            "status": "covered",
            "value": "1,000",  # model returned a formatted string
            "unit": "INR",
            "basis": "per claim",
            "evidence": "Ambulance charges payable up to a maximum amount of Rs. 1,000/- "
                        "per claim.",
        }
    }
    parsed = llm_extractor._parse_group_response(document, specs, payload)
    result = parsed["benefits.infertility_and_ambulance.ambulance_charges"]
    assert result.value == 1_000.0, "comma-formatted numbers must be normalised"
    assert result.page == 2


def test_not_found_responses_are_ignored(documents):
    document = documents["GHI Policy.pdf"]
    specs = [SPECS_BY_PATH["benefits.other_benefits.ayush_treatment"]]
    payload = {"ayush_treatment": {"status": "not_found", "value": None, "evidence": None}}
    assert llm_extractor._parse_group_response(document, specs, payload) == {}


def test_llm_layer_is_skipped_when_disabled(documents):
    settings = LLMSettings(provider="none")
    assert extract_with_llm(documents["GHI Policy.pdf"], None, settings) == {}


def test_settings_are_disabled_without_an_api_key(monkeypatch):
    monkeypatch.setenv("GMC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert LLMSettings.from_env().enabled is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert LLMSettings.from_env().enabled is True


# --------------------------------------------------------------------------------------
# End-to-end hybrid run against a stub provider
# --------------------------------------------------------------------------------------
def test_hybrid_pipeline_upgrades_confidence_when_the_llm_agrees(documents, monkeypatch):
    """With a stub that echoes correct values and real quotes, agreement must be recorded."""
    document = documents["GHI Policy.pdf"]

    def fake_complete_json(settings, system, user):
        # Only answer the ambulance field; every other field stays rule-only.
        if "ambulance_charges" not in user:
            return {}
        return {
            "ambulance_charges": {
                "status": "covered",
                "value": 1000,
                "unit": "INR",
                "basis": "per claim",
                "evidence": "Ambulance charges payable up to a maximum amount of "
                            "Rs. 1,000/- per claim.",
            }
        }

    monkeypatch.setattr(llm_extractor, "complete_json", fake_complete_json)
    settings = LLMSettings(provider="openai", model="stub", api_key="sk-test")
    record = process_document(document, PipelineOptions(llm=settings))

    ambulance = record.benefits.infertility_and_ambulance.ambulance_charges
    assert record.extraction.mode == "hybrid"
    assert ambulance.source == ExtractionSource.RULE_AND_LLM.value
    assert ambulance.confidence == Confidence.HIGH.value
    assert ambulance.value == 1_000


def test_llm_failure_degrades_to_rule_only_instead_of_crashing(documents, monkeypatch):
    """A provider outage must not abort a document -- it must fall back to rule-only."""
    document = documents["GHI Policy.pdf"]
    settings = LLMSettings(provider="openai", model="stub", api_key="sk-test")

    def exploding(settings, system, user):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(llm_extractor, "complete_json", exploding)
    record = process_document(document, PipelineOptions(llm=settings))
    assert record.extraction.mode == "rule_only"
    assert any("LLM extraction failed" in w for w in record.extraction.warnings)
    # The deterministic answer is still there.
    assert record.benefits.infertility_and_ambulance.ambulance_charges.value == 1_000


def test_llm_returning_nothing_degrades_to_rule_only(documents, monkeypatch):
    document = documents["GHI Policy.pdf"]
    settings = LLMSettings(provider="openai", model="stub", api_key="sk-test")
    monkeypatch.setattr(llm_extractor, "complete_json", lambda *a, **k: None)
    record = process_document(document, PipelineOptions(llm=settings))
    assert record.extraction.mode == "rule_only"
    assert record.benefits.infertility_and_ambulance.ambulance_charges.value == 1_000



# --------------------------------------------------------------------------------------
# Regressions found by running real models through OpenRouter
# --------------------------------------------------------------------------------------
def test_fraction_percentage_is_corrected_from_the_evidence(documents):
    """gpt-4o-mini returned 0.5 for a clause reading "upto 50% of the Sum Insured"."""
    document = documents["1.Policy Copy.pdf"]
    spec = SPECS_BY_PATH["benefits.other_benefits.modern_treatment"]
    payload = {
        "modern_treatment": {
            "status": "covered",
            "value": 0.5,
            "unit": "percent_of_sum_insured",
            "evidence": "Modern treatments, psychiatric treatments and Bariatric Surgery "
                        "covered under the policy as per terms and conditions upto 50% of "
                        "the Sum Insured.",
        }
    }
    result = llm_extractor._parse_group_response(document, [spec], payload)[spec.path]
    assert result.value == 50.0
    assert "corrected" in (result.notes or "")


def test_integer_zero_value_is_repaired_not_published(documents):
    """Regression: ``isinstance(0, float)`` is False, so integers bypassed evidence repair."""
    document = documents["1.Policy Copy.pdf"]
    spec = SPECS_BY_PATH["benefits.other_benefits.modern_treatment"]
    payload = {
        "modern_treatment": {
            "status": "covered",
            "value": 0,          # int, and wrong
            "unit": "percentage",  # not a unit we recognise -> None
            "evidence": "Modern treatments, psychiatric treatments and Bariatric Surgery "
                        "covered under the policy as per terms and conditions upto 50% of "
                        "the Sum Insured.",
        }
    }
    result = llm_extractor._parse_group_response(document, [spec], payload)[spec.path]
    assert result.value == 50.0
    assert result.unit == ValueUnit.PERCENT_OF_SUM_INSURED


def test_out_of_vocabulary_status_is_coerced_when_a_value_exists(documents):
    """Models label a head count "applied"; the fact is right, the label is not."""
    document = documents["GHI Policy.pdf"]
    spec = SPECS_BY_PATH["demographics.employees"]
    payload = {
        "employees": {
            "status": "applied",
            "value": 86,
            "unit": "count",
            "evidence": "Primary Insured Members  86",
        }
    }
    result = llm_extractor._parse_group_response(document, [spec], payload)[spec.path]
    assert result.status == FieldStatus.PRESENT
    assert result.value == 86.0


def test_out_of_vocabulary_status_without_a_value_is_dropped(documents):
    document = documents["GHI Policy.pdf"]
    spec = SPECS_BY_PATH["benefits.buffer_and_waivers.corporate_buffer_per_family"]
    payload = {
        "corporate_buffer_per_family": {
            "status": "applied",
            "value": None,
            "evidence": "up to family floater Sum Insured per Insured Family",
        }
    }
    assert llm_extractor._parse_group_response(document, [spec], payload) == {}


def test_labelled_schedule_field_beats_an_llm_prose_inference():
    """Niva Bupa: labelled "Co-payment  NA" vs a 50% co-pay in Special Conditions."""
    spec = SPECS_BY_PATH["benefits.buffer_and_waivers.co_payment"]
    rule = Candidate(status=FieldStatus.NOT_SPECIFIED, value=None, raw_text="Co-payment  NA",
                     page=3, score=6.7, strategy="row_right")
    llm = LLMFieldResult(status=FieldStatus.APPLIED, value=50.0, unit=ValueUnit.PERCENT,
                         evidence="50% co-pay For cyberknife treatment", page=3)
    field = merge_field(spec, rule, llm)
    assert field.status == FieldStatus.NOT_SPECIFIED
    assert field.needs_review is True
    assert field.alternate and "LLM extractor" in field.alternate


def test_prose_rule_hit_loses_to_the_llm():
    """The reverse case: a sentence-derived rule value should not outrank the LLM."""
    spec = SPECS_BY_PATH["benefits.other_benefits.modern_treatment"]
    rule = Candidate(status=FieldStatus.COVERED, value=25.0, unit=ValueUnit.PERCENT,
                     raw_text="...", page=2, score=5.0, strategy="sentence")
    llm = LLMFieldResult(status=FieldStatus.COVERED, value=50.0, unit=ValueUnit.PERCENT,
                         evidence="upto 50% of the Sum Insured", page=2)
    field = merge_field(spec, rule, llm)
    assert field.value == 50.0
    assert field.needs_review is True


def test_agreed_status_with_one_limit_keeps_the_limit():
    spec = SPECS_BY_PATH["benefits.maternity.baby_day_one_cover"]
    rule = Candidate(status=FieldStatus.COVERED, value="up to sum insured",
                     unit=ValueUnit.TEXT, raw_text="...", page=2, score=6.0,
                     strategy="sentence")
    llm = LLMFieldResult(status=FieldStatus.COVERED, value=None, evidence="New Born Baby "
                         "covered from day one within family floater Sum Insured")
    field = merge_field(spec, rule, llm)
    assert field.value == "up to sum insured"
    assert field.needs_review is False


def test_long_enumerations_agree_despite_small_quoting_differences():
    """The rule layer picks up the label glued into the middle of a wrapped capping block."""
    spec = SPECS_BY_PATH["benefits.buffer_and_waivers.disease_wise_capping"]
    clean = ", ".join(f"Procedure{i} including cost of implants- {10000 + i * 1000}"
                      for i in range(20))
    with_label = clean.replace("Procedure10", "Disease wise capping Procedure10")
    rule = Candidate(status=FieldStatus.PRESENT, value=with_label, unit=ValueUnit.TEXT,
                     raw_text=with_label, page=3, score=6.5, strategy="block")
    llm = LLMFieldResult(status=FieldStatus.PRESENT, value=clean, unit=ValueUnit.TEXT,
                         evidence=clean)
    field = merge_field(spec, rule, llm)
    assert field.confidence == Confidence.HIGH
    assert field.needs_review is False
