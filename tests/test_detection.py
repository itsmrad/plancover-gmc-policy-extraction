"""Detection tests: registry hygiene, insurer identity, product type and TPA resolution."""

import pytest

from gmc_extract.detection import detect_insurer, detect_product_type, detect_tpa
from gmc_extract.detection.registry import INSURERS
from gmc_extract.schema import Confidence, ProductType, TPAMode

CARE = "Care Health Insurance Ltd."
NIVA_BUPA = "Niva Bupa Health Insurance Company Ltd."
LIBERTY = "Liberty General Insurance Ltd."

EXPECTED_INSURER = {
    "1.Policy Copy.pdf": CARE,
    "GHI Policy.pdf": CARE,
    "Net Catalyst - GPA - Policy Copy - 2022-23.pdf": LIBERTY,
    "Policy liberty 2022-2023.pdf": LIBERTY,
    "olj4KTUo9B1546-1692687606_925469 - 00 GMC Renewal Policy 00.pdf": NIVA_BUPA,
}

EXPECTED_PRODUCT = {
    "1.Policy Copy.pdf": ProductType.GMC,
    "GHI Policy.pdf": ProductType.GMC,
    "Net Catalyst - GPA - Policy Copy - 2022-23.pdf": ProductType.GPA,
    "Policy liberty 2022-2023.pdf": ProductType.GPA,
    "olj4KTUo9B1546-1692687606_925469 - 00 GMC Renewal Policy 00.pdf": ProductType.GMC,
}


# --------------------------------------------------------------------------------------
# Registry hygiene -- guards against a typo in a fingerprint silently misattributing a doc
# --------------------------------------------------------------------------------------
def test_insurer_keys_are_unique():
    keys = [signature.key for signature in INSURERS]
    assert len(keys) == len(set(keys))


def test_irdai_registration_numbers_are_unique():
    """A duplicated IRDAI number would let one insurer's fingerprint match another."""
    numbers = [s.irdai_registration_no for s in INSURERS if s.irdai_registration_no]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    assert not duplicates, f"duplicate IRDAI registration numbers: {duplicates}"


def test_cins_are_unique():
    cins = [s.cin for s in INSURERS if s.cin]
    assert len(cins) == len(set(cins))


# --------------------------------------------------------------------------------------
# Behaviour on the sample set
# --------------------------------------------------------------------------------------
def test_every_sample_insurer_is_identified_with_high_confidence(documents):
    for name, document in documents.items():
        detection = detect_insurer(document)
        assert detection.name == EXPECTED_INSURER[name], name
        assert detection.confidence == Confidence.HIGH, name


def test_detection_is_supported_by_multiple_independent_signals(documents):
    """Identity should not rest on a single name match."""
    for name, document in documents.items():
        signals = {evidence.signal for evidence in detect_insurer(document).evidence}
        assert len(signals) >= 3, f"{name}: only {signals}"


def test_former_names_do_not_break_identification(documents):
    """Care Health prints "formerly Religare"; Niva Bupa prints "formerly Max Bupa"."""
    care = detect_insurer(documents["GHI Policy.pdf"])
    assert care.canonical_key == "care_health"
    assert care.irdai_registration_no == "148"
    assert care.cin == "U66000DL2007PLC161503"


def test_product_type_classification(documents):
    """The two Liberty files are accident policies, not medical cover."""
    for name, document in documents.items():
        product, _scores = detect_product_type(document)
        assert product is EXPECTED_PRODUCT[name], name


def test_in_house_claims_administration_is_reported_not_left_null(documents):
    """All three sample insurers service claims themselves; null would look like a failure."""
    for name in ("1.Policy Copy.pdf", "GHI Policy.pdf",
                 "olj4KTUo9B1546-1692687606_925469 - 00 GMC Renewal Policy 00.pdf"):
        document = documents[name]
        insurer = detect_insurer(document)
        tpa = detect_tpa(document, insurer.canonical_key)
        assert tpa.mode is TPAMode.IN_HOUSE, name
        assert tpa.evidence, name


def test_broker_is_never_reported_as_a_tpa(documents):
    """Both Niva Bupa and Liberty name "Hii Insurance Broking Services" as intermediary."""
    for name, document in documents.items():
        insurer = detect_insurer(document)
        tpa = detect_tpa(document, insurer.canonical_key)
        if tpa.name:
            lowered = tpa.name.lower()
            assert "broking" not in lowered and "broker" not in lowered, name


@pytest.mark.parametrize("name", list(EXPECTED_INSURER))
def test_insurer_is_not_confused_with_a_trademark_owner(name, documents):
    """The Liberty schedule credits "Liberty Mutual" as trade-logo owner, not as the issuer."""
    detection = detect_insurer(documents[name])
    assert detection.name != "Liberty Mutual"
