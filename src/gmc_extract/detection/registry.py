"""Signature registry for insurers, TPAs and product types.

**This is the file you edit to support a new insurer.** No code changes required -- that is
the concrete answer to "how does this scale beyond the insurers you tested?".

Each insurer carries several *independent* fingerprints. Detection sums the weights of
whichever fingerprints hit, so a document does not need to contain any one specific token to
be identified. This matters because policy documents routinely:

* rename themselves -- "Care Health Insurance Ltd. (formerly known as Religare Health
  Insurance Company Limited)", "Niva Bupa (formerly known as Max Bupa)";
* mention *other* insurers -- the Liberty schedule credits "Liberty Mutual" as the trade
  logo owner, which is not the issuing entity;
* omit the full legal name from the schedule page entirely, leaving only a website, an
  IRDAI registration number or a CIN in the footer.

IRDAI registration numbers and CINs are near-unique fingerprints and carry the highest
weight; they break ties that names alone cannot. Numbers here were taken from the sample
policy footers where available, otherwise from published IRDAI-registration listings, and a
uniqueness test guards against a typo silently colliding with another insurer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schema import ProductType

# Weights per signal type. Deliberately coarse -- fine-tuning these on five documents would
# be overfitting, and the ranking is stable across any sensible assignment.
WEIGHT_LEGAL_NAME = 3.0
WEIGHT_CIN = 3.0
WEIGHT_IRDAI = 2.5
WEIGHT_DOMAIN = 2.5
WEIGHT_ALIAS = 1.5
WEIGHT_BRAND = 0.8


@dataclass(frozen=True)
class InsurerSignature:
    key: str
    display_name: str
    #: Full legal names, including former names the document may still print.
    legal_names: List[str] = field(default_factory=list)
    #: Shorter aliases that are still specific to this insurer.
    aliases: List[str] = field(default_factory=list)
    #: Weak brand tokens -- only meaningful in combination with something else.
    brand_tokens: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    irdai_registration_no: Optional[str] = None
    cin: Optional[str] = None


INSURERS: List[InsurerSignature] = [
    InsurerSignature(
        key="care_health",
        display_name="Care Health Insurance Ltd.",
        legal_names=["care health insurance limited", "care health insurance ltd",
                     "religare health insurance company limited"],
        aliases=["care health insurance", "religare health insurance"],
        brand_tokens=["careinsurance", "care health"],
        domains=["careinsurance.com"],
        irdai_registration_no="148",
        cin="U66000DL2007PLC161503",
    ),
    InsurerSignature(
        key="niva_bupa",
        display_name="Niva Bupa Health Insurance Company Ltd.",
        legal_names=["niva bupa health insurance company limited",
                     "max bupa health insurance company limited"],
        aliases=["niva bupa", "max bupa"],
        brand_tokens=["nivabupa"],
        domains=["nivabupa.com"],
        irdai_registration_no="145",
        cin="U66000DL2008PLC182918",
    ),
    InsurerSignature(
        key="liberty_general",
        display_name="Liberty General Insurance Ltd.",
        legal_names=["liberty general insurance limited", "liberty general insurance ltd"],
        aliases=["liberty general insurance"],
        brand_tokens=["libertyinsurance"],
        domains=["libertyinsurance.in"],
        irdai_registration_no="150",
        cin="U66000MH2010PLC209656",
    ),
    InsurerSignature(
        key="tata_aig",
        display_name="TATA AIG General Insurance Co. Ltd.",
        legal_names=["tata aig general insurance company limited",
                     "tata aig general insurance co. ltd"],
        aliases=["tata aig"],
        brand_tokens=["tataaig"],
        domains=["tataaig.com"],
        irdai_registration_no="108",
        cin="U85110MH2000PLC128425",
    ),
    InsurerSignature(
        key="icici_lombard",
        display_name="ICICI Lombard General Insurance Co. Ltd.",
        legal_names=["icici lombard general insurance company limited"],
        aliases=["icici lombard"],
        brand_tokens=["icicilombard"],
        domains=["icicilombard.com"],
        irdai_registration_no="115",
    ),
    InsurerSignature(
        key="hdfc_ergo",
        display_name="HDFC ERGO General Insurance Co. Ltd.",
        legal_names=["hdfc ergo general insurance company limited",
                     "hdfc ergo health insurance limited", "apollo munich health insurance"],
        aliases=["hdfc ergo"],
        brand_tokens=["hdfcergo"],
        domains=["hdfcergo.com"],
        irdai_registration_no="146",
    ),
    InsurerSignature(
        key="bajaj_allianz",
        display_name="Bajaj Allianz General Insurance Co. Ltd.",
        legal_names=["bajaj allianz general insurance company limited"],
        aliases=["bajaj allianz"],
        brand_tokens=["bajajallianz"],
        domains=["bajajallianz.com", "bajajallianz.co.in"],
        irdai_registration_no="113",
    ),
    InsurerSignature(
        key="star_health",
        display_name="Star Health & Allied Insurance Co. Ltd.",
        legal_names=["star health and allied insurance company limited",
                     "star health & allied insurance co. ltd"],
        aliases=["star health"],
        brand_tokens=["starhealth"],
        domains=["starhealth.in"],
        irdai_registration_no="129",
    ),
    InsurerSignature(
        key="aditya_birla_health",
        display_name="Aditya Birla Health Insurance Co. Ltd.",
        legal_names=["aditya birla health insurance co. limited",
                     "aditya birla health insurance company limited"],
        aliases=["aditya birla health"],
        brand_tokens=["adityabirlahealth", "activ health"],
        domains=["adityabirlacapital.com"],
        irdai_registration_no="109",
    ),
    InsurerSignature(
        key="manipal_cigna",
        display_name="ManipalCigna Health Insurance Company Ltd.",
        legal_names=["manipalcigna health insurance company limited",
                     "cignattk health insurance company limited"],
        aliases=["manipalcigna", "manipal cigna", "cigna ttk"],
        brand_tokens=["manipalcigna"],
        domains=["manipalcigna.com"],
        irdai_registration_no="151",
    ),
    InsurerSignature(
        key="new_india",
        display_name="The New India Assurance Co. Ltd.",
        legal_names=["the new india assurance company limited",
                     "new india assurance co. ltd"],
        aliases=["new india assurance"],
        brand_tokens=["newindia"],
        domains=["newindia.co.in"],
    ),
    InsurerSignature(
        key="united_india",
        display_name="United India Insurance Co. Ltd.",
        legal_names=["united india insurance company limited"],
        aliases=["united india insurance"],
        domains=["uiic.co.in"],
    ),
    InsurerSignature(
        key="oriental_insurance",
        display_name="The Oriental Insurance Co. Ltd.",
        legal_names=["the oriental insurance company limited"],
        aliases=["oriental insurance"],
        domains=["orientalinsurance.org.in"],
    ),
    InsurerSignature(
        key="national_insurance",
        display_name="National Insurance Co. Ltd.",
        legal_names=["national insurance company limited"],
        aliases=["national insurance co"],
        domains=["nationalinsurance.nic.co.in"],
        irdai_registration_no="58",
    ),
    InsurerSignature(
        key="sbi_general",
        display_name="SBI General Insurance Co. Ltd.",
        legal_names=["sbi general insurance company limited"],
        aliases=["sbi general insurance"],
        brand_tokens=["sbigeneral"],
        domains=["sbigeneral.in"],
        irdai_registration_no="144",
    ),
    InsurerSignature(
        key="reliance_general",
        display_name="Reliance General Insurance Co. Ltd.",
        legal_names=["reliance general insurance company limited"],
        aliases=["reliance general insurance"],
        brand_tokens=["reliancegeneral"],
        domains=["reliancegeneral.co.in"],
        irdai_registration_no="103",
    ),
    InsurerSignature(
        key="cholamandalam_ms",
        display_name="Cholamandalam MS General Insurance Co. Ltd.",
        legal_names=["cholamandalam ms general insurance company limited"],
        aliases=["cholamandalam ms"],
        brand_tokens=["cholainsurance"],
        domains=["cholainsurance.com"],
        irdai_registration_no="123",
    ),
    InsurerSignature(
        key="future_generali",
        display_name="Future Generali India Insurance Co. Ltd.",
        legal_names=["future generali india insurance company limited"],
        aliases=["future generali"],
        domains=["futuregenerali.in"],
        irdai_registration_no="133",
    ),
    InsurerSignature(
        key="iffco_tokio",
        display_name="IFFCO Tokio General Insurance Co. Ltd.",
        legal_names=["iffco tokio general insurance company limited"],
        aliases=["iffco tokio"],
        domains=["iffcotokio.co.in"],
        irdai_registration_no="106",
    ),
    InsurerSignature(
        key="royal_sundaram",
        display_name="Royal Sundaram General Insurance Co. Ltd.",
        legal_names=["royal sundaram general insurance co. limited"],
        aliases=["royal sundaram"],
        domains=["royalsundaram.in"],
        irdai_registration_no="102",
    ),
    InsurerSignature(
        key="magma_hdi",
        display_name="Magma HDI General Insurance Co. Ltd.",
        legal_names=["magma hdi general insurance company limited"],
        aliases=["magma hdi"],
        domains=["magmahdi.com"],
        irdai_registration_no="149",
    ),
    InsurerSignature(
        key="kotak_general",
        display_name="Kotak Mahindra General Insurance Co. Ltd.",
        legal_names=["kotak mahindra general insurance company limited"],
        aliases=["kotak mahindra general"],
        domains=["kotakgeneral.com"],
        irdai_registration_no="152",
    ),
    InsurerSignature(
        key="go_digit",
        display_name="Go Digit General Insurance Ltd.",
        legal_names=["go digit general insurance limited"],
        aliases=["go digit", "digit insurance"],
        domains=["godigit.com"],
        irdai_registration_no="158",
    ),
    InsurerSignature(
        key="zuno_general",
        display_name="Zuno General Insurance Ltd.",
        legal_names=["zuno general insurance limited",
                     "edelweiss general insurance company limited"],
        aliases=["zuno general", "edelweiss general insurance"],
        domains=["hizuno.com"],
        irdai_registration_no="159",
    ),
    InsurerSignature(
        key="bharti_axa",
        display_name="Bharti AXA General Insurance Co. Ltd.",
        legal_names=["bharti axa general insurance company limited"],
        aliases=["bharti axa"],
        domains=["bharti-axagi.co.in"],
        irdai_registration_no="139",
    ),
    InsurerSignature(
        key="universal_sompo",
        display_name="Universal Sompo General Insurance Co. Ltd.",
        legal_names=["universal sompo general insurance company limited"],
        aliases=["universal sompo"],
        domains=["universalsompo.com"],
    ),
    InsurerSignature(
        key="raheja_qbe",
        display_name="Raheja QBE General Insurance Co. Ltd.",
        legal_names=["raheja qbe general insurance company limited"],
        aliases=["raheja qbe"],
        domains=["rahejaqbe.com"],
        irdai_registration_no="141",
    ),
    InsurerSignature(
        key="shriram_general",
        display_name="Shriram General Insurance Co. Ltd.",
        legal_names=["shriram general insurance company limited"],
        aliases=["shriram general insurance"],
        domains=["shriramgi.com"],
        irdai_registration_no="137",
    ),
]

INSURERS_BY_KEY: Dict[str, InsurerSignature] = {s.key: s for s in INSURERS}


# --------------------------------------------------------------------------------------
# TPAs
# --------------------------------------------------------------------------------------
#: Known third-party administrators. Order is irrelevant; matching is longest-alias-first.
KNOWN_TPAS: Dict[str, List[str]] = {
    "Medi Assist Insurance TPA Pvt. Ltd.": ["medi assist insurance tpa", "medi assist",
                                            "mediassist"],
    "Paramount Health Services & Insurance TPA Pvt. Ltd.": ["paramount health services",
                                                            "paramount tpa"],
    "Vidal Health Insurance TPA Pvt. Ltd.": ["vidal health insurance tpa", "vidal health",
                                             "good health tpa"],
    "Family Health Plan Insurance TPA Ltd. (FHPL)": ["family health plan insurance tpa",
                                                     "family health plan", "fhpl"],
    "Ericson Insurance TPA Pvt. Ltd.": ["ericson insurance tpa", "ericson tpa"],
    "Raksha Health Insurance TPA Pvt. Ltd.": ["raksha health insurance tpa", "raksha tpa"],
    "Health India Insurance TPA Services Pvt. Ltd.": ["health india insurance tpa",
                                                      "health india tpa"],
    "MDIndia Health Insurance TPA Pvt. Ltd.": ["mdindia health insurance tpa", "mdindia"],
    "Heritage Health Insurance TPA Pvt. Ltd.": ["heritage health insurance tpa",
                                                "heritage health tpa"],
    "Safeway Insurance TPA Pvt. Ltd.": ["safeway insurance tpa", "safeway tpa"],
    "East West Assist Insurance TPA Pvt. Ltd.": ["east west assist"],
    "Genins India Insurance TPA Ltd.": ["genins india"],
    "Vipul MedCorp Insurance TPA Pvt. Ltd.": ["vipul medcorp", "vipul med corp"],
    "Alankit Insurance TPA Ltd.": ["alankit insurance tpa", "alankit health care"],
    "Park Mediclaim Insurance TPA Pvt. Ltd.": ["park mediclaim"],
    "Rothshield Insurance TPA Ltd.": ["rothshield"],
    "Anmol Medicare Insurance TPA Ltd.": ["anmol medicare"],
    "Grand Healthcare Insurance TPA Pvt. Ltd.": ["grand healthcare"],
    "Volo Health Insurance TPA Pvt. Ltd.": ["volo health"],
    "Happy Insurance TPA Services Pvt. Ltd.": ["happy insurance tpa"],
}

#: Labels that introduce the claims-administering entity. Used for proximity detection so
#: an unknown TPA can still be found by position rather than by name.
TPA_LABELS = [
    "existing tpa",
    "tpa name",
    "name of tpa",
    "third party administrator",
    "claims administrator",
    "claim administrator",
    "tpa details",
    "servicing tpa",
    "claims servicing team",
    "claims servicing",
    "claims service provider",
    "tpa",
]

#: Phrases indicating the insurer administers claims itself (no external TPA).
IN_HOUSE_MARKERS = [
    "in-house claims",
    "in house claims",
    "no tpa",
    "without tpa",
    "claims are settled directly",
    "direct settlement by the company",
]


# --------------------------------------------------------------------------------------
# Product type
# --------------------------------------------------------------------------------------
#: Cue -> weight per product type. Highest total wins.
PRODUCT_CUES: Dict[ProductType, Dict[str, float]] = {
    ProductType.GMC: {
        "group mediclaim": 4.0,
        "group medical": 4.0,
        "group health insurance": 3.5,
        "group care": 2.5,
        "hospitalisation": 1.0,
        "hospitalization": 1.0,
        "in-patient care": 2.0,
        "inpatient care": 2.0,
        "room rent": 2.0,
        "maternity": 2.0,
        "pre-existing disease": 1.5,
        "day care treatment": 1.5,
        "mediclaim": 2.0,
        "health plus": 1.5,
        "sum insured": 0.5,
    },
    ProductType.GPA: {
        "personal accident": 4.0,
        "accidental death": 3.0,
        "permanent total disability": 3.0,
        "permanent partial disability": 3.0,
        "temporary total disability": 3.0,
        "capital sum insured": 2.5,
        "accidental medical expenses": 1.5,
    },
    ProductType.GTL: {
        "group term life": 4.0,
        "term life": 2.5,
        "sum assured": 2.0,
        "death benefit": 1.5,
    },
    ProductType.OPD: {
        "opd only": 3.0,
        "out patient department cover": 2.5,
    },
}
