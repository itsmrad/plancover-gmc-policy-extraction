"""Declarative catalogue of every QMS field.

**This is the file that carries the adaptability claim.** A spec never says "insurer X puts
room rent on page 2". It says "room rent is expressed using one of these cues, and its value
is a percentage of sum insured, a rupee cap, or one of these textual limits". That is
knowledge about *insurance language*, which transfers to unseen insurers; positional
knowledge would not.

Cue lists are ordered most-specific first, and specificity earns a scoring bonus, so
``"maximum eligibility for icu hospitalization"`` beats a bare ``"icu"`` when both match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..schema import ProductType
from .parsers import StatusMode

GMC_ONLY = (ProductType.GMC,)
ANY_PRODUCT = (ProductType.GMC, ProductType.GPA, ProductType.GTL, ProductType.OPD,
               ProductType.UNKNOWN)


class ValueKind(str, Enum):
    #: A rupee amount.
    MONEY = "money"
    #: Either a percentage of sum insured, a rupee cap, or a textual limit ("No Limit").
    PERCENT_OR_MONEY = "percent_or_money"
    #: A bare percentage (co-pay, discount).
    PERCENT = "percent"
    DAYS = "days"
    MONTHS = "months"
    #: Coverage verdict only.
    STATUS = "status"
    #: Coverage verdict plus a limit if one is stated.
    STATUS_WITH_LIMIT = "status_with_limit"
    COUNT = "count"
    DATE = "date"
    TEXT = "text"


@dataclass(frozen=True)
class FieldSpec:
    #: Dotted destination in :class:`~gmc_extract.schema.QMSPolicyRecord`, or a
    #: ``scratch.*`` key for intermediate values the mapper post-processes.
    path: str
    label: str
    group: str
    kind: ValueKind
    cues: Tuple[str, ...]
    #: Optional region anchors. When present, cues are only searched near an anchor, which
    #: is how "Normal 1%" (room rent) is kept distinct from "Normal 25,000" (maternity).
    anchor_cues: Tuple[str, ...] = ()
    #: A window containing any of these is rejected outright.
    negative_cues: Tuple[str, ...] = ()
    status_mode: StatusMode = StatusMode.BENEFIT
    products: Tuple[ProductType, ...] = GMC_ONLY
    window_before: int = 220
    window_after: int = 420
    require_currency: bool = False
    #: Which duration to take when several appear ("30 days and 60 days respectively").
    duration_index: int = 0
    #: Capture the whole paragraph rather than a single cell (long enumerations).
    capture_block: bool = False
    #: Capture the logical sentence containing the cue (prose conditions).
    capture_sentence: bool = False
    #: Treat a bare cue hit with no explicit polarity word as covered/applied.
    presence_implies_covered: bool = False
    #: Status assigned by ``presence_implies_covered``.
    presence_status: str = "covered"
    #: A bare "2%" against a room-rent style field means 2% *of sum insured*; there is
    #: nothing else in a GMC schedule for it to be a percentage of.
    percent_defaults_to_sum_insured: bool = False
    notes: Optional[str] = None


# ======================================================================================
# Policy meta
# ======================================================================================
POLICY_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="policy.policy_number", label="Policy Number", group="policy_meta",
        kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("policy number", "policy no.", "policy no", "certificate number"),
    ),
    FieldSpec(
        path="policy.policyholder_name", label="Policyholder Name", group="policy_meta",
        kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("name of policyholder", "policyholder's name", "policy holder's name",
              "policyholder name", "name of the insured", "insured name", "name of insured"),
    ),
    FieldSpec(
        path="policy.previous_year_premium.net_premium", label="Net Premium (pre-tax)",
        group="policy_meta", kind=ValueKind.MONEY, products=ANY_PRODUCT,
        cues=("total net premium", "net premium (taxable value)", "net premium",
              "basic premium", "premium (taxable value)", "hospitalization cover premium",
              "premium details", "premium"),
        negative_cues=("premium per life", "premium rater", "next premium due"),
        require_currency=False,
    ),
    FieldSpec(
        path="policy.previous_year_premium.gross_premium", label="Gross Premium (incl. tax)",
        group="policy_meta", kind=ValueKind.MONEY, products=ANY_PRODUCT,
        cues=("total premium", "gross premium", "premium including tax",
              "total premium payable"),
        negative_cues=("premium per life", "premium rater", "in words"),
    ),
    FieldSpec(
        path="policy.previous_year_premium.payment_mode", label="Premium Payment Mode",
        group="policy_meta", kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("premium payment mode", "premium payment frequency", "payment mode",
              "premium paying frequency", "mode of payment"),
    ),
]

# ======================================================================================
# Demographics
# ======================================================================================
DEMOGRAPHIC_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="demographics.employees", label="No. of Employees", group="demographics",
        kind=ValueKind.COUNT, products=ANY_PRODUCT,
        cues=("total no of employees/primary members", "total no of employees",
              "number of employees", "no. of employees", "no of employees",
              "primary insured members", "primary members", "total employees",
              "employee count", "no of insured persons"),
    ),
    FieldSpec(
        path="demographics.spouses", products=ANY_PRODUCT, label="No. of Spouses", group="demographics",
        kind=ValueKind.COUNT,
        cues=("no of spouse", "no. of spouses", "number of spouses", "total spouses",
              "spouse count", "spouses covered"),
    ),
    FieldSpec(
        path="demographics.children", products=ANY_PRODUCT, label="No. of Children", group="demographics",
        kind=ValueKind.COUNT,
        cues=("no of children", "no. of children", "number of children", "total children",
              "children covered", "no of dependent children"),
    ),
    FieldSpec(
        path="demographics.parents", products=ANY_PRODUCT, label="No. of Parents", group="demographics",
        kind=ValueKind.COUNT,
        cues=("no of parents", "no. of parents", "number of parents", "total parents",
              "parents covered"),
    ),
    FieldSpec(
        path="demographics.parents_in_law", products=ANY_PRODUCT, label="No. of Parents-in-law",
        group="demographics", kind=ValueKind.COUNT,
        cues=("no of parents in law", "parents-in-law", "parents in law", "in-laws covered"),
    ),
    FieldSpec(
        path="demographics.dependents_total", label="No. of Dependents",
        group="demographics", kind=ValueKind.COUNT, products=ANY_PRODUCT,
        cues=("total dependents", "dependents", "dependants", "no of dependents"),
    ),
    FieldSpec(
        path="demographics.total_lives", label="Total Lives Covered", group="demographics",
        kind=ValueKind.COUNT, products=ANY_PRODUCT,
        cues=("total no of insured members", "total number of lives", "total lives covered",
              "total lives", "total insured members", "total no of members",
              "total members covered", "no of insured persons", "number of lives"),
    ),
]

# ======================================================================================
# A. Room rent & hospitalisation
# ======================================================================================
_ROOM_ANCHORS = ("hospital accommodation", "room rent", "room and boarding",
                 "accommodation charges", "room category", "in-patient care",
                 "inpatient care", "hospitalisation benefit", "hospitalization benefit")

ROOM_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.room_and_hospitalisation.room_rent", label="Room Rent",
        group="room_and_hospitalisation", kind=ValueKind.PERCENT_OR_MONEY,
        anchor_cues=_ROOM_ANCHORS,
        cues=("maximum eligibility for normal hospitalization", "normal hospitalization",
              "normal hospitalisation", "room rent limit", "room rent per day",
              "other than icu", "non-icu", "normal", "room rent", "accommodation"),
        negative_cues=("air ambulance",),
        window_before=260, window_after=380, percent_defaults_to_sum_insured=True,
    ),
    FieldSpec(
        path="benefits.room_and_hospitalisation.icu_charges", label="ICU Charges",
        group="room_and_hospitalisation", kind=ValueKind.PERCENT_OR_MONEY,
        anchor_cues=_ROOM_ANCHORS,
        cues=("maximum eligibility for icu hospitalization", "icu hospitalization",
              "icu hospitalisation", "icu charges", "icu rent", "intensive care unit",
              "icu/day", "icu"),
        window_before=260, window_after=380, percent_defaults_to_sum_insured=True,
    ),
    FieldSpec(
        path="benefits.room_and_hospitalisation.pre_hospitalization",
        label="Pre-Hospitalization", group="room_and_hospitalisation", kind=ValueKind.DAYS,
        cues=("pre hospitalization medical expenses", "pre-hospitalization medical expenses",
              "pre hospitalization", "pre-hospitalization", "pre hospitalisation",
              "pre-hospitalisation", "pre & post hospitalization",
              "pre and post hospitalization", "pre & post hospitalisation"),
        duration_index=0,
    ),
    FieldSpec(
        path="benefits.room_and_hospitalisation.post_hospitalization",
        label="Post-Hospitalization", group="room_and_hospitalisation", kind=ValueKind.DAYS,
        cues=("post hospitalization medical expenses",
              "post-hospitalization medical expenses", "post hospitalization",
              "post-hospitalization", "post hospitalisation", "post-hospitalisation",
              "pre & post hospitalization", "pre and post hospitalization"),
        duration_index=1,
    ),
    FieldSpec(
        path="benefits.room_and_hospitalisation.room_rent_proportionate_deduction",
        label="Proportionate Deduction on Room Rent Breach",
        group="room_and_hospitalisation", kind=ValueKind.STATUS,
        cues=("ratable proportion", "rateable proportion", "pro-rated proportion",
              "proportionate deduction", "pro rata proportion", "ratable share"),
        presence_implies_covered=True, presence_status="applied",
        notes="Presence of a proportionate-deduction clause is reported as APPLIED.",
    ),
]

# ======================================================================================
# B. Maternity
# ======================================================================================
_MATERNITY_ANCHORS = ("maternity expenses", "maternity benefit", "maternity claims",
                      "maternity", "delivery")

MATERNITY_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.maternity.nine_month_waiting_period",
        label="9-Month Maternity Waiting Period", group="maternity", kind=ValueKind.STATUS,
        status_mode=StatusMode.WAITING_PERIOD,
        cues=("9 month waiting period", "9 months waiting period",
              "nine month waiting period", "nine months waiting period",
              "maternity waiting period", "waiting period for maternity",
              "9 month wait", "10 month waiting period", "waiting period in respect of "
              "maternity"),
    ),
    FieldSpec(
        path="benefits.maternity.baby_day_one_cover", label="Baby Day One Cover",
        group="maternity", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("new born baby covered from day one", "baby day one cover", "day one cover",
              "new born baby", "newborn baby", "new born", "newborn", "baby from day one",
              "child from day one"),
    ),
    FieldSpec(
        path="benefits.maternity.vaccination_cover", label="Vaccination Cover",
        group="maternity", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("baby vaccination", "vaccination expenses", "vaccination", "vaccine",
              "immunisation", "immunization", "inoculation"),
    ),
    # Metro / non-metro variants are attempted explicitly first; the flat limits below act
    # as the fallback the mapper uses when a document does not differentiate.
    FieldSpec(
        path="benefits.maternity.normal_delivery_metro", label="Normal Delivery (Metro)",
        group="maternity", kind=ValueKind.MONEY, anchor_cues=_MATERNITY_ANCHORS,
        cues=("normal delivery metro", "normal - metro", "normal (metro)", "metro normal",
              "normal delivery - metro"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="benefits.maternity.normal_delivery_non_metro",
        label="Normal Delivery (Non-Metro)", group="maternity", kind=ValueKind.MONEY,
        anchor_cues=_MATERNITY_ANCHORS,
        cues=("normal delivery non metro", "normal - non metro", "normal (non-metro)",
              "non metro normal", "normal delivery - non metro"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="benefits.maternity.c_section_metro", label="C-Section (Metro)",
        group="maternity", kind=ValueKind.MONEY, anchor_cues=_MATERNITY_ANCHORS,
        cues=("c-section metro", "caesarean metro", "lscs metro", "c section - metro",
              "c-section (metro)"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="benefits.maternity.c_section_non_metro", label="C-Section (Non-Metro)",
        group="maternity", kind=ValueKind.MONEY, anchor_cues=_MATERNITY_ANCHORS,
        cues=("c-section non metro", "caesarean non metro", "lscs non metro",
              "c section - non metro", "c-section (non-metro)"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="scratch.normal_delivery_limit", label="Normal Delivery Limit (flat)",
        group="maternity", kind=ValueKind.MONEY, anchor_cues=_MATERNITY_ANCHORS,
        cues=("for normal delivery", "normal delivery", "for normal", "normal"),
        negative_cues=("normal hospitalization", "normal hospitalisation"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="scratch.c_section_limit", label="C-Section Limit (flat)", group="maternity",
        kind=ValueKind.MONEY, anchor_cues=_MATERNITY_ANCHORS,
        cues=("for c-section", "c-section", "c section", "caesarean", "caesarian", "lscs",
              "cesarean"),
        window_before=350, window_after=520,
    ),
    FieldSpec(
        path="benefits.maternity.pre_post_natal_expenses",
        label="Pre & Post Natal Expenses", group="maternity",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("pre & post natal expenses", "pre and post natal expenses", "pre & post natal",
              "pre-natal and post-natal", "ante natal", "antenatal", "pre natal",
              "post natal"),
    ),
    FieldSpec(
        path="benefits.maternity.maternity_child_limit",
        label="Maternity Deliveries Covered", group="maternity", kind=ValueKind.TEXT,
        cues=("maternity claim is payable for first", "first two dependent children",
              "first two children", "two children only", "maximum of 2 pregnancies",
              "twice during the lifetime", "up to two deliveries"),
        capture_sentence=True,
    ),
]

# ======================================================================================
# C. Waiting periods
# ======================================================================================
WAITING_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.waiting_periods.thirty_day_waiting_period",
        label="30-Day Initial Waiting Period", group="waiting_periods",
        kind=ValueKind.STATUS, status_mode=StatusMode.WAITING_PERIOD,
        cues=("30 days wait period", "30 day waiting period", "30 days waiting period",
              "thirty days waiting period", "initial waiting period",
              "first 30 days waiting", "30 days exclusion", "initial 30 days"),
    ),
    FieldSpec(
        path="benefits.waiting_periods.first_and_second_year_waiting_period",
        label="1st / 2nd Year Waiting Period", group="waiting_periods",
        kind=ValueKind.STATUS, status_mode=StatusMode.WAITING_PERIOD,
        cues=("first & second year exclusion", "first and second year exclusion",
              "1st & 2nd year waiting period", "1st and 2nd year waiting period",
              "first and second year waiting period", "2 yr exclusions",
              "2 year exclusions", "two year waiting period", "24 months waiting period",
              "first year exclusion", "specific disease waiting period",
              "disease specific exclusions", "named ailment waiting period"),
    ),
    FieldSpec(
        path="benefits.waiting_periods.pre_existing_diseases",
        label="Pre-Existing Disease Waiting Period", group="waiting_periods",
        kind=ValueKind.STATUS, status_mode=StatusMode.WAITING_PERIOD,
        cues=("pre-existing disease (ped)", "pre-existing diseases", "pre existing diseases",
              "pre-existing disease", "pre existing disease", "pre-existing condition",
              "ped waiting period", "ped"),
    ),
]

# ======================================================================================
# D. Other benefits
# ======================================================================================
OTHER_BENEFIT_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.other_benefits.day_care_expenses", label="Day Care Expenses",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("listed day care treatment", "day care treatment", "day care procedures",
              "day care expenses", "daycare treatment", "day care", "daycare"),
        presence_implies_covered=True,
    ),
    FieldSpec(
        path="benefits.other_benefits.opd_benefit", label="OPD Benefit",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("opd coverage", "opd benefit", "opd treatment", "out patient department",
              "outpatient department", "out-patient treatment", "opd expenses", "opd"),
    ),
    FieldSpec(
        path="benefits.other_benefits.teleconsultation", label="Teleconsultation",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("teleconsultation", "tele consultation", "tele-consultation",
              "econsultation", "e-consultation", "e consultation", "telemedicine",
              "online consultation", "video consultation", "doctor on call"),
        presence_implies_covered=True,
    ),
    FieldSpec(
        path="benefits.other_benefits.pharmacy_discount", label="Pharmacy Discount",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("pharmacy discount", "discount on pharmacy", "chemist discount",
              "medicine discount", "discount on medicines", "pharmacy benefit"),
    ),
    FieldSpec(
        path="benefits.other_benefits.domiciliary_hospitalization",
        label="Domiciliary Hospitalization", group="other_benefits",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("domiciliary hospitalization", "domiciliary hospitalisation",
              "domiciliary treatment", "home treatment", "domiciliary"),
    ),
    FieldSpec(
        path="benefits.other_benefits.annual_health_checkup",
        label="Annual Health Check-Up", group="other_benefits",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("annual health check-up", "annual health checkup", "annual health check up",
              "preventive health check", "master health check", "health check-up",
              "health checkup", "wellness check"),
    ),
    FieldSpec(
        path="benefits.other_benefits.modern_treatment", label="Modern Treatment",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("modern treatments", "modern treatment", "advanced treatment",
              "modern medical treatment", "cyberknife", "robotic surgery",
              "stem cell therapy", "oral chemotherapy"),
        presence_implies_covered=True,
    ),
    FieldSpec(
        path="benefits.other_benefits.bariatric_treatment", label="Bariatric Treatment",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("bariatric surgery", "bariatric treatment", "obesity surgery", "bariatric"),
    ),
    FieldSpec(
        path="benefits.other_benefits.psychiatric_treatment", label="Psychiatric Treatment",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("psychiatric treatments", "psychiatric treatment", "psychiatric ailments",
              "mental illness", "mental health", "psychiatric"),
    ),
    FieldSpec(
        path="benefits.other_benefits.ayush_treatment", label="AYUSH Treatment",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("ayush treatment", "ayush", "ayurvedic treatment", "ayurveda", "homeopathy",
              "homoeopathy", "unani", "siddha", "non-allopathic treatment"),
    ),
    FieldSpec(
        path="benefits.other_benefits.lgbtq_coverage", label="LGBTQ+ Coverage",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("lgbtq", "lgbt", "same sex partner", "same-sex partner", "same gender partner",
              "transgender", "queer partner"),
    ),
    FieldSpec(
        path="benefits.other_benefits.live_in_partner_coverage",
        label="Live-in Partner Coverage", group="other_benefits",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("live-in partner", "live in partner", "livein partner", "domestic partner",
              "unmarried partner", "common law partner"),
    ),
    FieldSpec(
        path="benefits.other_benefits.organ_donor_expenses", label="Organ Donor Expenses",
        group="other_benefits", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("organ donor expenses", "organ donor", "organ donation", "donor expenses",
              "transplant donor"),
    ),
]

# ======================================================================================
# E. Infertility & ambulance
# ======================================================================================
INFERTILITY_AMBULANCE_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.infertility_and_ambulance.infertility_treatment",
        label="Infertility Treatment", group="infertility_and_ambulance",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("infertility treatment", "infertility & related ailments", "infertility",
              "in vitro fertilization", "in-vitro fertilisation", "ivf",
              "assisted reproduction", "assisted reproductive", "sub-fertility",
              "male sterility", "sterility"),
    ),
    FieldSpec(
        path="benefits.infertility_and_ambulance.surrogacy", label="Surrogacy",
        group="infertility_and_ambulance", kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("surrogacy", "surrogate mother", "surrogate"),
    ),
    FieldSpec(
        path="benefits.infertility_and_ambulance.ambulance_charges",
        label="Ambulance Charges", group="infertility_and_ambulance",
        kind=ValueKind.STATUS_WITH_LIMIT, products=ANY_PRODUCT,
        cues=("emergency ambulance", "ambulance charges", "road ambulance",
              "ambulance expenses", "ambulance cover", "ambulance"),
        negative_cues=("air ambulance",),
    ),
    FieldSpec(
        path="benefits.infertility_and_ambulance.air_ambulance_charges",
        label="Air Ambulance Charges", group="infertility_and_ambulance",
        kind=ValueKind.STATUS_WITH_LIMIT,
        cues=("air ambulance", "air-ambulance", "helicopter ambulance", "aeroplane ambulance"),
    ),
]

# ======================================================================================
# F. Buffer & waivers
# ======================================================================================
BUFFER_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="benefits.buffer_and_waivers.corporate_buffer", label="Corporate Buffer",
        group="buffer_and_waivers", kind=ValueKind.STATUS,
        cues=("corporate floater sum insured", "corporate buffer", "corporate floater",
              "company buffer", "buffer sum insured", "corporate reserve",
              "floater buffer"),
    ),
    FieldSpec(
        path="benefits.buffer_and_waivers.corporate_buffer_limit",
        label="Corporate Buffer Limit", group="buffer_and_waivers", kind=ValueKind.MONEY,
        cues=("corporate floater limit", "corporate buffer limit",
              "aggregate liability in respect of all such claims under corporate floater",
              "buffer limit", "corporate floater shall not exceed", "buffer of"),
        window_after=520,
    ),
    FieldSpec(
        path="benefits.buffer_and_waivers.corporate_buffer_per_family",
        label="Corporate Buffer Utilisation per Family", group="buffer_and_waivers",
        kind=ValueKind.MONEY,
        cues=("corporate floater utilization/family", "corporate floater utilisation",
              "buffer per family", "buffer utilization per family",
              "buffer utilisation per family"),
    ),
    FieldSpec(
        path="benefits.buffer_and_waivers.disease_wise_capping",
        label="Disease-wise Capping", group="buffer_and_waivers", kind=ValueKind.TEXT,
        cues=("disease wise capping", "disease-wise capping", "procedure wise capping",
              "ailment wise capping", "surgery wise capping", "disease wise sub limit",
              "procedure-wise sub-limits"),
        capture_block=True,
    ),
    FieldSpec(
        path="benefits.buffer_and_waivers.co_payment", label="Co-payment",
        group="buffer_and_waivers", kind=ValueKind.PERCENT, products=ANY_PRODUCT,
        cues=("co-payment", "co payment", "copayment", "co-pay", "copay"),
    ),
]

# ======================================================================================
# Policy period & structure (scratch values assembled by the mapper)
# ======================================================================================
PERIOD_SPECS: List[FieldSpec] = [
    FieldSpec(
        path="scratch.policy_start", label="Policy Inception / Start Date",
        group="policy_meta", kind=ValueKind.DATE, products=ANY_PRODUCT,
        cues=("policy period - start date", "date and time of policy commencement",
              "policy commencement date", "policy start date", "period of insurance",
              "policy period from", "inception/renewal date", "inception date",
              "effective date", "risk start date", "policy period", "period of cover",
              "from"),
        negative_cues=("date of issue", "date of printing", "next premium due"),
    ),
    FieldSpec(
        path="scratch.policy_end", label="Policy Expiry Date", group="policy_meta",
        kind=ValueKind.DATE, products=ANY_PRODUCT,
        cues=("policy period - end date", "date and time of policy expiry",
              "policy expiry date", "policy end date", "to midnight", "expiry date",
              "valid upto", "valid up to", "policy period to", "to"),
        negative_cues=("date of issue", "next premium due"),
    ),
    FieldSpec(
        path="scratch.first_inception", label="First Policy Inception Date",
        group="policy_meta", kind=ValueKind.DATE, products=ANY_PRODUCT,
        cues=("first policy inception date", "original inception date",
              "first inception date", "policy inception date"),
    ),
    FieldSpec(
        path="scratch.policy_tenure", label="Policy Tenure", group="policy_meta",
        kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("policy tenure", "tenure of policy", "tenure"),
    ),
    FieldSpec(
        path="scratch.family_structure", label="Family Structure", group="policy_meta",
        kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("family structure", "family definition", "family construct",
              "relationship covered", "family size", "family composition"),
    ),
    FieldSpec(
        path="scratch.cover_type", label="Cover Type", group="policy_meta",
        kind=ValueKind.TEXT, products=ANY_PRODUCT,
        cues=("cover type (individual/floater)", "cover type", "type of cover",
              "policy type", "sum insured type"),
    ),
]

ALL_SPECS: List[FieldSpec] = (
    POLICY_SPECS + PERIOD_SPECS + DEMOGRAPHIC_SPECS + ROOM_SPECS + MATERNITY_SPECS
    + WAITING_SPECS + OTHER_BENEFIT_SPECS + INFERTILITY_AMBULANCE_SPECS + BUFFER_SPECS
)

SPECS_BY_PATH: Dict[str, FieldSpec] = {spec.path: spec for spec in ALL_SPECS}


def specs_for_group(group: str) -> List[FieldSpec]:
    return [spec for spec in ALL_SPECS if spec.group == group]


GROUPS: Tuple[str, ...] = (
    "policy_meta", "demographics", "room_and_hospitalisation", "maternity",
    "waiting_periods", "other_benefits", "infertility_and_ambulance", "buffer_and_waivers",
)

#: Benefit groups that only make sense for medical cover. Used to mark fields
#: ``not_applicable`` on a Group Personal Accident schedule instead of inventing values.
GMC_ONLY_GROUPS: Tuple[str, ...] = (
    "room_and_hospitalisation", "maternity", "waiting_periods", "other_benefits",
    "buffer_and_waivers",
)


#: Textual limits that are valid answers in their own right -- a QMS cell reading
#: "No Limit" is correct, whereas leaving it blank would be a miss.
TEXTUAL_LIMITS: Tuple[str, ...] = (
    "no limit", "no capping", "no sub-limit", "no sublimit", "at actuals", "actuals",
    "covered upto sum insured", "upto sum insured", "up to sum insured", "upto si",
    "sum insured", "as per sum insured", "single private room", "single private a/c room",
    "single standard a/c room", "single occupancy room", "any room", "general ward",
    "twin sharing", "shared accommodation", "semi-private room", "semi private room",
    "as per annexure", "not applicable", "waived off", "waived", "no restriction",
)
