"""Canonical QMS schema.

This module is the integration contract. Every document processed by the pipeline
produces exactly one :class:`QMSPolicyRecord` with an identical key set, regardless of
insurer, product or how much of the document could actually be understood. Fields that
were not found are emitted as ``status="not_found"`` rather than dropped, so a downstream
QMS never has to deal with a missing key.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------
class FieldStatus(str, Enum):
    """Coverage verdict for a single QMS field."""

    COVERED = "covered"
    NOT_COVERED = "not_covered"
    WAIVED_OFF = "waived_off"
    APPLIED = "applied"
    #: Informational (non-coverage) field that was successfully read -- policy number,
    #: premium amount, headcount. "covered" would be meaningless for these.
    PRESENT = "present"
    #: Present in the document but the document itself says the value is NA / nil.
    NOT_SPECIFIED = "not_specified"
    #: The pipeline could not locate the field in this document.
    NOT_FOUND = "not_found"
    #: The field cannot apply to this product (e.g. maternity on an accident policy).
    NOT_APPLICABLE = "not_applicable"


class ValueUnit(str, Enum):
    """Unit attached to :attr:`QMSField.value`, so the number is machine-interpretable."""

    INR = "INR"
    PERCENT_OF_SUM_INSURED = "percent_of_sum_insured"
    PERCENT = "percent"
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    COUNT = "count"
    TEXT = "text"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractionSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    RULE_AND_LLM = "rule+llm"
    DERIVED = "derived"
    NONE = "none"


class ProductType(str, Enum):
    """Product classification. Drives which benefit groups are applicable."""

    GMC = "group_medical_cover"
    GPA = "group_personal_accident"
    GTL = "group_term_life"
    OPD = "group_opd"
    UNKNOWN = "unknown"


class TPAMode(str, Enum):
    EXTERNAL = "external_tpa"
    IN_HOUSE = "in_house_insurer_administered"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------
# The atomic unit of output
# --------------------------------------------------------------------------------------
class QMSField(BaseModel):
    """One QMS cell, carrying both a machine value and the human evidence behind it.

    ``value`` + ``unit`` + ``basis`` is what the QMS consumes; ``raw_text`` + ``page`` is
    what a human auditor checks. Emitting only one of the two would make the output either
    unverifiable or unusable.
    """

    model_config = ConfigDict(use_enum_values=True)

    status: FieldStatus = FieldStatus.NOT_FOUND
    value: Optional[Union[float, str]] = None
    unit: Optional[ValueUnit] = None
    #: Qualifier for the value, e.g. "per day", "per claim", "per hospitalization".
    basis: Optional[str] = None
    #: Human-readable rendering, e.g. "2% of sum insured per day".
    display: Optional[str] = None
    #: Verbatim source sentence/cell the value came from. Provenance for audit.
    raw_text: Optional[str] = None
    page: Optional[int] = None
    source: ExtractionSource = ExtractionSource.NONE
    confidence: Confidence = Confidence.LOW
    #: True when rule and LLM extractors disagreed, or a value looks ambiguous.
    needs_review: bool = False
    #: The losing candidate when the two extractors disagreed.
    alternate: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def missing(cls, note: Optional[str] = None) -> "QMSField":
        return cls(status=FieldStatus.NOT_FOUND, notes=note)

    @classmethod
    def not_applicable(cls, note: Optional[str] = None) -> "QMSField":
        return cls(status=FieldStatus.NOT_APPLICABLE, confidence=Confidence.HIGH,
                   source=ExtractionSource.DERIVED, notes=note)

    @property
    def is_populated(self) -> bool:
        return self.status not in (FieldStatus.NOT_FOUND, FieldStatus.NOT_APPLICABLE)


# --------------------------------------------------------------------------------------
# Detection blocks
# --------------------------------------------------------------------------------------
class DetectionEvidence(BaseModel):
    """Why a detector reached its conclusion. Makes detection auditable."""

    signal: str
    matched_text: str
    page: Optional[int] = None
    weight: float = 0.0


class InsurerDetection(BaseModel):
    name: Optional[str] = None
    canonical_key: Optional[str] = None
    irdai_registration_no: Optional[str] = None
    cin: Optional[str] = None
    confidence: Confidence = Confidence.LOW
    score: float = 0.0
    evidence: List[DetectionEvidence] = Field(default_factory=list)
    runner_up: Optional[str] = None


class TPADetection(BaseModel):
    name: Optional[str] = None
    mode: TPAMode = TPAMode.UNKNOWN
    confidence: Confidence = Confidence.LOW
    evidence: List[DetectionEvidence] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Policy detail blocks
# --------------------------------------------------------------------------------------
class PolicyPeriod(BaseModel):
    """The policy period stated on the document.

    Assumption (documented in README): the sample documents are the *expiring* policies
    being renewed, so the period printed on them is the "previous year" period the QMS
    asks for.
    """

    inception_date: Optional[date] = None
    expiry_date: Optional[date] = None
    inception_date_raw: Optional[str] = None
    expiry_date_raw: Optional[str] = None
    tenure_display: Optional[str] = None
    tenure_days: Optional[int] = None
    tenure_months: Optional[int] = None
    first_policy_inception_date: Optional[date] = None
    source: ExtractionSource = ExtractionSource.NONE
    confidence: Confidence = Confidence.LOW
    page: Optional[int] = None


class PremiumDetails(BaseModel):
    net_premium: QMSField = Field(default_factory=QMSField.missing)
    gross_premium: QMSField = Field(default_factory=QMSField.missing)
    tax_amount: QMSField = Field(default_factory=QMSField.missing)
    payment_mode: QMSField = Field(default_factory=QMSField.missing)


class PolicyDetails(BaseModel):
    policy_number: QMSField = Field(default_factory=QMSField.missing)
    policyholder_name: QMSField = Field(default_factory=QMSField.missing)
    product_name: QMSField = Field(default_factory=QMSField.missing)
    product_type: ProductType = ProductType.UNKNOWN
    previous_year_policy_period: PolicyPeriod = Field(default_factory=PolicyPeriod)
    previous_year_premium: PremiumDetails = Field(default_factory=PremiumDetails)


class FamilyStructure(BaseModel):
    raw_text: Optional[str] = None
    display: Optional[str] = None
    employee: bool = False
    spouse: bool = False
    children: bool = False
    parents: bool = False
    parents_in_law: bool = False
    max_children: Optional[int] = None
    cover_type: Optional[str] = None  # floater | individual | ...
    source: ExtractionSource = ExtractionSource.NONE
    confidence: Confidence = Confidence.LOW
    page: Optional[int] = None
    notes: Optional[str] = None


class PolicyStructure(BaseModel):
    family_structure: FamilyStructure = Field(default_factory=FamilyStructure)
    #: Distinct sum insured tiers offered, ascending, in INR.
    sum_insured_tiers: List[float] = Field(default_factory=list)
    sum_insured_basis: Optional[str] = None  # graded | flat | unknown
    aggregate_sum_insured: QMSField = Field(default_factory=QMSField.missing)
    sum_insured_evidence: Optional[str] = None
    sum_insured_page: Optional[int] = None
    sum_insured_source: ExtractionSource = ExtractionSource.NONE


class Demographics(BaseModel):
    employees: QMSField = Field(default_factory=QMSField.missing)
    spouses: QMSField = Field(default_factory=QMSField.missing)
    children: QMSField = Field(default_factory=QMSField.missing)
    parents: QMSField = Field(default_factory=QMSField.missing)
    parents_in_law: QMSField = Field(default_factory=QMSField.missing)
    dependents_total: QMSField = Field(default_factory=QMSField.missing)
    total_lives: QMSField = Field(default_factory=QMSField.missing)


# --------------------------------------------------------------------------------------
# Benefit groups -- mirror the section order of the assignment brief
# --------------------------------------------------------------------------------------
class RoomAndHospitalisation(BaseModel):
    room_rent: QMSField = Field(default_factory=QMSField.missing)
    icu_charges: QMSField = Field(default_factory=QMSField.missing)
    pre_hospitalization: QMSField = Field(default_factory=QMSField.missing)
    post_hospitalization: QMSField = Field(default_factory=QMSField.missing)
    room_rent_proportionate_deduction: QMSField = Field(default_factory=QMSField.missing)


class Maternity(BaseModel):
    nine_month_waiting_period: QMSField = Field(default_factory=QMSField.missing)
    baby_day_one_cover: QMSField = Field(default_factory=QMSField.missing)
    vaccination_cover: QMSField = Field(default_factory=QMSField.missing)
    normal_delivery_metro: QMSField = Field(default_factory=QMSField.missing)
    normal_delivery_non_metro: QMSField = Field(default_factory=QMSField.missing)
    c_section_metro: QMSField = Field(default_factory=QMSField.missing)
    c_section_non_metro: QMSField = Field(default_factory=QMSField.missing)
    pre_post_natal_expenses: QMSField = Field(default_factory=QMSField.missing)
    maternity_child_limit: QMSField = Field(default_factory=QMSField.missing)


class WaitingPeriods(BaseModel):
    thirty_day_waiting_period: QMSField = Field(default_factory=QMSField.missing)
    first_and_second_year_waiting_period: QMSField = Field(default_factory=QMSField.missing)
    pre_existing_diseases: QMSField = Field(default_factory=QMSField.missing)


class OtherBenefits(BaseModel):
    day_care_expenses: QMSField = Field(default_factory=QMSField.missing)
    opd_benefit: QMSField = Field(default_factory=QMSField.missing)
    teleconsultation: QMSField = Field(default_factory=QMSField.missing)
    pharmacy_discount: QMSField = Field(default_factory=QMSField.missing)
    domiciliary_hospitalization: QMSField = Field(default_factory=QMSField.missing)
    annual_health_checkup: QMSField = Field(default_factory=QMSField.missing)
    modern_treatment: QMSField = Field(default_factory=QMSField.missing)
    bariatric_treatment: QMSField = Field(default_factory=QMSField.missing)
    psychiatric_treatment: QMSField = Field(default_factory=QMSField.missing)
    ayush_treatment: QMSField = Field(default_factory=QMSField.missing)
    lgbtq_coverage: QMSField = Field(default_factory=QMSField.missing)
    live_in_partner_coverage: QMSField = Field(default_factory=QMSField.missing)
    organ_donor_expenses: QMSField = Field(default_factory=QMSField.missing)


class InfertilityAndAmbulance(BaseModel):
    infertility_treatment: QMSField = Field(default_factory=QMSField.missing)
    surrogacy: QMSField = Field(default_factory=QMSField.missing)
    ambulance_charges: QMSField = Field(default_factory=QMSField.missing)
    air_ambulance_charges: QMSField = Field(default_factory=QMSField.missing)


class BufferAndWaivers(BaseModel):
    corporate_buffer: QMSField = Field(default_factory=QMSField.missing)
    corporate_buffer_limit: QMSField = Field(default_factory=QMSField.missing)
    corporate_buffer_per_family: QMSField = Field(default_factory=QMSField.missing)
    disease_wise_capping: QMSField = Field(default_factory=QMSField.missing)
    co_payment: QMSField = Field(default_factory=QMSField.missing)


class BenefitSchedule(BaseModel):
    room_and_hospitalisation: RoomAndHospitalisation = Field(default_factory=RoomAndHospitalisation)
    maternity: Maternity = Field(default_factory=Maternity)
    waiting_periods: WaitingPeriods = Field(default_factory=WaitingPeriods)
    other_benefits: OtherBenefits = Field(default_factory=OtherBenefits)
    infertility_and_ambulance: InfertilityAndAmbulance = Field(default_factory=InfertilityAndAmbulance)
    buffer_and_waivers: BufferAndWaivers = Field(default_factory=BufferAndWaivers)


# --------------------------------------------------------------------------------------
# Envelope
# --------------------------------------------------------------------------------------
class DocumentMeta(BaseModel):
    file_name: str
    file_sha256: str
    page_count: int
    #: Pages that had no usable text layer and were routed through OCR.
    ocr_pages: List[int] = Field(default_factory=list)
    characters_extracted: int = 0
    has_text_layer: bool = True


class ExtractionMeta(BaseModel):
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = "1.0.0"
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    mode: str = "rule_only"  # rule_only | hybrid
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    duration_seconds: Optional[float] = None
    fields_total: int = 0
    fields_populated: int = 0
    coverage_pct: float = 0.0
    confidence_breakdown: Dict[str, int] = Field(default_factory=dict)
    fields_needing_review: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class QMSPolicyRecord(BaseModel):
    """One policy document, mapped into the QMS schema."""

    document: DocumentMeta
    insurer: InsurerDetection = Field(default_factory=InsurerDetection)
    tpa: TPADetection = Field(default_factory=TPADetection)
    policy: PolicyDetails = Field(default_factory=PolicyDetails)
    structure: PolicyStructure = Field(default_factory=PolicyStructure)
    demographics: Demographics = Field(default_factory=Demographics)
    benefits: BenefitSchedule = Field(default_factory=BenefitSchedule)
    extraction: ExtractionMeta = Field(default_factory=ExtractionMeta)


def json_schema() -> dict:
    """Published JSON Schema for the QMS contract."""
    return QMSPolicyRecord.model_json_schema()
