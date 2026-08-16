"""Assemble merged fields into a :class:`QMSPolicyRecord`, then derive what can be derived.

Two responsibilities beyond plain assignment:

**Derivation.** Some QMS cells are not written anywhere in the document but follow
unambiguously from cells that are. Total lives is the clearest case: the Care Health
schedules print "Primary Insured Members 115" and "Dependents 32" with the total in an
unlabelled row, so the total is computed and marked ``source: "derived"`` rather than left
blank or guessed.

**Explicit non-applicability.** On a Group Personal Accident schedule, maternity is not
"missing data" -- it cannot apply. Those fields are set to ``not_applicable`` so a QMS can
tell "we could not find it" apart from "it does not exist here".
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel

from ..ingestion import PolicyDocument
from ..schema import (
    Confidence,
    Demographics,
    DocumentMeta,
    ExtractionMeta,
    ExtractionSource,
    FieldStatus,
    InsurerDetection,
    PolicyPeriod,
    ProductType,
    QMSField,
    QMSPolicyRecord,
    TPADetection,
    ValueUnit,
)
from .field_specs import ALL_SPECS, SPECS_BY_PATH
from .parsers import format_inr, parse_years, tenure_between
from .structural import build_family_structure, extract_product_name, extract_sum_insured

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _set_by_path(record: QMSPolicyRecord, path: str, value: QMSField) -> bool:
    parts = path.split(".")
    target = record
    for part in parts[:-1]:
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        return False
    setattr(target, parts[-1], value)
    return True


def iter_qms_fields(model: BaseModel, prefix: str = "") -> Iterator[Tuple[str, QMSField]]:
    """Walk a record and yield every ``(dotted_path, QMSField)``."""
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, QMSField):
            yield path, value
        elif isinstance(value, BaseModel):
            yield from iter_qms_fields(value, path)


def _as_date(value) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def _build_period(fields: Dict[str, QMSField]) -> PolicyPeriod:
    start_field = fields.get("scratch.policy_start")
    end_field = fields.get("scratch.policy_end")
    first_field = fields.get("scratch.first_inception")
    tenure_field = fields.get("scratch.policy_tenure")

    start = _as_date(start_field.value) if start_field else None
    end = _as_date(end_field.value) if end_field else None
    days, months, derived_display = tenure_between(start, end)

    display = None
    if tenure_field and isinstance(tenure_field.value, str):
        display = tenure_field.value
        if days is None:
            years = parse_years(display)
            if years:
                months = years[0] * 12
    display = display or derived_display

    sources = [f.source for f in (start_field, end_field) if f]
    confidence = Confidence.HIGH if start and end else (
        Confidence.MEDIUM if start or end else Confidence.LOW)

    return PolicyPeriod(
        inception_date=start,
        expiry_date=end,
        inception_date_raw=start_field.raw_text if start_field else None,
        expiry_date_raw=end_field.raw_text if end_field else None,
        tenure_display=display,
        tenure_days=days,
        tenure_months=months,
        first_policy_inception_date=_as_date(first_field.value) if first_field else None,
        source=sources[0] if sources else ExtractionSource.NONE,
        confidence=confidence,
        page=start_field.page if start_field else None,
    )


def _derive_total_lives(demographics: Demographics) -> Optional[QMSField]:
    """Total lives = employees + dependents, when the document leaves the total unlabelled.

    The Care Health schedules print the total in a bare "Total  147" row with no cue a
    label-based extractor can latch onto, but both addends are cleanly labelled.
    """
    employees = demographics.employees
    dependents = demographics.dependents_total
    if not isinstance(employees.value, (int, float)):
        return None
    if not isinstance(dependents.value, (int, float)):
        return None
    total = float(employees.value) + float(dependents.value)
    return QMSField(
        status=FieldStatus.PRESENT,
        value=total,
        unit=ValueUnit.COUNT,
        display=str(int(total)),
        raw_text=f"derived: employees ({employees.value:g}) + dependents "
                 f"({dependents.value:g})",
        page=employees.page,
        source=ExtractionSource.DERIVED,
        confidence=Confidence.MEDIUM,
        notes="document does not label a total-lives field; computed from its components",
    )


_METRO_FALLBACK_NOTE = ("document states a single limit without a metro / non-metro split; "
                        "the same limit is reported for both")


def _apply_maternity_fallbacks(record: QMSPolicyRecord, fields: Dict[str, QMSField]) -> None:
    """Populate metro/non-metro cells from an undifferentiated limit.

    The QMS schema has separate metro and non-metro columns, but none of the sample insurers
    differentiate. Copying the flat limit into both -- with a note saying so -- is more useful
    to an integrator than leaving two columns blank, and more honest than pretending the
    document made a distinction it did not.
    """
    pairs = (
        ("scratch.normal_delivery_limit",
         ("normal_delivery_metro", "normal_delivery_non_metro")),
        ("scratch.c_section_limit", ("c_section_metro", "c_section_non_metro")),
    )
    for scratch_path, targets in pairs:
        flat = fields.get(scratch_path)
        if flat is None or not flat.is_populated:
            continue
        for target in targets:
            existing: QMSField = getattr(record.benefits.maternity, target)
            if existing.is_populated:
                continue
            setattr(record.benefits.maternity, target, QMSField(
                status=flat.status,
                value=flat.value,
                unit=flat.unit,
                basis=flat.basis,
                display=flat.display,
                raw_text=flat.raw_text,
                page=flat.page,
                source=flat.source,
                confidence=(Confidence.MEDIUM if flat.confidence is Confidence.HIGH
                            else flat.confidence),
                needs_review=flat.needs_review,
                notes=(f"{flat.notes}; {_METRO_FALLBACK_NOTE}" if flat.notes
                       else _METRO_FALLBACK_NOTE),
            ))


_GENERAL_COPAY_MARKERS = ("all cases", "all claims", "every claim", "shall apply",
                          "will apply", "each and every claim", "all admissible claims")


def _review_specific_copay(record: QMSPolicyRecord) -> None:
    """Flag a co-payment that is item-specific rather than policy-wide.

    ``1.Policy Copy.pdf`` states only "50% co-pay for Bio-absorbable Stent/Toric lens/Multi
    Focal lens" -- a procedure-specific co-pay, not a policy-wide one. Reporting 50% as *the*
    co-payment without comment would overstate it.
    """
    field = record.benefits.buffer_and_waivers.co_payment
    if not field.is_populated or not field.raw_text:
        return
    lowered = field.raw_text.lower()
    if not any(marker in lowered for marker in _GENERAL_COPAY_MARKERS):
        field.needs_review = True
        note = ("co-payment appears item/procedure-specific rather than policy-wide; "
                "verify scope")
        field.notes = f"{field.notes}; {note}" if field.notes else note


def _derive_corporate_buffer(record: QMSPolicyRecord) -> None:
    """A stated buffer limit means the buffer exists, whatever the surrounding prose says."""
    buffer_field = record.benefits.buffer_and_waivers.corporate_buffer
    limit = record.benefits.buffer_and_waivers.corporate_buffer_limit
    if limit.status is FieldStatus.PRESENT and isinstance(limit.value, (int, float)):
        if buffer_field.status in (FieldStatus.NOT_FOUND, FieldStatus.APPLIED,
                                   FieldStatus.NOT_SPECIFIED):
            note = f"inferred from a stated corporate buffer limit of {format_inr(limit.value)}"
            record.benefits.buffer_and_waivers.corporate_buffer = QMSField(
                status=FieldStatus.COVERED,
                display="Covered",
                raw_text=limit.raw_text,
                page=limit.page,
                source=ExtractionSource.DERIVED,
                confidence=Confidence.MEDIUM,
                notes=note,
            )


def _mark_not_applicable(record: QMSPolicyRecord, product_type: ProductType) -> List[str]:
    """Set fields that cannot apply to this product to ``not_applicable``."""
    warnings: List[str] = []
    if product_type is ProductType.GMC:
        return warnings
    marked = 0
    for spec in ALL_SPECS:
        if product_type in spec.products or not spec.path.startswith("benefits."):
            continue
        field: QMSField = record
        try:
            parts = spec.path.split(".")
            parent = record
            for part in parts[:-1]:
                parent = getattr(parent, part)
            field = getattr(parent, parts[-1])
        except AttributeError:
            continue
        if not field.is_populated:
            _set_by_path(record, spec.path, QMSField.not_applicable(
                f"not applicable to a {product_type.value.replace('_', ' ')} policy"))
            marked += 1
    if marked:
        warnings.append(
            f"document classified as {product_type.value}, not group medical cover; "
            f"{marked} medical-benefit fields marked not_applicable rather than extracted"
        )
    return warnings


def build_record(
    document: PolicyDocument,
    insurer: InsurerDetection,
    tpa: TPADetection,
    product_type: ProductType,
    merged: Dict[str, QMSField],
    *,
    mode: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    extra_warnings: Optional[List[str]] = None,
) -> QMSPolicyRecord:
    """Assemble the final QMS record."""
    record = QMSPolicyRecord(
        document=DocumentMeta(
            file_name=document.file_name,
            file_sha256=document.sha256,
            page_count=document.page_count,
            ocr_pages=document.ocr_pages,
            characters_extracted=len(document.text),
            has_text_layer=document.has_text_layer,
        ),
        insurer=insurer,
        tpa=tpa,
    )
    record.policy.product_type = product_type

    for path, field in merged.items():
        if path.startswith("scratch."):
            continue
        _set_by_path(record, path, field)

    # --- policy period / product name ---------------------------------------------------
    record.policy.previous_year_policy_period = _build_period(merged)
    product_name, product_page = extract_product_name(document)
    if product_name:
        record.policy.product_name = QMSField(
            status=FieldStatus.PRESENT, value=product_name, unit=ValueUnit.TEXT,
            display=product_name, raw_text=product_name, page=product_page,
            source=ExtractionSource.RULE, confidence=Confidence.MEDIUM,
        )

    # --- structure ----------------------------------------------------------------------
    family_raw = merged.get("scratch.family_structure")
    cover_type = merged.get("scratch.cover_type")
    record.structure.family_structure = build_family_structure(
        family_raw.value if family_raw and isinstance(family_raw.value, str) else None,
        family_raw.page if family_raw else None,
        cover_type.value if cover_type and isinstance(cover_type.value, str) else None,
    )

    tiers, basis, aggregate, evidence, si_page = extract_sum_insured(document)
    record.structure.sum_insured_tiers = tiers
    record.structure.sum_insured_basis = basis
    record.structure.sum_insured_evidence = evidence
    record.structure.sum_insured_page = si_page
    record.structure.sum_insured_source = (ExtractionSource.RULE if tiers or aggregate
                                           else ExtractionSource.NONE)
    if aggregate is not None:
        record.structure.aggregate_sum_insured = QMSField(
            status=FieldStatus.PRESENT, value=aggregate, unit=ValueUnit.INR,
            display=format_inr(aggregate), raw_text=evidence, page=si_page,
            source=ExtractionSource.RULE, confidence=Confidence.HIGH,
        )

    # --- derivations --------------------------------------------------------------------
    if not record.demographics.total_lives.is_populated:
        derived = _derive_total_lives(record.demographics)
        if derived:
            record.demographics.total_lives = derived

    _apply_maternity_fallbacks(record, merged)
    _derive_corporate_buffer(record)
    _review_specific_copay(record)
    warnings = list(document.warnings) + (extra_warnings or [])
    warnings.extend(_mark_not_applicable(record, product_type))

    if tpa.mode.value == "unknown" and product_type is not ProductType.GMC:
        warnings.append(
            "no TPA identified; group personal accident policies are commonly administered "
            "without a third-party administrator"
        )

    # --- extraction metadata ------------------------------------------------------------
    all_fields = list(iter_qms_fields(record))
    countable = [(path, field) for path, field in all_fields
                 if not path.startswith(("document", "extraction"))]
    # Coverage measures "of the fields that *could* apply to this document, how many were
    # filled". Counting maternity against an accident policy would understate the system.
    applicable = [(path, field) for path, field in countable
                  if field.status != FieldStatus.NOT_APPLICABLE]
    populated = [field for _path, field in applicable if field.is_populated]
    breakdown: Dict[str, int] = {level.value: 0 for level in Confidence}
    for _path, field in applicable:
        if field.is_populated:
            key = field.confidence if isinstance(field.confidence, str) else field.confidence.value
            breakdown[key] = breakdown.get(key, 0) + 1

    record.extraction = ExtractionMeta(
        mode=mode,
        llm_provider=llm_provider,
        llm_model=llm_model,
        duration_seconds=round(duration_seconds, 3) if duration_seconds else None,
        fields_total=len(applicable),
        fields_populated=len(populated),
        coverage_pct=round(100.0 * len(populated) / len(applicable), 1) if applicable else 0.0,
        confidence_breakdown=breakdown,
        fields_needing_review=sorted(path for path, field in countable if field.needs_review),
        warnings=warnings,
    )
    record.extraction.warnings.append(
        f"{len(countable) - len(applicable)} field(s) marked not_applicable and excluded "
        "from the coverage denominator"
        if len(countable) != len(applicable) else
        "all declared QMS fields are applicable to this product type"
    )
    return record
