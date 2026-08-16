"""Output writers.

Four artefacts, each with a distinct audience:

``<doc>.json``
    The integration contract. One file per document, full evidence and confidence.
``qms_flat.csv``
    One row per document, one column per QMS field -- the spreadsheet shape an operations
    reviewer actually opens first, and what makes "maps cleanly to the QMS schema"
    demonstrable rather than asserted.
``run_summary.json``
    Self-reported coverage, confidence histogram and review flags across the batch. The
    brief lists accuracy analysis as optional; it costs very little and it is the honest way
    to state how well the system did.
``qms_schema.json``
    The JSON Schema generated from the Pydantic models, so the contract is machine-checkable
    by whatever consumes it.
"""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from ..extraction.mapping import iter_qms_fields
from ..schema import QMSPolicyRecord, json_schema

JSON_INDENT = 2


def _slug(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return slug or "document"


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False, default=str)


def write_record(record: QMSPolicyRecord, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{_slug(record.document.file_name)}.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_dump(record.model_dump(mode="json")))
    return path


def write_schema(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "qms_schema.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_dump(json_schema()))
    return path


# --------------------------------------------------------------------------------------
# Flat CSV
# --------------------------------------------------------------------------------------
_HEADER_PREFIX = [
    "file_name", "insurer", "insurer_confidence", "irdai_registration_no", "tpa",
    "tpa_mode", "product_type", "product_name", "policy_number", "policyholder_name",
    "policy_inception_date", "policy_expiry_date", "policy_tenure",
    "sum_insured_tiers", "sum_insured_basis", "aggregate_sum_insured",
    "family_structure", "cover_type", "extraction_mode", "coverage_pct",
    "fields_needing_review",
]


def _field_columns(record: QMSPolicyRecord) -> List[str]:
    return [path for path, _field in iter_qms_fields(record)
            if not path.startswith(("document", "extraction"))]


def _render_field(field) -> str:
    """One CSV cell: the limit if there is one, otherwise the coverage verdict.

    A QMS cell wants "Rs. 75,000" or "Waived Off", not a nested object -- so the display
    string wins when present and the status is the fallback.
    """
    if field.display:
        return field.display
    status = field.status if isinstance(field.status, str) else field.status.value
    return status.replace("_", " ").title()


def write_flat_csv(records: Sequence[QMSPolicyRecord], output_dir: str,
                   file_name: str = "qms_flat.csv") -> Optional[str]:
    if not records:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)

    columns: List[str] = []
    for record in records:
        for column in _field_columns(record):
            if column not in columns:
                columns.append(column)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER_PREFIX + columns)
        for record in records:
            period = record.policy.previous_year_policy_period
            structure = record.structure
            family = structure.family_structure
            fields = dict(iter_qms_fields(record))
            writer.writerow([
                record.document.file_name,
                record.insurer.name or "",
                record.insurer.confidence,
                record.insurer.irdai_registration_no or "",
                record.tpa.name or "",
                record.tpa.mode,
                record.policy.product_type,
                record.policy.product_name.value or "",
                record.policy.policy_number.value or "",
                record.policy.policyholder_name.value or "",
                period.inception_date or "",
                period.expiry_date or "",
                period.tenure_display or "",
                "; ".join(f"{int(tier):d}" for tier in structure.sum_insured_tiers),
                structure.sum_insured_basis or "",
                structure.aggregate_sum_insured.display or "",
                family.display or family.raw_text or "",
                family.cover_type or "",
                record.extraction.mode,
                record.extraction.coverage_pct,
                len(record.extraction.fields_needing_review),
            ] + [
                _render_field(fields[column]) if column in fields else ""
                for column in columns
            ])
    return path


# --------------------------------------------------------------------------------------
# Run summary
# --------------------------------------------------------------------------------------
def build_summary(records: Sequence[QMSPolicyRecord]) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    aggregate_breakdown: Dict[str, int] = {}
    total_fields = 0
    total_populated = 0

    for record in records:
        meta = record.extraction
        total_fields += meta.fields_total
        total_populated += meta.fields_populated
        for key, count in meta.confidence_breakdown.items():
            aggregate_breakdown[key] = aggregate_breakdown.get(key, 0) + count
        documents.append({
            "file_name": record.document.file_name,
            "pages": record.document.page_count,
            "ocr_pages": record.document.ocr_pages,
            "insurer": record.insurer.name,
            "insurer_confidence": record.insurer.confidence,
            "insurer_score": record.insurer.score,
            "tpa": record.tpa.name,
            "tpa_mode": record.tpa.mode,
            "product_type": record.policy.product_type,
            "fields_total": meta.fields_total,
            "fields_populated": meta.fields_populated,
            "coverage_pct": meta.coverage_pct,
            "confidence_breakdown": meta.confidence_breakdown,
            "fields_needing_review": meta.fields_needing_review,
            "warnings": meta.warnings,
        })

    return {
        "documents_processed": len(records),
        "fields_total": total_fields,
        "fields_populated": total_populated,
        "overall_coverage_pct": (round(100.0 * total_populated / total_fields, 1)
                                 if total_fields else 0.0),
        "confidence_breakdown": aggregate_breakdown,
        "insurers_detected": sorted({r.insurer.name for r in records if r.insurer.name}),
        "modes": sorted({r.extraction.mode for r in records}),
        "documents": documents,
    }


def write_summary(records: Sequence[QMSPolicyRecord], output_dir: str) -> Optional[str]:
    if not records:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_dump(build_summary(records)))
    return path
