"""Pipeline orchestration: PDF in, :class:`QMSPolicyRecord` out.

    load -> detect (insurer / TPA / product) -> rule extract -+
                                                             |-> merge -> map -> QMS record
                                            -> LLM extract  -+

The LLM stage is skipped silently when no provider is configured and the run continues in
``rule_only`` mode. Every record states which mode produced it, so an output file is never
ambiguous about how it was generated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from .config import LLMSettings
from .detection import detect_insurer, detect_product_type, detect_tpa
from .extraction.llm_extractor import extract_with_llm
from .extraction.mapping import build_record
from .extraction.merge import merge_all
from .extraction.rule_extractor import extract_fields
from .ingestion import PolicyDocument, discover_pdfs, load_document
from .schema import QMSPolicyRecord

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    llm: LLMSettings
    extract_tables: bool = True
    allow_ocr: bool = True


def process_document(document: PolicyDocument, options: PipelineOptions) -> QMSPolicyRecord:
    """Run detection, both extractors, merge and mapping for one loaded document."""
    started = time.perf_counter()
    warnings: List[str] = []

    insurer = detect_insurer(document)
    product_type, product_scores = detect_product_type(document)
    tpa = detect_tpa(document, insurer.canonical_key)

    if insurer.name is None:
        warnings.append("insurer could not be identified from any known signature")
    if product_type.value == "unknown":
        warnings.append(f"product type could not be classified (scores: {product_scores})")

    rule_results = extract_fields(document, product_type)

    llm_results = {}
    mode = "rule_only"
    if options.llm.enabled:
        # The LLM stage is best-effort by contract. Provider failures are already swallowed
        # inside the provider shim, but this guard also covers schema drift or an unexpected
        # response shape: a document must still produce a rule-only record.
        try:
            llm_results = extract_with_llm(document, product_type, options.llm)
        except Exception as exc:
            LOGGER.warning("LLM stage failed for %s: %s", document.file_name, exc)
            llm_results = {}
            warnings.append(f"LLM extraction failed ({exc}); output is rule-only")
        if llm_results:
            mode = "hybrid"
        elif not any(w.startswith("LLM extraction failed") for w in warnings):
            warnings.append(
                "LLM extraction was requested but returned nothing usable; "
                "output is rule-only"
            )

    merged = merge_all(rule_results, llm_results)

    return build_record(
        document,
        insurer=insurer,
        tpa=tpa,
        product_type=product_type,
        merged=merged,
        mode=mode,
        llm_provider=options.llm.provider if options.llm.enabled else None,
        llm_model=options.llm.model if options.llm.enabled else None,
        duration_seconds=time.perf_counter() - started,
        extra_warnings=warnings,
    )


def process_file(path: str, options: PipelineOptions) -> QMSPolicyRecord:
    document = load_document(path, extract_tables=options.extract_tables,
                             allow_ocr=options.allow_ocr)
    return process_document(document, options)


def process_path(input_path: str, options: PipelineOptions,
                 on_result: Optional[callable] = None) -> List[QMSPolicyRecord]:
    """Process a single PDF or every PDF in a directory."""
    paths = discover_pdfs(input_path)
    if not paths:
        raise FileNotFoundError(f"no PDF files found at {input_path}")

    records: List[QMSPolicyRecord] = []
    for path in paths:
        LOGGER.info("processing %s", path)
        try:
            record = process_file(path, options)
        except Exception as exc:  # one bad document must not abort a batch
            LOGGER.error("failed to process %s: %s", path, exc, exc_info=True)
            continue
        records.append(record)
        if on_result:
            on_result(record)
    return records
