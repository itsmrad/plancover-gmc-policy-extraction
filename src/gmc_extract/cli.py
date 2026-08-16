"""Command line interface.

``argparse`` rather than Typer/Click: three subcommands do not justify a dependency.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from . import __version__
from .config import LLMSettings
from .extraction.field_specs import ALL_SPECS, GROUPS
from .outputs import build_summary, write_flat_csv, write_record, write_schema, write_summary
from .pipeline import PipelineOptions, process_path

DEFAULT_INPUT = os.path.join("data", "input")
DEFAULT_OUTPUT = os.path.join("data", "output")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmc-extract",
        description="Extract GMC policy data from PDFs and map it to the QMS schema.",
    )
    parser.add_argument("--version", action="version", version=f"gmc-extract {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="process a PDF or a directory of PDFs")
    run.add_argument("--input", "-i", default=DEFAULT_INPUT,
                     help=f"PDF file or directory (default: {DEFAULT_INPUT})")
    run.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                     help=f"output directory (default: {DEFAULT_OUTPUT})")
    run.add_argument("--llm-provider", default=None,
                     choices=["none", "openai", "gemini", "anthropic", "ollama"],
                     help="override GMC_LLM_PROVIDER; 'none' forces deterministic mode")
    run.add_argument("--no-llm", action="store_true",
                     help="force rule-only extraction even if a provider is configured")
    run.add_argument("--no-tables", action="store_true",
                     help="skip pdfplumber table extraction (faster, slightly less accurate)")
    run.add_argument("--no-ocr", action="store_true",
                     help="never attempt OCR on pages without a text layer")
    run.add_argument("--verbose", "-v", action="store_true")

    schema = subparsers.add_parser("schema", help="write the QMS JSON Schema")
    schema.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    schema.add_argument("--verbose", "-v", action="store_true")

    fields = subparsers.add_parser("fields", help="list every QMS field the system extracts")
    fields.add_argument("--group", default=None, choices=list(GROUPS))
    fields.add_argument("--verbose", "-v", action="store_true")

    return parser


def _run(args: argparse.Namespace) -> int:
    provider = "none" if args.no_llm else args.llm_provider
    settings = LLMSettings.from_env(provider)
    options = PipelineOptions(
        llm=settings,
        extract_tables=not args.no_tables,
        allow_ocr=not args.no_ocr,
    )

    if settings.enabled:
        print(f"LLM layer: {settings.provider} / {settings.model} (hybrid mode)")
    else:
        print("LLM layer: disabled -- running deterministic rule extraction only.")
        print("           Set GMC_LLM_PROVIDER and the matching API key to enable "
              "cross-validation.")

    records = process_path(args.input, options)
    if not records:
        print("No documents were processed successfully.", file=sys.stderr)
        return 1

    for record in records:
        path = write_record(record, args.output)
        meta = record.extraction
        print(
            f"  {record.document.file_name}\n"
            f"    insurer   : {record.insurer.name} "
            f"(confidence {record.insurer.confidence}, score {record.insurer.score})\n"
            f"    tpa       : {record.tpa.name or record.tpa.mode}\n"
            f"    product   : {record.policy.product_type}\n"
            f"    coverage  : {meta.fields_populated}/{meta.fields_total} fields "
            f"({meta.coverage_pct}%), {len(meta.fields_needing_review)} flagged for review\n"
            f"    written   : {path}"
        )

    csv_path = write_flat_csv(records, args.output)
    summary_path = write_summary(records, args.output)
    schema_path = write_schema(args.output)
    summary = build_summary(records)

    print(f"\nDocuments : {summary['documents_processed']}")
    print(f"Insurers  : {', '.join(summary['insurers_detected']) or 'none identified'}")
    print(f"Coverage  : {summary['overall_coverage_pct']}% of QMS fields populated overall")
    print(f"Confidence: {summary['confidence_breakdown']}")
    print(f"Artefacts : {csv_path}\n            {summary_path}\n            {schema_path}")
    return 0


def _schema(args: argparse.Namespace) -> int:
    print(write_schema(args.output))
    return 0


def _fields(args: argparse.Namespace) -> int:
    groups = [args.group] if args.group else list(GROUPS)
    for group in groups:
        specs = [spec for spec in ALL_SPECS if spec.group == group]
        if not specs:
            continue
        print(f"\n{group}  ({len(specs)} fields)")
        for spec in specs:
            print(f"  {spec.path:64} {spec.kind.value:18} cues={len(spec.cues)}")
    print(f"\nTotal: {len(ALL_SPECS)} declared fields")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if args.command == "run":
        return _run(args)
    if args.command == "schema":
        return _schema(args)
    if args.command == "fields":
        return _fields(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
