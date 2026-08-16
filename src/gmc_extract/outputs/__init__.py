"""Output artefacts: per-document JSON, flat CSV, run summary and published JSON Schema."""

from .writers import (  # noqa: F401
    build_summary,
    write_flat_csv,
    write_record,
    write_schema,
    write_summary,
)
