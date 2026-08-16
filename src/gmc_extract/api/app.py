"""Optional REST endpoint: ``POST /extract`` with a policy PDF, get the QMS record back.

Included because the brief speaks of "the uploaded policy document" and of output that is
"easy to integrate into another application" -- an HTTP boundary demonstrates that directly,
and it reuses the pipeline rather than duplicating any logic.

Kept optional so the core install stays lean::

    pip install "fastapi==0.116.1" "uvicorn==0.35.0" "python-multipart==0.0.20"
    uvicorn gmc_extract.api.app:app --reload

**Security note:** this endpoint has no authentication, no rate limiting and no upload quota
beyond the size cap below. It is a local demonstration surface. Do not expose it on a public
interface without putting authentication, request limits and antivirus scanning of uploads in
front of it.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .. import __version__
from ..config import LLMSettings
from ..extraction.field_specs import ALL_SPECS
from ..outputs.writers import build_summary
from ..pipeline import PipelineOptions, process_file
from ..schema import json_schema

#: Refuse anything larger. A policy PDF is a few MB; this is a denial-of-service guard.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="GMC Policy Extraction -> QMS",
    version=__version__,
    description="Upload a group policy PDF and receive the QMS-mapped record as JSON.",
)


@app.get("/health")
def health() -> dict:
    settings = LLMSettings.from_env()
    return {
        "status": "ok",
        "version": __version__,
        "declared_fields": len(ALL_SPECS),
        "mode": "hybrid" if settings.enabled else "rule_only",
        "llm_provider": settings.provider if settings.enabled else None,
    }


@app.get("/schema")
def schema() -> dict:
    """The QMS JSON Schema this service emits."""
    return json_schema()


@app.post("/extract")
async def extract(
    file: UploadFile = File(..., description="Policy PDF"),
    llm_provider: Optional[str] = Query(
        None, description="Override the LLM provider; 'none' forces deterministic mode."
    ),
) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="only PDF uploads are supported")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    temp_dir = tempfile.mkdtemp(prefix="gmc-extract-")
    temp_path = os.path.join(temp_dir, os.path.basename(file.filename))
    try:
        with open(temp_path, "wb") as handle:
            handle.write(payload)
        options = PipelineOptions(llm=LLMSettings.from_env(llm_provider))
        record = process_file(temp_path, options)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not process PDF: {exc}") from exc
    finally:
        for path in (temp_path,):
            if os.path.exists(path):
                os.unlink(path)
        os.rmdir(temp_dir)

    return {
        "record": record.model_dump(mode="json"),
        "summary": build_summary([record]),
    }
