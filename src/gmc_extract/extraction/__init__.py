"""Field extraction: declarative specs, rule engine, retrieval, LLM layer and merge."""

from .field_specs import ALL_SPECS, GROUPS, FieldSpec, ValueKind  # noqa: F401
from .rule_extractor import Candidate, extract_fields  # noqa: F401
