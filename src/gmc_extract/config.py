"""Runtime configuration.

The LLM layer is strictly optional. With no provider configured the pipeline runs the
deterministic extractor alone and says so in the output (``mode: "rule_only"``). This is a
deliberate design constraint, not a convenience: a submission whose sample output cannot be
regenerated without someone's paid API key is not reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:  # python-dotenv is a declared dependency, but never let config import fail.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROVIDER_NONE = "none"

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.1",
}
_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # local, no key
}


@dataclass
class LLMSettings:
    provider: str = PROVIDER_NONE
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    timeout: int = 90
    max_retries: int = 2
    #: Snippets of document text passed per field group.
    snippets_per_group: int = 6

    @property
    def enabled(self) -> bool:
        if self.provider in (PROVIDER_NONE, "", None):
            return False
        if _KEY_ENV.get(self.provider) and not self.api_key:
            return False
        return True

    @classmethod
    def from_env(cls, provider_override: Optional[str] = None) -> "LLMSettings":
        provider = (provider_override or os.getenv("GMC_LLM_PROVIDER", PROVIDER_NONE)
                    or PROVIDER_NONE).strip().lower()
        if provider not in _DEFAULT_MODELS:
            provider = PROVIDER_NONE

        key_env = _KEY_ENV.get(provider)
        api_key = os.getenv(key_env, "").strip() if key_env else ""

        if provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            model = os.getenv("OPENAI_MODEL") or _DEFAULT_MODELS[provider]
        elif provider == "gemini":
            base_url = os.getenv(
                "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
            ).rstrip("/")
            model = os.getenv("GEMINI_MODEL") or _DEFAULT_MODELS[provider]
        elif provider == "anthropic":
            base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
            model = os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODELS[provider]
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            model = os.getenv("OLLAMA_MODEL") or _DEFAULT_MODELS[provider]
        else:
            base_url, model = "", ""

        return cls(provider=provider, model=model, api_key=api_key, base_url=base_url)
