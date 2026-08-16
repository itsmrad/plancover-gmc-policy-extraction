"""Thin JSON-mode chat shims for four providers.

Four small HTTP functions rather than four vendor SDKs, and emphatically not LangChain: the
entire requirement is "POST a prompt, get JSON back". Every provider here is used in its
native JSON/structured-output mode so the response does not need coaxing.

``complete_json`` returns ``None`` on any failure. A failed LLM call must degrade the run to
rule-only output, never abort it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ..config import LLMSettings

LOGGER = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    pass


def _loads(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object, tolerating markdown fences or surrounding prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            LOGGER.warning("LLM returned unparseable JSON")
    return None


def _post(url: str, *, headers: Dict[str, str], payload: Dict[str, Any],
          timeout: int) -> Dict[str, Any]:
    import requests

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise LLMError(f"{response.status_code}: {response.text[:300]}")
    return response.json()


def _openai(settings: LLMSettings, system: str, user: str) -> Optional[str]:
    data = _post(
        f"{settings.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.api_key}",
                 "Content-Type": "application/json"},
        payload={
            "model": settings.model,
            "temperature": settings.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        },
        timeout=settings.timeout,
    )
    return data["choices"][0]["message"]["content"]


def _gemini(settings: LLMSettings, system: str, user: str) -> Optional[str]:
    data = _post(
        f"{settings.base_url}/models/{settings.model}:generateContent"
        f"?key={settings.api_key}",
        headers={"Content-Type": "application/json"},
        payload={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": settings.temperature,
                                 "responseMimeType": "application/json"},
        },
        timeout=settings.timeout,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _anthropic(settings: LLMSettings, system: str, user: str) -> Optional[str]:
    data = _post(
        f"{settings.base_url}/v1/messages",
        headers={"x-api-key": settings.api_key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        payload={
            "model": settings.model,
            "max_tokens": 4096,
            "temperature": settings.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=settings.timeout,
    )
    return "".join(block.get("text", "") for block in data.get("content", []))


def _ollama(settings: LLMSettings, system: str, user: str) -> Optional[str]:
    data = _post(
        f"{settings.base_url}/api/chat",
        headers={"Content-Type": "application/json"},
        payload={
            "model": settings.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": settings.temperature},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        },
        timeout=settings.timeout,
    )
    return data["message"]["content"]


_DISPATCH = {
    "openai": _openai,
    "gemini": _gemini,
    "anthropic": _anthropic,
    "ollama": _ollama,
}


def complete_json(settings: LLMSettings, system: str, user: str
                  ) -> Optional[Dict[str, Any]]:
    """Ask the configured provider for a JSON object. ``None`` on any failure."""
    if not settings.enabled:
        return None
    handler = _DISPATCH.get(settings.provider)
    if handler is None:
        return None

    last_error: Optional[Exception] = None
    for attempt in range(settings.max_retries + 1):
        try:
            payload = _loads(handler(settings, system, user) or "")
            if payload is not None:
                return payload
            last_error = LLMError("empty or unparseable response")
        except Exception as exc:  # network, auth, rate limit, schema drift
            last_error = exc
            LOGGER.warning("LLM call failed (attempt %s/%s): %s",
                           attempt + 1, settings.max_retries + 1, exc)
    LOGGER.warning("LLM extraction unavailable, continuing rule-only: %s", last_error)
    return None
