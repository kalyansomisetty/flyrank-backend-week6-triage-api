"""Stage 3 service decisions for support-message triage.

This module deliberately has no OpenAI or network integration yet.
"""

from __future__ import annotations

from config import get_settings
from schemas import TriageRequest, TriageResponse


class ServiceUnavailableError(RuntimeError):
    """Raised when the current LLM mode cannot produce a triage result."""


def triage_support_message(request: TriageRequest) -> TriageResponse:
    """Choose the safe Stage 3 behavior for a validated support message.

    Stub mode has priority so local deterministic testing remains available
    even when the live LLM kill switch is enabled. Settings are read without
    the API key because Stage 3 never creates or calls an LLM client.
    """
    _ = request
    settings = get_settings(include_api_key=False)

    if settings.llm_stub:
        return TriageResponse(
            category="other",
            urgency="normal",
            confidence=1.0,
            reason="Deterministic stub response for local testing.",
        )

    if not settings.llm_enabled:
        raise ServiceUnavailableError("LLM service is disabled.")

    raise ServiceUnavailableError("Live LLM service is not yet implemented.")
