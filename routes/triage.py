"""HTTP endpoint for support-message triage."""

from __future__ import annotations

from fastapi import APIRouter

from schemas import TriageRequest, TriageResponse


router = APIRouter()


@router.post("/triage", response_model=TriageResponse)
def triage_message(request: TriageRequest) -> TriageResponse:
    """Return a deterministic placeholder until the LLM service is added.

    FastAPI validates ``request`` against TriageRequest before this function
    runs. Stage 3 will replace this placeholder with the LLM service call.
    """
    _ = request
    return TriageResponse(
        category="other",
        urgency="normal",
        confidence=0.0,
        reason="Placeholder response pending LLM integration.",
    )
