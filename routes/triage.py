"""HTTP endpoint for support-message triage."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llm.service import ServiceUnavailableError, triage_support_message
from schemas import TriageRequest, TriageResponse


router = APIRouter()


@router.post("/triage", response_model=TriageResponse)
def triage_message(request: TriageRequest) -> TriageResponse:
    """Triage a request using the currently configured safe service mode.

    FastAPI validates ``request`` against TriageRequest before this function
    runs. Stage 3 supports a deterministic stub; live integration is deferred.
    """
    try:
        return triage_support_message(request)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
