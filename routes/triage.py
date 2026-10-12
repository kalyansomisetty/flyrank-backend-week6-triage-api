"""HTTP endpoint for support-message triage."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llm.service import (
    InvalidModelOutputError,
    ProviderError,
    ServiceUnavailableError,
    triage_support_message,
)
from schemas import TriageRequest, TriageResponse


router = APIRouter()


@router.post("/triage", response_model=TriageResponse)
def triage_message(request: TriageRequest) -> TriageResponse:
    """Triage a request using the currently configured safe service mode.

    FastAPI validates ``request`` against TriageRequest before this function
    runs. The service selects stub, disabled, or live OpenRouter behavior.
    """
    try:
        return triage_support_message(request)
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidModelOutputError:
        raise HTTPException(
            status_code=422,
            detail="LLM provider returned invalid structured output.",
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
