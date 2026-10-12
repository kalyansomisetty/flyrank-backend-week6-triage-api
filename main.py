"""FastAPI application entry point for the support-triage API."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from routes.triage import router as triage_router


app = FastAPI(title="FlyRank Support Triage API")


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return field-focused HTTP 400 errors for invalid request input."""
    detail = []
    for error in exc.errors():
        location = error.get("loc", ())
        field_parts = [str(part) for part in location if part not in {"body", "query", "path"}]
        field = ".".join(field_parts) or "request"
        detail.append({"field": field, "message": error["msg"]})

    return JSONResponse(status_code=400, content={"detail": detail})


@app.get("/health")
def health() -> dict[str, str]:
    """Provide a minimal liveness check for local development."""
    return {"status": "ok"}


app.include_router(triage_router)
