"""Pydantic schemas for the public triage API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Category = Literal["billing", "bug", "feature", "other"]
Urgency = Literal["low", "normal", "high"]


class TriageRequest(BaseModel):
    """Validated input accepted by the future POST /triage endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Customer support message to classify.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank.")
        return value


class TriageResponse(BaseModel):
    """Only valid model output may be returned by the API."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    urgency: Urgency
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=240)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank.")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_must_be_numeric(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a number.")
        return value
