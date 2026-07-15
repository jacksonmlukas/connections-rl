"""Pydantic request/response models for the serving API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Arm(str, Enum):
    base = "base"
    grpo = "grpo"


class SolveRequest(BaseModel):
    words: list[str] = Field(..., description="the 16 board words")
    arm: Arm = Arm.grpo
    temperature: float = 0.0
    max_tokens: int = 1024

    @field_validator("words")
    @classmethod
    def _sixteen_unique(cls, v: list[str]) -> list[str]:
        if len(v) != 16 or len({w.strip().upper() for w in v}) != 16:
            raise ValueError("exactly 16 unique words required")
        return v


class SolveResponse(BaseModel):
    arm: Arm
    model: str
    groups: list[list[str]] | None  # None if the completion was unparseable
    valid: bool
    raw_completion: str
    latency_ms: float


class CompareResponse(BaseModel):
    base: SolveResponse
    grpo: SolveResponse
    agree: bool  # both valid and identical grouping


class HealthResponse(BaseModel):
    status: str
    base_model: str
    grpo_model: str
    uptime_s: float
