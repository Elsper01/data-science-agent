from typing import Optional, Any

from pydantic import BaseModel, Field

from data_science_agent.dtos.base.responses.judge_verdict_base import JudgeVerdictBase


class CodeBase(BaseModel):
    """Base DTO for code responses (natural language neutral)."""

    code: str = Field(...)
    std_out: Optional[str] = Field(...)
    std_err: Optional[str] = Field(...)
    needs_regeneration: Optional[bool] = Field(...)
    regeneration_attempts: Optional[int] = Field(...)
    judge_result: Optional[Any] = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
