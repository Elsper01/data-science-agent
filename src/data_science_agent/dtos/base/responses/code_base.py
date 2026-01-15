from typing import Optional, Any

from pydantic import BaseModel, Field

from data_science_agent.dtos.base.responses.judge_verdict_base import JudgeVerdictBase


class CodeBase(BaseModel):
    """Base DTO for code responses (natural language neutral)."""

    code: str = Field(...)
