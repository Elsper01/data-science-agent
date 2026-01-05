from typing import Optional

from pydantic import BaseModel, Field


class GoalBase(BaseModel):
    """Base DTO for visualization goals."""

    index: int = Field(...)
    question: str = Field(...)
    visualization: str = Field(...)
    rationale: str = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
