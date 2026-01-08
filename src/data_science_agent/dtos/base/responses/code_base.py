from typing import Optional

from pydantic import BaseModel, Field


class CodeBase(BaseModel):
    """Base DTO for code responses (natural language neutral)."""

    code: str = Field(...)
    std_out: Optional[str] = Field(...)
    std_err: Optional[str] = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
