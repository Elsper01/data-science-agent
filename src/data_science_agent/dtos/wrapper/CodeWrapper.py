from typing import Optional, Any

from pydantic import BaseModel, Field


class CodeWrapper(BaseModel):
    """Contains all important information about code. We use this wrapper to not always force the LLM to parse / create such a complex object."""

    code: str = Field(...)
    std_out: Optional[str] = Field(...)
    std_err: Optional[str] = Field(...)
    needs_regeneration: Optional[list[bool]] = Field(..., default_factory=list)
    regeneration_attempts: Optional[int] = Field(...)
    refactoring_attempts: Optional[int] = Field(...)
    judge_result: Optional[Any] = Field(...)
