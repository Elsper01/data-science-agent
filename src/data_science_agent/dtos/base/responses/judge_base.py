from pydantic import BaseModel, Field
from . import JudgeVerdictBase


class JudgeBase(BaseModel):
    """Base DTO that contains all verdicts of the judge regarding one script."""

    verdicts: list[JudgeVerdictBase] = Field(...)
    needs_regeneration: bool = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
