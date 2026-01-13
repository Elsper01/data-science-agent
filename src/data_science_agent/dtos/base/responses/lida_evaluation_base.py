from pydantic import BaseModel, Field


class LidaEvaluationBase(BaseModel):
    """Base DTO to evaluate a visualization by different criteria by a LLM."""

    dimension: str = Field(...)
    score: int = Field(...)
    rationale: str = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
