from pydantic import BaseModel, Field


class LidaEvaluationMetrikBase(BaseModel):
    """Base DTO that contains all """

    score: int = Field(...)
    rationale: str = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
