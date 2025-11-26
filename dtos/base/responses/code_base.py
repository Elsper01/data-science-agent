from pydantic import BaseModel, Field


class CodeBase(BaseModel):
    """Base DTO for code responses (language neutral)."""

    explanation: str = Field(...)
    code: str = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        allow_population_by_field_name = True
        validate_assignment = True