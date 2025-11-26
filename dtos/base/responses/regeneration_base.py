from pydantic import BaseModel, Field


class RegenerationBase(BaseModel):
    """Base DTO for regeneration information (language neutral)."""

    should_be_regenerated: bool = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
