from pydantic import BaseModel, Field


class MetadataBase(BaseModel):
    """Base DTO for RDF triple (language neutral)."""

    subject: str = Field(...)
    predicate: str = Field(...)
    object: str = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True