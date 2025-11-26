from typing import Optional

from pydantic import BaseModel, Field


class ColumnBase(BaseModel):
    """Base DTO for dataset summary column (language neutral)."""

    name: str = Field(...)
    unit: Optional[str] = Field(None)
    description: str = Field(...)
    missing_values: int = Field(...)
    unique_values: int = Field(...)
    most_frequent_value: str = Field(...)
    visualization_hint: Optional[str] = Field(None)
    note: Optional[str] = Field(None)

    # allow population by field name or alias and validate on every assignment
    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
