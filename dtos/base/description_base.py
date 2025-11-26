from typing import Optional, Union
from pydantic import BaseModel, Field


class DescriptionBase(BaseModel):
    """Base DTO for column description (language neutral)."""

    column_name: str = Field(...)
    count: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    percentile_25: Optional[float] = Field(None, alias="25%")
    percentile_50: Optional[float] = Field(None, alias="50%")
    percentile_75: Optional[float] = Field(None, alias="75%")
    max: Optional[float] = None
    unique: Optional[int] = None
    top: Optional[Union[int, float, str, bool]] = None
    freq: Optional[int] = None

    # used to allow population by field name or alias and validation of every assignment
    class Config:
        allow_population_by_field_name = True
        validate_assignment = True