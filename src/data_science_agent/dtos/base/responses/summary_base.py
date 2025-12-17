from pydantic import BaseModel, Field
from typing import List
from data_science_agent.dtos.base.responses.column_base import ColumnBase


class SummaryBase(BaseModel):
    """Base DTO for dataset summary (language neutral)."""

    summary: str = Field(...)
    columns: List[ColumnBase] = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True