from typing import Optional

from pydantic import Field

from data_science_agent.dtos.base.responses.column_base import ColumnBase


class Column(ColumnBase):
    """Describes a column in the dataset summary."""
    name: str = Field(..., description="Name of the column in the dataset summary.")
    unit: Optional[str] = Field(..., description="Unit of the values in this column.")
    description: str = Field(..., description="Description of the dataset column.")
    missing_values: Optional[int] = Field(..., description="Number of missing values in this column.")
    unique_values: Optional[int] = Field(..., description="Number of unique values in this column.")
    most_frequent_value: Optional[str | int | float] = Field(..., description="The most frequent value in this column.")
    visualization_hint: str = Field(None, description="Suggestion for how to best visualize this column.")
    note: Optional[str] = Field(None, description="Additional notes about the column.")
