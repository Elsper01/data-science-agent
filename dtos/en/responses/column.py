from pydantic import BaseModel, Field
from typing import Optional


class Column(BaseModel):
    """Describes a column in the dataset summary."""
    name: str = Field(..., description="Name of the column in the dataset summary.")
    unit: Optional[str] = Field(..., description="Unit of the values in this column.")
    description: str = Field(..., description="Description of the dataset column.")
    missing_values: int = Field(..., description="Number of missing values in this column.")
    unique_values: int = Field(..., description="Number of unique values in this column.")
    most_frequent_value: str = Field(..., description="The most frequent value in this column.")
    visualization_hint: str = Field(None, description="Suggestion for how to best visualize this column.")
    note: Optional[str] = Field(None, description="Additional notes about the column.")
