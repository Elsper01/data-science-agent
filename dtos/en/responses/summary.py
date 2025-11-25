from pydantic import BaseModel, Field
from .column import Column


class Summary(BaseModel):
    """Describes the response format of the LLM when asked for a summary of the dataset and its columns."""
    summary: str = Field(..., description="Detailed description of the dataset.")
    columns: list[Column] = Field(...,
                                  description="One entry per column in the dataset summary: column name, short explanation of the column name and unit of measurement, number of missing values, and key notes about this column.")
