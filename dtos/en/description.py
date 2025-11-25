from typing import Optional, Union
from pydantic import BaseModel, Field

class Description(BaseModel):
    """
    Represents the descriptive statistics of a single column from the output
    of pandas.DataFrame.describe().
    """

    column_name: str = Field(
        ...,
        description="The name of the column this description refers to."
    )

    count: Optional[float] = Field(
        None, description="Number of non-null values (observations)."
    )
    mean: Optional[float] = Field(
        None, description="Arithmetic mean of the column."
    )
    std: Optional[float] = Field(
        None, description="Standard deviation of the column."
    )
    min: Optional[float] = Field(
        None, description="Smallest observed value."
    )
    percentile_25: Optional[float] = Field(
        None, alias="25%", description="25th percentile (first quartile)."
    )
    percentile_50: Optional[float] = Field(
        None, alias="50%", description="50th percentile (median)."
    )
    percentile_75: Optional[float] = Field(
        None, alias="75%", description="75th percentile (third quartile)."
    )
    max: Optional[float] = Field(
        None, description="Largest observed value."
    )

    unique: Optional[int] = Field(
        None, description="Number of unique values in the column."
    )
    top: Optional[Union[int, float, str, bool]] = Field(
        None, description="Most frequent (top) value in the column."
    )
    freq: Optional[int] = Field(
        None, description="Frequency of the top value."
    )