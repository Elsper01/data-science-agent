from typing import Optional, Union
from pydantic import BaseModel, Field

class Description(BaseModel):
    """
    Repräsentiert die beschreibenden Statistiken (Descriptive Statistics)
    einer einzelnen Spalte aus der Ausgabe von pandas.DataFrame.describe().
    """

    column_name: str = Field(
        ...,
        description="Der Name der Spalte, auf die sich diese Beschreibung bezieht."
    )

    count: Optional[float] = Field(
        None, description="Anzahl der nicht‑null Werte (Beobachtungen)."
    )
    mean: Optional[float] = Field(
        None, description="Arithmetisches Mittel der Spalte."
    )
    std: Optional[float] = Field(
        None, description="Standardabweichung der Spalte."
    )
    min: Optional[float] = Field(
        None, description="Kleinster beobachteter Wert."
    )
    percentile_25: Optional[float] = Field(
        None, alias="25%", description="25. Perzentil (erstes Quartil)."
    )
    percentile_50: Optional[float] = Field(
        None, alias="50%", description="50. Perzentil (Median)."
    )
    percentile_75: Optional[float] = Field(
        None, alias="75%", description="75. Perzentil (drittes Quartil)."
    )
    max: Optional[float] = Field(
        None, description="Größter beobachteter Wert."
    )

    unique: Optional[int] = Field(
        None, description="Anzahl der eindeutigen Werte in der Spalte."
    )
    top: Optional[Union[int, float, str, bool]] = Field(
        None, description="Der häufigste (Top‑)Wert in der Spalte."
    )
    freq: Optional[int] = Field(
        None, description="Häufigkeit des Top‑Wertes."
    )