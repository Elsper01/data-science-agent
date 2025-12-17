from typing import Optional

from pydantic import Field

from data_science_agent.dtos.base.responses.column_base import ColumnBase


class Column(ColumnBase):
    """Beschreibt eine Spalte in der Zusammenfassung des Datensatzes."""
    name: str = Field(..., description="Name der Spalte in der Zusammenfassung des Datensatzes.")
    unit: Optional[str] = Field(..., description="Einheit der Werte in dieser Spalte.")
    description: str = Field(..., description="Beschreibung der Spalte des Datensatzes.")
    missing_values: Optional[int] = Field(..., description="Anzahl der fehlenden Werte in dieser Spalte.")
    unique_values: Optional[int] = Field(..., description="Anzahl der eindeutigen Werte in dieser Spalte.")
    most_frequent_value: Optional[str | int | float] = Field(..., description="Der am häufigsten vorkommende Wert in dieser Spalte.")
    visualization_hint: str = Field(None, description="Vorschlag wie man diese Spalte best möglich visualisieren kann. Es sollen auch andere Spalten mit angegeben werden, wenn diese für die Visualisierung relevant sind.")
    note: Optional[str] = Field(None, description="Zusätzliche Anmerkungen zur Spalte.")
