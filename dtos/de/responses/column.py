from pydantic import BaseModel, Field
from typing import Optional


class Column(BaseModel):
    """Beschreibt eine Spalte in der Zusammenfassung des Datensatzes."""
    name: str = Field(..., description="Name der Spalte in der Zusammenfassung des Datensatzes.")
    unit: Optional[str] = Field(..., description="Einheit der Werte in dieser Spalte.")
    description: str = Field(..., description="Beschreibung der Spalte des Datensatzes.")
    missing_values: int = Field(..., description="Anzahl der fehlenden Werte in dieser Spalte.")
    unique_values: int = Field(..., description="Anzahl der eindeutigen Werte in dieser Spalte.")
    most_frequent_value: str = Field(..., description="Der am häufigsten vorkommende Wert in dieser Spalte.")
    visualization_hint: str = Field(None, description="Vorschlag wie man diese Spalte best möglich visualisieren kann.")
    note: Optional[str] = Field(None, description="Zusätzliche Anmerkungen zur Spalte.")
