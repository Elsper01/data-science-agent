from pydantic import BaseModel, Field
from typing import Optional


class Column(BaseModel):
    """Beschreibt eine Spalte in der Zusammenfassung des Datensatzes."""
    name: str = Field(..., description="Name der Spalte in der Zusammenfassung des Datensatzes.")
    unit: Optional[str] = Field(..., description="Einheit der Werte in dieser Spalte.")
    description: str = Field(..., description="Beschreibung der Spalte des Datensatzes.")
