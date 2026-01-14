from pydantic import Field

from data_science_agent.dtos.base.responses.summary_base import SummaryBase
from data_science_agent.dtos.de.responses.column import Column


class Summary(SummaryBase):
    """Beschreibt das Antwortformat des LLM, wenn nach einer Zusammenfassung des Datensatzes und seiner Spalten gefragt wird."""
    summary: str = Field(..., description="Detaillierte Beschreibung des Datensatzes.  Verwende maximal 500 Wörter.")
    columns: list[Column] = Field(..., description="Ein Eintrag pro Spalte in der Zusammenfassung des Datensatzes: Name der Spalte, kurze Erklärung des Spaltennamens und der Messeinheit, Anzahl fehlender Werte und wesentliche Hinweise zu dieser Spalte.")
