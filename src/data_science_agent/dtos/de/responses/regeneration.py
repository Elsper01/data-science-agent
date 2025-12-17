from pydantic import Field

from data_science_agent.dtos.base.responses.regeneration_base import RegenerationBase


class Regeneration(RegenerationBase):
    """Gibt zurück, ob bei der Ausführung des Codes tatsächlich Fehler aufgetreten sind und der Code erneut erzeugt werden muss."""
    should_be_regenerated: bool = Field(..., description="Gibt an, ob der Code neu generiert werden muss.")
