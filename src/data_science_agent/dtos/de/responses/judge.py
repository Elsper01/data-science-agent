from pydantic import Field
from data_science_agent.dtos.base.responses.judge_base import JudgeBase
from data_science_agent.dtos.de.responses.judge_verdict import JudgeVerdict

class Judge(JudgeBase):
    """Urteil des Judge-LLMs bezüglich eines ganzen Skripts das alle Grafiken und Figuren erzeugt."""

    verdicts: list[JudgeVerdict] = Field(..., description="Liste der Urteile des Judge-LLMs bezüglich jeder einzelnen Figur und dem entsprechenden Code.")
    needs_regeneration: bool = Field(..., description="Gibt an, ob überhaupt etwas am Skript angepasst werden sollte.")