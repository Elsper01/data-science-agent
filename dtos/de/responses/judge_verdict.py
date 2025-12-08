from pydantic import Field
from typing import Optional

from dtos.base.responses.judge_verdict_base import JudgeVerdictBase


class JudgeVerdict(JudgeVerdictBase):
    """Repräsentiert das Urteil eines Judge-LLMs bezüglich einer einzelnen Figur und dem entsprechenden Code, welcher die Figur erzeugt."""

    file_name: str = Field(..., description="Der Name der Datei, unter welcher die Figur gespeichert wurde.")
    figure_name: str = Field(..., description="Der Name der Figur, auf die sich dieses Urteil bezieht.")
    critic_notes: str = Field(..., description="Kritikpunkte und Anmerkungen des Judge-LLMs bezüglich der Figur und des Codes.")
    suggestion_code: Optional[str] = Field(..., description="Vorgeschlagener Code des Judge-LLMs zur Verbesserung oder Korrektur der Figur.")
    needs_regeneration: bool = Field(..., description="Gibt an, ob die Figur basierend auf dem Urteil des Judge-LLMs neu generiert werden sollte.")
    can_be_deleted: bool = Field(..., description="Gibt an, ob die Figur und der zugehörige Code gelöscht werden können, da sie nicht den Anforderungen entsprechen bzw. keinen Mehrwert liefert.")