from typing import Optional, Any

from pydantic import Field
from data_science_agent.dtos.base.responses.code_base import CodeBase
from data_science_agent.dtos.de.responses.judge_verdict import JudgeVerdict


class Code(CodeBase):
    """Antwortformat: Code zur Umsetzung des Visualisierungsziels, generiert vom LLM."""
    code: str = Field(..., description="Syntaktisch korrekter code.")
    std_err: Optional[str] = Field(..., description="Fehlermeldungen, die bei der Ausführung des Codes aufgetreten sind.")
    std_out: Optional[str] = Field(..., description="Die Ausgabe, die bei der Ausführung des Codes ausgegeben wurde.")
    needs_regeneration: Optional[bool] = Field(..., description="Gibt an, ob der Code basierend auf den Testergebnissen neu generiert werden muss.")
    regeneration_attempts: Optional[int] = Field(..., description="Die Anzahl der bisherigen Versuche, den Code neu zu generieren.")
    judge_result: Optional[Any] = Field(..., description="Das Urteil des LLM über die Qualität des generierten Codes.")