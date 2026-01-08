from typing import Optional

from pydantic import Field
from data_science_agent.dtos.base.responses.code_base import CodeBase

class Code(CodeBase):
    """Antwortformat: Code zur Umsetzung des Visualisierungsziels, generiert vom LLM."""
    code: str = Field(..., description="Syntaktisch korrekter code.")
    std_err: Optional[str] = Field(..., description="Fehlermeldungen, die bei der Ausführung des Codes aufgetreten sind.")
    std_out: Optional[str] = Field(..., description="Die Ausgabe, die bei der Ausführung des Codes ausgegeben wurde.")
    # TODO: wir müssen hier die messages von der Erzeugung des Codes abspeichern um sie später verwenden zu können
