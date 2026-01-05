from pydantic import Field
from data_science_agent.dtos.base.responses.code_base import CodeBase

class Code(CodeBase):
    """Antwortformat: Code, generiert vom LLM, gefolgt von einer kurzen Erklärung des Codes."""
    explanation: str = Field(..., description="Erklärt den Code kurz.")
    code: str = Field(..., description="Syntaktisch korrekter code.")
