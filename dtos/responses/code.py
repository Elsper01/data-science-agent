from pydantic import BaseModel, Field


class Code(BaseModel):
    """Antwortformat: Python-Code, generiert vom LLM, gefolgt von einer kurzen Erklärung des Codes."""
    explanation: str = Field(..., description="Erklärt den Code kurz.")
    code: str = Field(..., description="Syntactic correct python code.")
