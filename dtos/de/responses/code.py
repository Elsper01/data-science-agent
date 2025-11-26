from pydantic import Field
from dtos.base.metadata_base import MetadataBase

class Code(MetadataBase):
    """Antwortformat: Code, generiert vom LLM, gefolgt von einer kurzen Erklärung des Codes."""
    explanation: str = Field(..., description="Erklärt den Code kurz.")
    code: str = Field(..., description="Syntaktisch korrekter code.")
