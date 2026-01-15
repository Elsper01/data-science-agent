from typing import Optional, Any

from pydantic import Field
from data_science_agent.dtos.base.responses.code_base import CodeBase


class Code(CodeBase):
    """Antwortformat: Code zur Umsetzung des Visualisierungsziels, generiert vom LLM."""
    code: str = Field(..., description="Syntaktisch korrekter code.")