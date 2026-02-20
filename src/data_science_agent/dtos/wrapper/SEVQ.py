from typing import Any

from pydantic import BaseModel, Field


class SEVQ(BaseModel):
    """A wrapper to save the LIDA SEVQ evaluation result."""

    fig_index: int = Field(...)
    model: str = Field(...)
    lida_evaluation_score: Any = Field(...)
