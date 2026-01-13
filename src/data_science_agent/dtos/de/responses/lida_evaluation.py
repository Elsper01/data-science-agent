from pydantic import Field

from data_science_agent.dtos.base.responses.lida_evaluation_base import LidaEvaluationBase


class LidaEvaluation(LidaEvaluationBase):
    """DTO zur Bewertung einer Visualisierung nach verschiedenen Kriterien durch ein LLM."""

    dimension: str = Field(...,
                           description="Die zu bewertende Dimension. Es gibt folgende Dimensionen: bugs, transformation, compliance, type, encoding, aesthetics")
    score: int = Field(...,
                       description="Der Score ist ein ganzzahliger Wert zwischen 1 und 10, wobei 1 die schlechteste und 10 die beste Bewertung darstellt.")
    rationale: str = Field(..., description="Begründung für den vergebenen Score.")
