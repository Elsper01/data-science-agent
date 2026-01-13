from pydantic import BaseModel, Field

from data_science_agent.dtos.base.responses.lida_evaluation_metrik_base import LidaEvaluationMetrikBase


class LidaEvaluationMetrik(LidaEvaluationMetrikBase):
    """DTO das eine einzelne Metrikbewertung für die LIDA-Evaluation darstellt."""

    score: int = Field(..., description="Die Bewertung der Metrik auf einer Skala von 1 bis 10. 1 bedeutet schlecht, 10 bedeutet ausgezeichnet.")
    rationale: str = Field(..., description="Die Begründung für die gegebene Bewertung.")
