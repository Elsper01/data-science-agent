from pydantic import Field

from data_science_agent.dtos.base.responses.lida_evaluation_metrik_base import LidaEvaluationMetrikBase


class LidaEvaluationMetrik(LidaEvaluationMetrikBase):
    """DTO representing a single metric rating for the LIDA evaluation."""

    score: int = Field(..., description="The rating of the metric on a scale from 1 to 10. 1 means poor, 10 means excellent.")
    rationale: str = Field(..., description="The justification for the given rating.")