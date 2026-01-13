from pydantic import Field

from data_science_agent.dtos.base.responses.lida_evaluation_base import LidaEvaluationBase


class LidaEvaluation(LidaEvaluationBase):
    """DTO for evaluating a visualization according to various criteria by an LLM."""

    dimension: str = Field(...,
                           description="The dimension to be evaluated. The following dimensions exist: bugs, transformation, compliance, type, encoding, aesthetics")
    score: int = Field(...,
                       description="The score is an integer value between 1 and 10, where 1 is the worst and 10 is the best rating.")
    rationale: str = Field(..., description="Justification for the assigned score.")
