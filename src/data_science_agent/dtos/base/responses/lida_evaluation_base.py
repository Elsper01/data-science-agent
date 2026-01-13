from pydantic import BaseModel, Field

from data_science_agent.dtos.base.responses.lida_evaluation_metrik_base import LidaEvaluationMetrikBase


class LidaEvaluationBase(BaseModel):
    """Base DTO to evaluate a visualization by different criteria by a LLM."""

    bugs: LidaEvaluationMetrikBase = Field(...)
    transformation: LidaEvaluationMetrikBase = Field(...)
    compliance: LidaEvaluationMetrikBase = Field(...)
    type: LidaEvaluationMetrikBase = Field(...)
    encoding: LidaEvaluationMetrikBase = Field(...)
    aesthetics: LidaEvaluationMetrikBase = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
