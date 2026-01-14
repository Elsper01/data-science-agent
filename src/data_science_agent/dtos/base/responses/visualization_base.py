from typing import Optional

from pydantic import BaseModel, Field

from data_science_agent.dtos.base.responses.lida_evaluation_base import LidaEvaluationBase
from data_science_agent.dtos.base.responses.goal_base import GoalBase

from data_science_agent.dtos.base.responses.code_base import CodeBase


class VisualizationBase(BaseModel):
    """DTO to hold all information related to a visualization."""
    goal: GoalBase = Field(...)
    code: CodeBase = Field(...)
    pre_refactoring_evaluation: Optional[LidaEvaluationBase] = Field(...)
    post_refactoring_evaluation: Optional[LidaEvaluationBase] = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
