from pydantic import Field
from data_science_agent.dtos.base.responses.goal_base import GoalBase

class Goal(GoalBase):
    """Response format: A visualization goal that aims to answer a specific question and explains how it will answer it."""
    question: str = Field(..., description="A specific question that the goal should answer.")
    visualization: str = Field(..., description="The type of visualization used to answer the question.")
    rationale: str = Field(..., description="The rationale for why this visualization was chosen to answer the question.")
