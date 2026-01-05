from pydantic import Field, BaseModel
from data_science_agent.dtos.base.responses.goal_base import GoalBase

class GoalContainerBase(BaseModel):
    """Contains all visualization goals that the agent has identified for the given dataset based on the metadata and the generated summary."""
    goals: list[GoalBase] = Field(..., description="The goals that the agent has identified for the given dataset based on the metadata and the generated summary.")
