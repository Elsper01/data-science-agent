from pydantic import Field
from data_science_agent.dtos.base.responses.goal_container_base import GoalContainerBase
from data_science_agent.dtos.base.responses.goal_base import GoalBase

class GoalContainer(GoalContainerBase):
    """Contains all visualization goals that the agent has identified for the given dataset based on the metadata and the generated summary."""
    goals: list[GoalBase] = Field(..., description="The goals that the agent has identified for the given dataset based on the metadata and the generated summary.")
