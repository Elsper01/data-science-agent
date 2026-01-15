from typing import Optional, Any

from pydantic import BaseModel, Field

from data_science_agent.dtos.base import GoalBase


class VisualizationWrapper(BaseModel):
    """Wrapper contains all important information about a visualization. We use this wrapper to not always force the LLM to parse / create such a complex object."""
    goal: Optional[Any] = Field(...)
    code: Optional[Any] = Field(...)
    pre_refactoring_evaluation: Optional[Any] = Field(...)
    post_refactoring_evaluation: Optional[Any] = Field(...)
