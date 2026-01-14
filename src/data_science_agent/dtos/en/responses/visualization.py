from typing import Optional

from pydantic import Field

from data_science_agent.dtos.de.responses.goal import Goal
from data_science_agent.dtos.de.responses.code import Code
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase
from data_science_agent.dtos.en.responses.lida_evaluation import LidaEvaluation


class Visualization(VisualizationBase):
    """DTO containing all information about a visualization."""
    goal: Goal = Field(..., description="The visualization goal to be achieved with this visualization.")
    code: Code = Field(..., description="The generated code to create the visualization.")
    pre_refactoring_evaluation: Optional[LidaEvaluation] = Field(..., description="The LIDA-Evaluation of the visualization before refactoring.")
    post_refactoring_evaluation: Optional[LidaEvaluation] = Field(..., description="The LIDA-Evaluation of the visualization after refactoring.")