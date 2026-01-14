from typing import Optional

from pydantic import Field

from data_science_agent.dtos.de.responses.lida_evaluation import LidaEvaluation
from data_science_agent.dtos.de.responses.goal import Goal
from data_science_agent.dtos.de.responses.code import Code
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase


class Visualization(VisualizationBase):
    """DTO das alle Informationen zu einer Visualisierung enthält."""
    goal: Goal = Field(..., description="Das Visualierungsziel, welches mit dieser Visualisierung erreicht werden soll.")
    code: Code = Field(..., description="Der generierte Code zur Erstellung der Visualisierung.")
    pre_refactoring_evaluation: Optional[LidaEvaluation] = Field(..., description="Die LIDA-Evaluation der Visualisierung vor dem Refactoring")
    post_refactoring_evaluation: Optional[LidaEvaluation] = Field(..., description="Die LIDA-Evaluation der Visualisierung nach einer möglichen Refaktorisierung.")
