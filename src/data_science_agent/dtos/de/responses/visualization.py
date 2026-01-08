from pydantic import Field

from data_science_agent.dtos.de.responses.goal import Goal
from data_science_agent.dtos.de.responses.code import Code
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase


class Visualization(VisualizationBase):
    """DTO das alle Informationen zu einer Visualisierung enthält."""
    goal: Goal = Field(..., description="Das Visualierungsziel, welches mit dieser Visualisierung erreicht werden soll.")
    code: Code = Field(..., description="Der generierte Code zur Erstellung der Visualisierung.")
