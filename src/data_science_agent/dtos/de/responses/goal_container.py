from pydantic import Field
from data_science_agent.dtos.base.responses.goal_container_base import GoalContainerBase
from data_science_agent.dtos.base.responses.goal_base import GoalBase

class GoalContainer(GoalContainerBase):
    """Enthält alle Visualisierungsziele, die der Agent für den gegebenen Datensatz basierend auf den Metadaten und der erzeugten Zusammenfassung identifiziert hat."""
    goals: list[GoalBase] = Field(..., description="Die Ziele, die der Agent für den gegebenen Datensatz basierend auf den Metadaten und der erzeugten Zusammenfassung identifiziert hat.")
