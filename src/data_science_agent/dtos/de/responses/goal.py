from pydantic import Field
from data_science_agent.dtos.base.responses.goal_base import GoalBase

class Goal(GoalBase):
    """Antwortformat: Ein Visualisierungsziel, das eine konkrete Frage beantworten soll und begründet wie es diese beantwortet."""
    question: str = Field(..., description="Eine spezifische Frage, die das Ziel beantworten soll.")
    visualization: str = Field(..., description="Die Art der Visualisierung, die verwendet wird, um die Frage zu beantworten.")
    rationale: str = Field(..., description="Die Begründung, warum diese Visualisierung gewählt wurde, um die Frage zu beantworten.")
