from pydantic import Field
from data_science_agent.dtos.base.responses.visualization_container_base import VisualizationContainerBase
from data_science_agent.dtos.de.responses.visualization import Visualization

class VisualizationContainer(VisualizationContainerBase):
    """Enthält alle Visualisierungen."""
    visualizations: list[Visualization] = Field(..., description="Die vom Agenten zu den Visualierungszielen generierten Visualisierungen.")
