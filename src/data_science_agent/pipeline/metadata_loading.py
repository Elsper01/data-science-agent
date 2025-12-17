from rdflib.graph import Graph

from data_science_agent.dtos.base import MetadataBase
from data_science_agent.graph import AgentState
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE

Metadata = import_language_dto(AGENT_LANGUAGE, base_dto_class=MetadataBase)


@track_duration
def load_metadata(state: AgentState) -> AgentState:
    """Loads whole metadata from an RDF file into the agent state."""
    g = Graph()
    g.parse(state["metadata_path"], format="xml")
    metadata = []
    for s, p, o in g.triples((None, None, None)):
        try:
            metadata.append(Metadata(s, p, o))
        except Exception:
            continue
    state["metadata"] = metadata
    return state
