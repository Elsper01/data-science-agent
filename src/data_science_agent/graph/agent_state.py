from typing import Any, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.language import import_all_language_dtos
from data_science_agent.utils import AGENT_LANGUAGE, DurationMetadata, LLMMetadata
from data_science_agent.utils.enums import ProgrammingLanguage

all_dtos = import_all_language_dtos(AGENT_LANGUAGE)


class AgentState(TypedDict):
    """Represents the state of the data science agent throughout its workflow."""
    # General
    output_path: str
    messages: list[HumanMessage, AIMessage]
    durations: list[DurationMetadata]
    llm_metadata: list[LLMMetadata]
    statistics_path: str
    # Dataset
    dataset_path: str
    dataset_df: Any  # any because typing of np.dataframe is not easily possible
    # Metadata
    metadata_path: str
    metadata: list[all_dtos["Metadata"]]
    # Summary
    column_names: list[str]
    descriptions: list[all_dtos["Description"]]
    summary: all_dtos["Summary"]
    # Code Generation and Testing
    regeneration_attempts: int
    programming_language: ProgrammingLanguage
    is_refactoring: bool
    # visualization goals
    goals: all_dtos["GoalContainer"]
    # visualizations
    visualizations: all_dtos["VisualizationContainer"]
    # LIDA evaluation
    is_before_refactoring: bool
