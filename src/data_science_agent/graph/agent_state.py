from typing import Any, Union, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.language import import_all_language_dtos
from data_science_agent.utils import AGENT_LANGUAGE
from data_science_agent.utils.enums import ProgrammingLanguage

Description, Metadata, Code, _, Summary, _ = import_all_language_dtos(AGENT_LANGUAGE)

class AgentState(TypedDict):
    """Represents the state of the data science agent throughout its workflow."""
    messages: list[Union[HumanMessage, AIMessage]]
    judge_messages: list[Union[HumanMessage, AIMessage]]
    dataset_path: str
    dataset_df: Any  # any because typing of np.dataframe is not easily possible
    metadata_path: str
    metadata: list[Metadata]
    column_names: list[str]
    descriptions: list[Description]
    summary: Summary
    code_test_stdout: str
    code_test_stderr: str
    code: Code
    script_path: str
    regeneration_attempts: int
    programming_language: ProgrammingLanguage
    is_refactoring: bool