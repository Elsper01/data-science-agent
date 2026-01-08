import importlib
from types import ModuleType

from pydantic import BaseModel

from data_science_agent.utils.enums import Language

IMPORT_BASE_PATH = "data_science_agent.dtos"


def import_all_language_dtos(language: Language):
    """Imports all language specific DTO's."""
    code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module, goal_container_module, visualization_module, visualization_container_module = __get_all_dto_paths(
        language)

    Description = getattr(description_module, "Description")
    Metadata = getattr(metadata_module, "Metadata")
    Code = getattr(code_module, "Code")
    Regeneration = getattr(regeneration_module, "Regeneration")
    Summary = getattr(summary_module, "Summary")
    Judge = getattr(judge_module, "Judge")
    GoalContainer = getattr(goal_container_module, "GoalContainer")
    Visualization = getattr(visualization_module, "Visualization")
    VisualizationContainer = getattr(visualization_container_module, "VisualizationContainer")

    return Description, Metadata, Code, Regeneration, Summary, Judge, GoalContainer, Visualization, VisualizationContainer


def __get_all_dto_paths(language: Language) -> tuple[
    ModuleType, ModuleType, ModuleType, ModuleType, ModuleType, ModuleType, ModuleType, ModuleType, ModuleType]:
    """Gets all module paths for a specific language DTO."""
    code_module = importlib.import_module(f"dtos.{language}.responses.code")
    regeneration_module = importlib.import_module(f"dtos.{language}.responses.regeneration")
    summary_module = importlib.import_module(f"dtos.{language}.responses.summary")
    judge_module = importlib.import_module(f"dtos.{language}.responses.judge")
    description_module = importlib.import_module(f"dtos.{language}.description")
    metadata_module = importlib.import_module(f"dtos.{language}.metadata")
    goal_container_module = importlib.import_module(f"dtos.{language}.responses.goal_container")
    visualization_module = importlib.import_module(f"dtos.{language}.responses.visualization")
    visualization_container_module = importlib.import_module(f"dtos.{language}.responses.visualization_container")
    return code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module, goal_container_module, visualization_module, visualization_container_module


def import_language_dto(language: Language, base_dto_class: BaseModel) -> BaseModel:
    """Imports a specific language DTO."""
    code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module, goal_container_module, visualization_module, visualization_container_module = __get_all_dto_paths(
        language)
    if base_dto_class.__name__ == "CodeBase":
        return getattr(code_module, "Code")
    elif base_dto_class.__name__ == "RegenerationBase":
        return getattr(regeneration_module, "Regeneration")
    elif base_dto_class.__name__ == "SummaryBase":
        return getattr(summary_module, "Summary")
    elif base_dto_class.__name__ == "JudgeBase":
        return getattr(judge_module, "Judge")
    elif base_dto_class.__name__ == "DescriptionBase":
        return getattr(description_module, "Description")
    elif base_dto_class.__name__ == "MetadataBase":
        return getattr(metadata_module, "Metadata")
    elif base_dto_class.__name__ == "GoalContainerBase":
        return getattr(goal_container_module, "GoalContainer")
    elif base_dto_class.__name__ == "VisualizationBase":
        return getattr(visualization_module, "Visualization")
    elif base_dto_class.__name__ == "VisualizationContainerBase":
        return getattr(visualization_container_module, "VisualizationContainer")
    else:
        raise ValueError(f"Requested DTO {base_dto_class.__name__} not found for language {language}.")
