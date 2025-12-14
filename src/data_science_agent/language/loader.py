import importlib
from types import ModuleType

from data_science_agent.utils.enums import Language

from pydantic import BaseModel

IMPORT_BASE_PATH = "data_science_agent.dtos"

def import_all_language_dtos(language: Language):
    code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module = __get_all_dto_paths(
        language)

    Description = getattr(description_module, "Description")
    Metadata = getattr(metadata_module, "Metadata")
    Code = getattr(code_module, "Code")
    Regeneration = getattr(regeneration_module, "Regeneration")
    Summary = getattr(summary_module, "Summary")
    Judge = getattr(judge_module, "Judge")

    return Description, Metadata, Code, Regeneration, Summary, Judge


def __get_all_dto_paths(language: Language) -> tuple[
    ModuleType, ModuleType, ModuleType, ModuleType, ModuleType, ModuleType]:
    """Imports the language specific DTO's."""
    code_module = importlib.import_module(f"dtos.{language}.responses.code")
    regeneration_module = importlib.import_module(f"dtos.{language}.responses.regeneration")
    summary_module = importlib.import_module(f"dtos.{language}.responses.summary")
    judge_module = importlib.import_module(f"dtos.{language}.responses.judge")
    description_module = importlib.import_module(f"dtos.{language}.description")
    metadata_module = importlib.import_module(f"dtos.{language}.metadata")
    return code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module


def import_language_dto(language: Language, requested_dto: BaseModel) -> BaseModel:
    """Imports a specific language DTO."""
    code_module, description_module, judge_module, metadata_module, regeneration_module, summary_module = __get_all_dto_paths(
        language)
    if requested_dto.__name__ == "CodeBase":
        return getattr(code_module, "Code")
    elif requested_dto.__name__ == "RegenerationBase":
        return getattr(regeneration_module, "Regeneration")
    elif requested_dto.__name__ == "SummaryBase":
        return getattr(summary_module, "Summary")
    elif requested_dto.__name__ == "JudgeBase":
        return getattr(judge_module, "Judge")
    elif requested_dto.__name__ == "DescriptionBase":
        return getattr(description_module, "Description")
    elif requested_dto.__name__ == "MetadataBase":
        return getattr(metadata_module, "Metadata")
    else:
        raise ValueError(f"Requested DTO {requested_dto.__name__} not found for language {language}.")


