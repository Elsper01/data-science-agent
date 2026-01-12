import importlib
from typing import Dict, Type

from pydantic import BaseModel

from data_science_agent.utils.enums import Language

IMPORT_BASE_PATH = "data_science_agent.dtos"

DTO_MODULES = {
    "Description": "description",
    "Metadata": "metadata",
    "Code": "responses.code",
    "Regeneration": "responses.regeneration",
    "Summary": "responses.summary",
    "Judge": "responses.judge",
    "JudgeVerdict": "responses.judge_verdict",
    "GoalContainer": "responses.goal_container",
    "Visualization": "responses.visualization",
    "VisualizationContainer": "responses.visualization_container",
}


def import_all_language_dtos(language: Language) -> Dict[str, Type]:
    """
    Dynamically import all DTO classes for a language.
    Returns a dictionary with {class name: class}.
    """
    dtos: Dict[str, Type] = {}

    for class_name, rel_module_path in DTO_MODULES.items():
        module_path = f"dtos.{language.value}.{rel_module_path}"
        try:
            module = importlib.import_module(module_path)
            dto_class = getattr(module, class_name)
            dtos[class_name] = dto_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Error occurred while importing {class_name} from {module_path}: {e}") from e

    return dtos


def import_language_dto(language: Language, base_dto_class: Type[BaseModel]) -> Type[BaseModel]:
    """
    Gets the module path for a specific language DTO.
    """
    base_name = base_dto_class.__name__

    dto_name = base_name.removesuffix("Base")
    module_path = DTO_MODULES.get(dto_name)

    if not module_path:
        raise ValueError(f"No DTO-module found for the base class '{base_name}'.")

    full_module_path = f"dtos.{language.value}.{module_path}"

    try:
        module = importlib.import_module(full_module_path)
        dto_class = getattr(module, dto_name)
        return dto_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Error occurred while importing '{dto_name}' for language '{language.value}': {e}"
        ) from e
