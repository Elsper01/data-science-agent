import importlib
from src.data_science_agent.utils.enums import Language

IMPORT_BASE_PATH = "src.data_science_agent.dtos"

def import_language_dtos(language: Language):
    """Imports the language specific DTO's."""
    code_module = importlib.import_module(f"dtos.{language}.responses.code")
    regeneration_module = importlib.import_module(f"dtos.{language}.responses.regeneration")
    summary_module = importlib.import_module(f"dtos.{language}.responses.summary")
    judge_module = importlib.import_module(f"dtos.{language}.responses.judge")
    description_module = importlib.import_module(f"dtos.{language}.description")
    metadata_module = importlib.import_module(f"dtos.{language}.metadata")

    Description = getattr(description_module, "Description")
    Metadata = getattr(metadata_module, "Metadata")
    Code = getattr(code_module, "Code")
    Regeneration = getattr(regeneration_module, "Regeneration")
    Summary = getattr(summary_module, "Summary")
    Judge = getattr(judge_module, "Judge")

    return Description, Metadata, Code, Regeneration, Summary, Judge

