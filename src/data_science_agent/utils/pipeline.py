import os
import shutil

from langchain_openai import ChatOpenAI

from data_science_agent.utils import OPENROUTER_API_KEY, BASE_URL
from data_science_agent.utils.enums import LLMModel


def get_llm_model(model: LLMModel) -> ChatOpenAI:
    """Returns the LLM model instance based on the selected model."""
    llm = ChatOpenAI(
        model=model.value,
        api_key=OPENROUTER_API_KEY,
        base_url=BASE_URL,
        temperature=0,  # for less hallucination
        max_tokens=50000,
        timeout=60,
    )
    return llm


def clear_output_dir(path: str):
    """Clears the output directory except for specific files."""
    files_to_keep = [".gitignore", ".gitkeep", "graph.png", "generate_plots.r", "generate_plots.py"]
    for file_name in os.listdir(path):
        if file_name not in files_to_keep:
            file_path = os.path.join(path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def archive_images(path, index):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".png") and not file == "graph.png":
            os.makedirs(os.path.join(path, f"code_generation_#{index}"), exist_ok=True)
            archive_dir = os.path.join(path, f"code_generation_#{index}")
            shutil.copy(os.path.join(path, file), os.path.join(archive_dir, file))