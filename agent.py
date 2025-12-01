import importlib
import os
import subprocess
import sys
from enum import StrEnum
from typing import TypedDict, List, Union, Any

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pandas import DataFrame
from rdflib import Graph

from prompts import get_prompt


class Language(StrEnum):
    DE = "de"
    EN = "en"


def determine_language() -> Language:
    """Entscheidet initial, welche Sprache geladen werden soll."""
    env = os.getenv("AGENT_LANGUAGE")
    if env:
        v = env.lower()
        if v.startswith("de"):
            return Language.DE
        elif v.startswith("en"):
            return Language.EN
    # Fallback: use german
    return Language.DE


def import_language_dtos(language: Language):
    """Importiert DTO-Klassen je nach Sprachwahl dynamisch."""
    try:
        code_module = importlib.import_module(f"dtos.{language}.responses.code")
        regeneration_module = importlib.import_module(f"dtos.{language}.responses.regeneration")
        summary_module = importlib.import_module(f"dtos.{language}.responses.summary")
        description_module = importlib.import_module(f"dtos.{language}.description")
        metadata_module = importlib.import_module(f"dtos.{language}.metadata")

        Description = getattr(description_module, "Description")
        Metadata = getattr(metadata_module, "Metadata")
        Code = getattr(code_module, "Code")
        Regeneration = getattr(regeneration_module, "Regeneration")
        Summary = getattr(summary_module, "Summary")

        return Description, Metadata, Code, Regeneration, Summary

    except ModuleNotFoundError as e:
        raise ImportError(f"DTO modules for language '{language}' not found: {e}")


load_dotenv()

# determine language and load corresponding DTOs and prompts
language = determine_language()
Description, Metadata, Code, Regeneration, Summary = import_language_dtos(language)


class ProgrammingLanguage(StrEnum):
    PYTHON = "py"
    R = "r"


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    dataset_path: str
    dataset_df: Any  # any because otherwise we get problems because of strict typing
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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class LLMModel(StrEnum):
    GPT_4o = "gpt-4o"
    GPT_5 = "gpt-5"
    GROK = "x-ai/grok-code-fast-1"


def decide_programming_language(state: AgentState) -> AgentState:
    """Entscheidet die Programmiersprache, in welcher der Code erzeugt werden soll."""
    programming_language: ProgrammingLanguage = state["programming_language"]
    if programming_language == ProgrammingLanguage.PYTHON:
        return "python"
    elif programming_language == ProgrammingLanguage.R:
        return "r"
    else:
        raise ValueError("Unsupported programming language")


def load_dataset(state: AgentState) -> AgentState:
    """Lädt den Datensatz und speichert ihn in einem pandas DataFrame."""
    state["dataset_df"] = pd.read_csv(
        state["dataset_path"],
        sep=";"
    )
    return state


def analyse_dataset(state: AgentState) -> AgentState:
    """Analysiert den Datensatz und speichert das Analyseergebnis im Zustand."""
    df: DataFrame = state["dataset_df"]
    state["column_names"] = list(df.columns)

    desc_df = df.describe(include="all").T.reset_index().rename(columns={"index": "column_name"})
    desc = [
        Description(**convert_nan_to_none(row.to_dict()))
        for _, row in desc_df.iterrows()
    ]
    state["descriptions"] = desc
    return state


def convert_nan_to_none(record: dict) -> dict:
    """Konvertiert einen Eintrag in einen Nicht-NaN-Wert."""
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


def load_metadata(state: AgentState) -> AgentState:
    """Lädt die Metadaten und speichert alle Tripel des eingelesenen Graphen."""
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


def load_messages(state: AgentState) -> AgentState:
    """Lädt neue Nachrichten und fügt sie dem Zustand hinzu. Nutzt zentrale Prompts."""
    sys_content = get_prompt(
        language.value,
        "summary_system_prompt",
        column_names=str(state.get("column_names", [])),
        descriptions=str(state.get("descriptions", [])),
        metadata=str(state.get("metadata", [])[:30])
    )
    sys_msg = SystemMessage(content=sys_content)

    user_content = get_prompt(
        language.value,
        "summary_user_prompt"
    )
    user_msg = HumanMessage(content=user_content)

    state["messages"] = [sys_msg, user_msg]
    return state


def llm_summary(state: AgentState) -> AgentState:
    """Dieser Knoten generiert mittel einem LLM die Zusammenfassung des Datensatzes."""
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Summary,
    )

    llm_response = temp_agent.invoke({"messages": state["messages"]})
    print(f"{bcolors.HEADER}LLM result: {bcolors.ENDC}")
    summary: Summary = llm_response["structured_response"]
    print(f"{bcolors.OKGREEN}Dataset Summary:{bcolors.ENDC}")
    print(summary.summary)
    print(f"\n{bcolors.OKBLUE}Column Descriptions:{bcolors.ENDC}")
    print(summary.columns)

    state["messages"] = llm_response["messages"]
    state["summary"] = summary

    return state


def clear_output_dir():
    files_to_keep = [".gitignore", ".gitkeep", "graph.png", "generate_plots.r"]
    for file_name in os.listdir("./output/"):
        if file_name not in files_to_keep:
            file_path = os.path.join("./output/", file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


def llm_generate_python_code(state: AgentState) -> AgentState:
    """Generiert Python Code für die Datenvisualisierung."""
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code
    )

    # clear output directory before generating new plots
    clear_output_dir()

    human_content = get_prompt(
        language.value,
        "generate_python_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head().to_markdown())
    )

    human_msg = HumanMessage(content=human_content)

    state["messages"].append(human_msg)
    return _generate_and_write_code(state, temp_agent)


def llm_generate_r_code(state: AgentState) -> AgentState:
    """Generiert R Code für die Datenvisualisierung."""
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code
    )

    # clear output directory before generating new plots
    clear_output_dir()

    human_content = get_prompt(
        language.value,
        "generate_r_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head(10).to_markdown()),
        summary_columns=str(getattr(state.get("summary", None), "columns", "")),
        summary=str(getattr(state.get("summary", None), "summary", ""))
    )

    human_msg = HumanMessage(content=human_content)

    state["messages"].append(human_msg)
    return _generate_and_write_code(state, temp_agent)


def test_generated_code(state: AgentState) -> AgentState:
    """Testet den vom LLM generierten Code für die Datenvisualisierung."""
    language: ProgrammingLanguage = state["programming_language"]
    script_path = state["script_path"]
    if language is ProgrammingLanguage.R:
        cmd = ["Rscript", script_path]
    else:  # default to python
        cmd = [sys.executable, script_path]

    generated_code_test_result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    # we save both stdout and stderr to see what the LLM produced and to determine if code must be regenerated
    if generated_code_test_result.stdout:
        state["code_test_stdout"] = generated_code_test_result.stdout
    else:
        state["code_test_stdout"] = "No output from generated code."
    if generated_code_test_result.stderr:
        state["code_test_stderr"] = generated_code_test_result.stderr
    else:
        state["code_test_stderr"] = "No errors from generated code."
    print(f"{bcolors.HEADER}Testing generated code ({language.value}): {bcolors.ENDC}")
    print("output: ", generated_code_test_result.stdout)
    print("error: ", generated_code_test_result.stderr)
    return state


def decide_regenerate_code(state: AgentState) -> AgentState:
    """Überprüft, ob eine erneute Codegenerierung erforderlich ist."""
    # LLM decides if the text in stdout and stderr are actual errors or just infos / deprecated warnings.
    if state["code_test_stdout"] or state["code_test_stderr"]:
        model = get_llm_model(LLMModel.GPT_4o)
        system_prompt = SystemMessage(
            content=get_prompt(language.value, "decide_regenerate_code_system_prompt")
        )
        user_prompt = HumanMessage(
            content=get_prompt(
                language.value,
                "decide_regenerate_code_user_prompt",
                test_stdout=state.get("code_test_stdout", ""),
                test_stderr=state.get("code_test_stderr", "")
            )
        )

        decide_agent = create_agent(
            model=model,
            response_format=Regeneration
        )

        messages = [system_prompt, user_prompt]

        llm_response = decide_agent.invoke({"messages": messages})
        regeneration_response: Regeneration = llm_response["structured_response"]
        print(f"{bcolors.OKCYAN}Regeneration decision: {regeneration_response.should_be_regenerated}{bcolors.ENDC}")
        print(regeneration_response.should_be_regenerated)
        MAX_ATTEMPTS = 3
        if regeneration_response.should_be_regenerated and state["regeneration_attempts"] < MAX_ATTEMPTS:
            print(f"{bcolors.WARNING}Regenerating code, attempt {state['regeneration_attempts']}{bcolors.ENDC}")
            return "regenerate_code"
        else:
            print(f"{bcolors.OKGREEN}No regeneration needed or max attempts reached.{bcolors.ENDC}")
            return "end"
    else:
        return "end"


def llm_regenerate_code(state: AgentState) -> AgentState:
    """Regeneriert den Code für die Datenvisualisierung basierend auf den aufgetretenen Fehlern."""
    state["messages"].append(
        HumanMessage(
            content=get_prompt(
                language.value,
                "regenerate_code_user_prompt",
                test_stdout=state.get("code_test_stdout", ""),
                test_stderr=state.get("code_test_stderr", ""),
                code_explanation=getattr(state.get("code"), "explanation", ""),
                code=getattr(state.get("code"), "code", "")
            )
        )
    )

    # clean output directory before regenerating plots
    clear_output_dir()

    state["regeneration_attempts"] += 1
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code
    )
    return _generate_and_write_code(state, temp_agent)


def _generate_and_write_code(state: AgentState, temp_agent) -> AgentState:
    llm_response = temp_agent.invoke({"messages": state["messages"]})
    state["messages"] = llm_response["messages"]
    path_base = "./output/generate_plots."
    state["script_path"] = path_base + state["programming_language"].value
    with open(state["script_path"], "w", encoding="UTF-8") as f:
        print(f"{bcolors.HEADER}LLM regenerated code: {bcolors.ENDC}")
        code: Code = llm_response["structured_response"]
        state["code"] = code
        print(f"\n{bcolors.OKGREEN}Generated Code:{bcolors.ENDC}")
        print(state["code"].code)
        f.write(state["code"].code)
    return state


def get_llm_model(model: LLMModel) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=model.value,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,  # for less hallucination
        max_tokens=50000,
        timeout=60,
    )
    return llm


graph = StateGraph(AgentState)
graph.add_node("load_data", load_dataset)
graph.add_edge(START, "load_data")
graph.add_node("analyse_data", analyse_dataset)
graph.add_edge("load_data", "analyse_data")
graph.add_node("load_metadata", load_metadata)
graph.add_edge("analyse_data", "load_metadata")
graph.add_node("load_messages", load_messages)
graph.add_edge("load_metadata", "load_messages")
graph.add_node("LLM create_summary", llm_summary)
graph.add_edge("load_messages", "LLM create_summary")
graph.add_node("LLM generate_python_code", llm_generate_python_code)
graph.add_node("LLM generate_r_code", llm_generate_r_code)
graph.add_conditional_edges(
    "LLM create_summary",
    decide_programming_language, {
        "python": "LLM generate_python_code",
        "r": "LLM generate_r_code"
    })
graph.add_node("test_generated_code", test_generated_code)
graph.add_edge("LLM generate_python_code", "test_generated_code")
graph.add_edge("LLM generate_r_code", "test_generated_code")
graph.add_conditional_edges(
    "test_generated_code",
    decide_regenerate_code,
    {
        "regenerate_code": "LLM regenerate_code",
        "end": END
    }
)
graph.add_node("LLM regenerate_code", llm_regenerate_code)
graph.add_edge("LLM regenerate_code", "test_generated_code")
agent = graph.compile()

png_bytes = agent.get_graph().draw_mermaid_png()

with open("./output/graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved graph.png successfully.")

result = agent.invoke(
    {
        "dataset_path": "./data/pegel.csv",
        "metadata_path": "./data/pegel.rdf",
        "regeneration_attempts": 0,
        "programming_language": ProgrammingLanguage.R
    }
)
