import importlib
import os
import subprocess
import sys
from enum import StrEnum
from typing import TypedDict, List, Union, Any
from time import time

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

    except ModuleNotFoundError as e:
        raise ImportError(f"DTO modules for language '{language}' not found: {e}")


load_dotenv()

# determine language and load corresponding DTOs and prompts
language = determine_language()
Description, Metadata, Code, Regeneration, Summary, Judge = import_language_dtos(language)


class ProgrammingLanguage(StrEnum):
    PYTHON = "py"
    R = "r"


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    judge_messages: List[Union[HumanMessage, AIMessage]]
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


def llm_summary(state: AgentState) -> AgentState:
    """Dieser Knoten generiert mittel einem LLM die Zusammenfassung des Datensatzes."""

    system_prompt = get_prompt(
        language.value,
        "summary_system_prompt",
        column_names=str(state.get("column_names", [])),
        descriptions=str(state.get("descriptions", [])),
        metadata=str(state.get("metadata", [])[:30])
    )

    user_content = get_prompt(
        language.value,
        "summary_user_prompt"
    )
    user_msg = HumanMessage(content=user_content)

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Summary,
        system_prompt=system_prompt
    )

    llm_response = temp_agent.invoke({"messages": [user_msg]})
    print(f"{bcolors.HEADER}LLM result: {bcolors.ENDC}")
    summary: Summary = llm_response["structured_response"]
    print(f"{bcolors.OKGREEN}Dataset Summary:{bcolors.ENDC}")
    print(summary.summary)
    print(f"\n{bcolors.OKBLUE}Column Descriptions:{bcolors.ENDC}")
    print(summary.columns)

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
    # TODO: einen neuen Prompt erstellen mit System Prompt, und alle bis jetzt erarbeiteten Infos mit übergeben

    description_user_message, temp_agent = _get_generate_code_agent(state)

    code_user_message = get_prompt(
        language.value,
        "generate_python_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head().to_markdown())
    )

    code_user_message = HumanMessage(content=code_user_message)

    messages = [description_user_message, code_user_message]
    return _generate_and_write_code(state, temp_agent, messages)


def _get_generate_code_agent(state: AgentState):
    programming_language = state["programming_language"]
    system_prompt = \
        f"""
        Du bist ein Experte für Data Science, {programming_language.value}‑Programmierung und Datenvisualisierung. 
        Deine Hauptaufgabe ist Code zur Berechnung und Visualisierung von Data-Science Analysen eines Datensatzes zu erzeugen.
        Allgemeine Prinzipien deines Handelns:
        
        1. Schreibe stets korrekten und ausführungssicheren {programming_language.value}‑Code.
        {
            """
            2. Nutze ausschließlich die folgenden Bibliotheken, wenn nicht anders erlaubt: 
            pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap.
            """ 
        if programming_language == ProgrammingLanguage.PYTHON 
        else "2. Installiere alle benötigten Pakete am Anfang des Skripts."
        }
        3. Beachte bei der Erstellung oder Bewertung von Visualisierungen die Qualitätskriterien für gute Datenvisualisierung:
           - Angemessenheit des Diagrammtyps (z.B. Karte bei Geokoordinaten),
           - Klarheit und Lesbarkeit,
           - Daten-Treue,
           - Ästhetische Gestaltung,
           - Technische Korrektheit,
           - Effektivität der Kommunikation,
           - Konstruktive Verbesserungsvorschläge.
        4. Du darfst keinerlei vertrauliche oder urheberrechtlich geschützte Daten erzeugen oder wiedergeben.
        5. Alle Antworten sollen UTF‑8‑kompatiblen {programming_language.value}‑Code enthalten.
        
        Wenn du Code generierst, soll dieser sofort lauffähig, sauber strukturiert, modulartig und kommentiert sein. 
    """

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code,
        system_prompt=system_prompt
    )

    description_user_message = \
        f"""
        Folgende Daten wurden in einem vorherigen Schritt durch einen anderen Agenten ermittelt und analysiert:
           
        Beschreibung bzw. Zusammenfassung des Datensatzes:
        {state["summary"].summary}
        
        Beschreibung der Spalten des Datensatzes:
        {state["summary"].columns}
    """
    description_user_message = HumanMessage(content=description_user_message)

    # clear output directory before generating new plots / code
    clear_output_dir()
    return description_user_message, temp_agent


def llm_generate_r_code(state: AgentState) -> AgentState:
    """Generiert R Code für die Datenvisualisierung."""
    # TODO: einen neuen Prompt erstellen mit System Prompt, und alle bis jetzt erarbeiteten Infos mit übergeben
    description_user_message, temp_agent = _get_generate_code_agent(state)

    code_user_message = get_prompt(
        language.value,
        "generate_r_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head(10).to_markdown()),
        summary_columns=str(getattr(state.get("summary", None), "columns", "")),
        summary=str(getattr(state.get("summary", None), "summary", ""))
    )

    code_user_message = HumanMessage(content=code_user_message)

    messages = [description_user_message, code_user_message]
    return _generate_and_write_code(state, temp_agent, messages)


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
        MAX_ATTEMPTS = 5 # TODO: ins env file packen
        if regeneration_response.should_be_regenerated and state["regeneration_attempts"] < MAX_ATTEMPTS:
            print(f"{bcolors.WARNING}Regenerating code, attempt {state['regeneration_attempts']}{bcolors.ENDC}")
            return "regenerate_code"
        elif state["regeneration_attempts"] == MAX_ATTEMPTS:
            print(f"{bcolors.OKGREEN}Max attempts limit ({MAX_ATTEMPTS}) succeeded.{bcolors.ENDC}")
            return "judge"
        else:
            print(f"{bcolors.OKGREEN}Code is functionally working.{bcolors.ENDC}")
            return "judge"
    else:
        return "judge"


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
    return _generate_and_write_code(state, temp_agent, state["messages"])


def _generate_and_write_code(state: AgentState, temp_agent, messages) -> AgentState:
    llm_response = temp_agent.invoke({"messages": messages})
    path_base = "./output/generate_plots."
    state["script_path"] = path_base + state["programming_language"].value
    with open(state["script_path"], "w", encoding="UTF-8") as file:
        print(f"{bcolors.HEADER}LLM regenerated code: {bcolors.ENDC}")
        code: Code = llm_response["structured_response"]
        state["code"] = code
        print(f"\n{bcolors.OKGREEN}Generated Code:{bcolors.ENDC}")
        print(code.code)
        file.write(code.code)
    state["messages"] = llm_response["messages"]
    return state

def llm_judge_plots(state: AgentState) -> AgentState:
    system_prompt = \
        f"""
            Du bist eine Expertin bzw. ein Experte für Datenvisualisierung und analytische Kommunikation. 
            Deine Aufgabe ist es, Code zu überprüfen und zu bewerten, der Diagramme oder andere Visualisierungen erzeugt. 
            Erstelle eine detaillierte Kritik auf Grundlage der folgenden Kriterien:
            
            1. **Angemessenheit des Visualisierungstyps**  
               - Stellt der gewählte Diagrammtyp die Daten effektiv dar?  
               - Gibt es eine geeignetere Visualisierungsart für den Datentyp (z.B. kartenbasierte Darstellungen für geografische Daten, Streudiagramme für Zusammenhänge usw.)?
            
            2. **Klarheit und Verständlichkeit**  
               - Ist die Visualisierung für die Zielgruppe leicht verständlich?  
               - Sind Beschriftungen, Legenden und Titel klar, informativ und ausreichend?  
               - Ist das gewählte Farbschema nachvollziehbar und barrierefrei?
            
            3. **Treue zu den Daten**  
               - Spiegelt die Visualisierung die zugrunde liegenden Daten korrekt wider?  
               - Sind Skalierungen, Achsen und Transformationen korrekt angewendet und nicht irreführend?  
               - Werden die Daten ohne Verzerrung oder unnötige Verzierungen dargestellt?
            
            4. **Ästhetik und Gestaltungsqualität**  
               - Werden visuelle Elemente (Farben, Größen, Abstände, Schriftarten) effektiv und konsistent eingesetzt?  
               - Entspricht das Design bewährten Prinzipien der Datenvisualisierung (z.B. wenig „Chartjunk“, sinnvoller Weißraum)?
            
            5. **Technische Korrektheit**  
               - Scheint der Code korrekt zu sein und erfolgreich ausführbar?  
               - Werden Bibliotheken angemessen verwendet?  
               - Sind alle erforderlichen Komponenten definiert (Datenvariablen, Figurenobjekte usw.)?
            
            6. **Wirksamkeit in der Kommunikationsleistung**  
               - Beantwortet die Visualisierung die beabsichtigte analytische Fragestellung oder verdeutlicht sie den zentralen Punkt?  
               - Ist die Kernaussage leicht zu erkennen?
            
            7. **Verbesserungsvorschläge**  
               - Gib konkrete Empfehlungen, wie das Diagramm klarer, exakter oder informativer gestaltet werden kann.
            
            Deine Kritik soll detailliert, konstruktiv und gut begründet sein und bewerten, ob die Visualisierung die Daten für die Zielgruppe effektiv vermittelt.
         """
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        system_prompt=SystemMessage(content=system_prompt),
        response_format=Judge
    )

    with open(state["script_path"], "r", encoding="UTF-8") as file:
        code_content = \
            f"""
                Mit dem folgenden Code wurden Diagramme zur Visualisierung eines Datensatzes erzeugt.
                Bitte bewerte die erzeugten Visualisierungen anhand der zuvor genannten Kriterien und gib eine ausführliche Kritik ab.
                Gib explizite Verbesserungsvorschläge, falls die Visualisierungen nicht optimal sind.
                Generierter Code:
                {file.read()}
            """

    llm_response = temp_agent.invoke({"messages": [HumanMessage(content=code_content)]})
    judge_result: Judge = llm_response["structured_response"]
    state["judge_messages"] = judge_result.verdicts
    print(f"{bcolors.OKGREEN} LLM Judge {bcolors.ENDC}")
    for x in judge_result.verdicts:
        print(f"Figure: {x.figure_name}, File: {x.file_name}")
        print(f"From line {x.line_number_from} to {x.line_number_to}")
        print(f"Critic notes: {x.critic_notes}")
        print(f"Suggested code: {x.suggestion_code}")
        print(f"Needs regeneration: {x.needs_regeneration}")
        print(f"Can be deleted: {x.can_be_deleted}")
        print("-----")

    # TODO: ich muss jetzt noch irgendwie das ganze generierete zeugs zu neu generierung weiterreichen
    # TODO: step two - we judge the generated plots as well -> das lagern wir direkt in einen evaluate agent aus, der uns alle erzeugten plots bewertet
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

start = time()

graph = StateGraph(AgentState)
graph.add_node("load_data", load_dataset)
graph.add_edge(START, "load_data")
graph.add_node("analyse_data", analyse_dataset)
graph.add_edge("load_data", "analyse_data")
graph.add_node("load_metadata", load_metadata)
graph.add_edge("analyse_data", "load_metadata")
graph.add_node("LLM create_summary", llm_summary)
graph.add_edge("load_metadata", "LLM create_summary")
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
        "judge": "LLM judge_plots"
    }
)
graph.add_node("LLM regenerate_code", llm_regenerate_code)
graph.add_node("LLM judge_plots", llm_judge_plots)
graph.add_edge("LLM judge_plots", END)
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

print(f"{bcolors.WARNING}Agent finished in {time() - start} seconds.{bcolors.ENDC}")