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

from dtos.description import Description
from dtos.metadata import Metadata
from dtos.responses.code import Code
from dtos.responses.regeneration import Regeneration
from dtos.responses.summary import Summary

load_dotenv()

class ProgrammingLanguage(StrEnum):
    PYTHON = "py"
    R = "r"

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    dataset_path: str
    dataset_df: Any  # any because otherwise we get problems because of strict typing
    metadata_path: str
    metadata: list[Metadata]
    columns: list[str]  # TODO: wieso verwenden wir hier noch nie das Column DTO?
    descriptions: list[Description]
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
    language: ProgrammingLanguage = state["programming_language"]
    if language == ProgrammingLanguage.PYTHON:
        return "python"
    elif language == ProgrammingLanguage.R:
        return "r"
    else:
        raise ValueError("Unsupported programming language")


def load_dataset(state: AgentState) -> AgentState:
    """Lädt den Datensatz und speichert ihn in einem pandas DataFrame."""
    state["dataset_df"] = pd.read_csv(state["dataset_path"],
                                      sep=";")
    return state


def analyse_dataset(state: AgentState) -> AgentState:
    """Analysiert den Datensatz und speichert das Analyseergebnis im Zustand."""
    df: DataFrame = state["dataset_df"]
    state["columns"] = list(df.columns)

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
    """Lädt neue Nachrichten und fügt sie dem Zustand hinzu."""
    sys_msg = SystemMessage(
        content= \
            f"""
                Du bist ein Data Science-Experte und hilft mir dabei den tabellarische Datensatz zu analysieren.
                Als Antwort hätte ich gerne eine Zusammenfassung über den Datensatz. Das heißt, um was geht es im Datensatz, was ist auffällig, gibt es Trends, fehlen Werte. 
                Das Ganze soll so dargestellt werden, dass ein beliebiger Nutzer etwas damit anfangen kann. 
                Ein Experte soll aber auch eine Eindruck davon bekommen ob der Datensatz für ihn geeignet ist oder nicht.

                Hier sind alle relevanten Daten:
                - Columns: {state["columns"]}
                - Descriptions: {state["descriptions"]}
                - Metadata: {state["metadata"][:30]}
            """
    )

    user_msg = HumanMessage(
        content= \
            """
             Fasse mir den Datensatz passend zusammen.
             Ich möchte als Ergebnis eine Zusammenfassung bzw. Erklärung des Datensatzes und eine Erklärung / Beschreibung der einzelnen Spalten des Datensatzes.
             Die verwendete Sprache soll Deutsch sein.  
            """
    )
    state["messages"] = [sys_msg, user_msg]
    return state


def llm_summary(state: AgentState) -> AgentState:
    """Dieser Knoten generiert mittel einem LLM die Zusammenfassung des Datensatzes."""
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GROK),
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

    return state


def clear_output_dir():
    files_to_keep = [".gitignore", "graph.png", "generate_plots.r"]
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

    human_msg = HumanMessage(
        content= \
        f"""
        Erzeuge mir basierend auf der vorherigen Zusammenfassung und der Datenstruktur Python-Code,
        der eine explorative Datenanalyse (EDA) des Datensatzes durchführt und passende Visualisierungen erstellt.

        Die Daten können mit folgendem Befehl geladen werden:
        `df = pd.read_csv("./data/pegel.csv", sep=";")`

        Vorgaben für den Code:
        - Verwende ausschließlich `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, `basemap`.
        - Der Code soll modular, gut kommentiert und direkt ausführbar sein, ohne syntaktische Fehler.
        - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
          `./output/<plot_name>.png`
        - Wähle Diagrammtypen entsprechend der Datenbedeutung:
          - Geographische Variablen → räumliche Verteilung (z.B. Karte mit Markierung der Punkte).
          - Zeitliche Variablen → Untersuchen ob sich ein zeitlicher Verlauf einer anderen Variable abbilden lässt.
          - Numerische Variablen → Histogramme, Boxplots und Scatterplots für Zusammenhänge.
          - Kategorische Variablen → Balkendiagramme der Häufigkeitsverteilung (ggf. Top 10 für lange Listen).
        - Führe auch kurze statistische Analysen durch, gegebenen falls mit Visualisierung:
          - Anteil fehlender Werte je Spalte,
          - Korrelationen numerischer Variablen,
          - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
        - Verwende Farben, Beschriftungen und Titel sinnvoll:
          - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
          - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
          - Achsen in SI-Einheiten oder sinnvollen Skalen beschriften.
        - Füge kurze erklärende Kommentare hinzu, **warum** bestimmte Visualisierungen sinnvoll sind.
        - Priorisiere Plot-Typen, die einem Data-Science-Workflow entsprechen (Datenqualität, Verteilung, Beziehung, Geografie, Zeit).
        - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
        - Der zurückgegebene Code soll bitte in UTF-8 kodiert sein.
        - Für jede Visualisierung soll eine separate Methode erstellt werden, welche am Ende des Skripte mit try except ausgeführt wird. Die Fehler Meldung soll ausgegeben werden, aber die Ausführung des restlichen Codes soll nicht abgebrochen werden.

        Zum besseren Verständnis:
        Das ist das Ergebnis von `df.head(10)`:
        {str(state["dataset_df"].head().to_markdown())}
        """
    )

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

    human_msg = HumanMessage(
        # TODO: hier später noch anpassen, welche Dateityp die Datei ist und welcher Seperator verwendet wird
        content= \
            f"""
            Erzeuge mir basierend auf der vorherigen Zusammenfassung und der Datenstruktur R-Code, 
            der eine explorative Datenanalyse (EDA) des Datensatzes durchführt und passende Visualisierungen erstellt.

            Die Daten können  aus folgender CSV geladen werden:
            - Pfad zur CSV Datei: `{state["dataset_path"]}`
            - Trennzeichen: `;`

            Vorgaben für den Code:
            - Der Code soll modular, gut kommentiert und direkt ausführbar sein, ohne syntaktische Fehler.
            - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
              `./output/<plot_name>.png`
            - Wähle Diagrammtypen entsprechend der Datenbedeutung:
              - Geographische Variablen → räumliche Verteilung (z.B. Karte mit Markierung der Punkte).
              - Zeitliche Variablen → Untersuchen ob sich ein zeitlicher Verlauf einer anderen Variable abbilden lässt.
              - Numerische Variablen → Histogramme, Boxplots und Scatterplots für Zusammenhänge.
              - Kategorische Variablen → Balkendiagramme der Häufigkeitsverteilung (ggf. Top 10 für lange Listen).
            - Führe auch kurze statistische Analysen durch, gegebenen falls mit Visualisierung:
              - Anteil fehlender Werte je Spalte,
              - Korrelationen numerischer Variablen,
              - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
            - Verwende Farben, Beschriftungen und Titel sinnvoll:
              - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
              - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
              - Achsen in SI-Einheiten oder sinnvollen Skalen beschriften.
            - Füge kurze erklärende Kommentare hinzu, **warum** bestimmte Visualisierungen sinnvoll sind.
            - Priorisiere Plot-Typen, die einem Data-Science-Workflow entsprechen (Datenqualität, Verteilung, Beziehung, Geografie, Zeit).
            - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
            - Der zurückgegebene Code soll bitte in UTF-8 kodiert sein.
            - Für jede Visualisierung soll eine separate Methode erstellt werden, welche am Ende des Skripte mit try except ausgeführt wird. Die Fehler Meldung soll ausgegeben werden, aber die Ausführung des restlichen Codes soll nicht abgebrochen werden.

            Zum besseren Verständnis:  
            Das ist das Ergebnis von `df.head(10)`:
            {str(state["dataset_df"].head().to_markdown())}
            """
    )

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
    """Überprüft, ob eine erneute Codegenerierung erforderlich ist. Eine erneute Generierung wird durchgeführt, wenn Fehler im Code auftreten und die maximale Anzahl an Versuchen noch nicht erreicht ist."""
    # LLM decides if the text in stdout and stderr are actual errors or just infos / deprecated warnings.
    if state["code_test_stdout"] or state["code_test_stderr"]:
        model = get_llm_model(LLMModel.GPT_4o)
        system_prompt = SystemMessage(
            content= \
                f"""
                    Du bist ein Experte darin Python Code Output zu interpretieren, der entscheidet, ob der gegebene Text Fehler enthält, die eine erneute Generierung des Codes erforderlich machen.
                    Antworte mit einer bool Antwort, welche true ist, genau dann wenn der Code Fehler enthält, die eine erneute Erzeugung des Codes zwingend notwendig machen.
                    Ansonsten antworte mit false.
                    Wichtig, du bekommst den Text von stdout und stderr des Codes. Das heißt gegebenfalls sind dort auch nur Infos oder Deprecated Warnings enthalten, diese musst du von wahren Fehlern bzw. Exceptions unterscheiden, welche unbedingt korrigiert werden müssen damit ein Diagramm erzeugt werden kann und den restlichen Ablauf des Skriptes nicht behindern.
                """
        )
        user_prompt = HumanMessage(
            content=
            f"""
                    Hier ist die Ausgabe (stdout) und die Fehlerausgabe (stderr) des Codes:
                    stdout:
                    {state["code_test_stdout"]}
                    
                    stderr:
                    {state["code_test_stderr"]}
                    
                    Bitte entscheide, ob der Code unbedingt neu generiert werden muss.
                """
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
            print(f"{bcolors.WARNING}Regenerating code, attempt {state["regeneration_attempts"]}{bcolors.ENDC}")
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
            content=f"""
            Der vorherige Code hatte folgende Fehler:
            stdout:
            {state["code_test_stdout"]}
            stderr:
            {state["code_test_stderr"]}
            Bitte generiere den Code erneut und behebe die oben genannten Fehler.
            Das ist die Beschreibung des Codes:
            {state["code"].explanation}
            Das ist der vorherige Code:
            {state["code"].code}
            """
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
    decide_programming_language,{
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
