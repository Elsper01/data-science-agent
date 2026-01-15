import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.dtos.base import GoalBase, CodeBase
from data_science_agent.dtos.wrapper.CodeWrapper import CodeWrapper
from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, LLMMetadata, print_color
from data_science_agent.utils.enums import LLMModel, ProgrammingLanguage, Color
from data_science_agent.utils.pipeline import clear_output_dir

prompt: Prompt = Prompt(
    de={
        "generate_code_system_prompt": \
            """
                Du bist ein Experte für Data Science, {programming_language}‑Programmierung und Datenvisualisierung.
                {library_instruction}
                Deine Hauptaufgabe ist Code zu Generieren, um ein Visualisierungsziel für den Datensatz umzusetzen und zu visualisieren.
                Beachte hierbei folgende Grundsätze:
                1. Schreibe stets korrekten und ausführungssicheren {programming_language}‑Code.
                2. Beachte bei der Erstellung oder Bewertung von Visualisierungen die Qualitätskriterien für gute Datenvisualisierung:
                   - Angemessenheit des Diagrammtyps,
                   - Klarheit und Lesbarkeit,
                   - Daten-Treue,
                   - Ästhetische Gestaltung,
                   - Technische Korrektheit,
                   - Effektivität der Kommunikation,
                   - Konstruktive Verbesserungsvorschläge
                3. Du darfst keinerlei vertrauliche oder urheberrechtlich geschützte Daten erzeugen oder wiedergeben.
                4. Alle Antworten sollen UTF‑8‑kompatiblen {programming_language}‑Code enthalten.
                Wenn du Code generierst, soll dieser sofort lauffähig sein.
                5. PRO VISUALISIERUNGSZIEL soll EIN SEPARATES SKRIPT erstellt werden und PRO SKRIPT MAXIMAL EINE VISUALISIERUNG umgesetzt werden.
                6. JEDES SKRIPT soll die Visualisierung als PNG-Datei im Output Verzeichnis speichern.
            """,
        "generate_code_python_lib_instruction": \
            "Nutze ausschließlich die folgenden Bibliotheken: pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap und plotly.",
        "generate_code_r_lib_instruction": \
            "Installiere und lade alle benötigten Pakete am Anfang des Skripts (füge install.packages/libraries hinzu).",
        "generate_code_description_user_prompt": \
            """
                Du erhältst eine Zusammenfassung des Datensatzes, eine Erklärung aller Spalten und das umzusetzende Visualisierungsziel.
                Deine Aufgabe ist es, basierend darauf Code zu generieren, der eine explorative Datenanalyse (EDA) durchführt und jeweils EINE passende Visualisierungen für das Visualisierungsziel erstellt.
                Das Visualisierungsziel muss immer umgesetzt werden.
                
                Zusammenfassung des Datensatzes und Erklärung der Spalten:
                '{summary}'
                
                Das Visualisierungsziel, welches umgesetzt werden sollen:
                '{visualization_goal}' 
            """,
        "generate_python_code": \
            """
                Erzeuge mir basierend auf den vorherigen Informationen Python-Code,
                der eine explorative Datenanalyse (EDA) des Datensatzes durchführt und das Visualierungsziel umsetzt.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann mit folgendem Befehl geladen werden:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Vorgaben für den Code:
                - Verwende ausschließlich `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, `basemap` und `plotly`.
                - Der Code direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Wähle Diagrammtypen basierend auf die Hinweise im Visualisierungsziel. 
                - Führe auch kurze statistische Analysen durch, wenn diese benötigt werden um das Visualisierungsziel zu erreichen. Beispiele für kurze, sinnvolle Analysen:
                  - Anteil fehlender Werte je Spalte,
                  - Korrelationen numerischer Variablen,
                  - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
                - Verwende Farben, Beschriftungen und Titel sinnvoll:
                  - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
                  - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
                  - Achsen in SI-Einheiten oder sinnvollen Skalen beschriften.
                - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
                - Der zurückgegebene Code soll in UTF-8 kodiert sein.

                Zum besseren Verständnis:
                Das ist das Ergebnis von `df.head(10)` auf den Datensatz:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Erzeuge mir ein R-Skript, das eine explorative Datenanalyse (EDA) des Datensatzes durchführt und das Visualisierungsziel umsetzt.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann aus der folgenden Datei geladen werden:
                - Pfad zur Datei: `'{dataset_path}'`
                - Trennzeichen: `'{dataset_sep}'`

                Vorgaben für den Code:
                - Der Code soll direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Visualisierungen sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Wähle Diagrammtypen entsprechend der Hinweise im Visualisierungsziel.
                - Führe auch kurze statistische Analysen durch, falls diese benötigt werden um das Visualisierungsziel zu erreichen. Beispiele für kurze, sinnvolle Analysen:
                  - Anteil fehlender Werte je Spalte,
                  - Korrelationen numerischer Variablen,
                  - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
                - Verwende Farben, Beschriftungen und Titel sinnvoll:
                  - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
                  - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
                  - Achsen in Einheiten oder sinnvollen Skalen beschriften.
                - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
                - Der zurückgegebene Code soll in UTF-8 kodiert sein.

                Zum besseren Verständnis:  
                Das ist das Ergebnis von `df.head(10)` auf den Datensatz:
                '{df_head_markdown}'
            """,
    },
    # TODO: wir müssen, encoding und seperator direkt beim load_data in den agentstate packen
    en={
        "generate_code_system_prompt": \
            """
                You are an expert in data science, {programming_language} programming, and data visualization.
                {library_instruction}
                Your main task is to generate code to implement and visualize a visualization goal for the dataset.
                Observe the following principles:
                1. Always write correct and executable {programming_language} code.
                2. When creating or evaluating visualizations, consider the quality criteria for good data visualization:
                   - Appropriateness of the chart type,
                   - Clarity and readability,
                   - Data fidelity,
                   - Aesthetic design,
                   - Technical correctness,
                   - Communication effectiveness,
                   - Constructive improvement suggestions
                3. You must not generate or reproduce any confidential or copyrighted data.
                4. All responses shall contain UTF-8 compatible {programming_language} code.
                When generating code, it should be immediately executable.
                5. FOR EACH VISUALIZATION GOAL, A SEPARATE SCRIPT shall be created and MAXIMALLY ONE VISUALIZATION PER SCRIPT.
                6. EACH SCRIPT shall save the visualization as a PNG file in the output directory.
            """,
        "generate_code_python_lib_instruction": \
            "Use only the following libraries: pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap, and plotly.",
        "generate_code_r_lib_instruction": \
            "Install and load all required packages at the beginning of the script (add install.packages/libraries).",
        "generate_code_description_user_prompt": \
            """
                You receive a summary of the dataset, an explanation of all columns, and the visualization goal to implement.
                Your task is to generate code based on this that performs an exploratory data analysis (EDA) and creates ONE suitable visualization for each visualization goal.
                The visualization goal must always be implemented.

                Summary of the dataset and explanation of the columns:
                '{summary}'

                The visualization goals to implement:
                '{visualization_goal}' 
            """,
        "generate_python_code": \
            """
                Generate {programming_language} code based on the previous information,
                that performs an exploratory data analysis (EDA) of the dataset and implements the visualization goal.
                Use the information from the previous message for this purpose.

                The dataset can be loaded using the following command:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Requirements for the code:
                - Use only `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, `basemap`, and `plotly`.
                - The code must be directly executable without syntax errors.
                - All charts shall be visually appealing, well-labeled (in German), readable, and saved as PNG files at:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Choose chart types based on the hints in the visualization goal.
                - Also perform short statistical analyses if needed to achieve the visualization goal. Examples of short, meaningful analyses:
                  - Proportion of missing values per column,
                  - Correlations of numerical variables,
                  - Overview tables of central measures (mean, standard deviation, etc.).
                - Use colors, labels, and titles meaningfully:
                  - Titles should describe what is shown (in German),
                  - Legends and axis labels should not cut off information,
                  - Label axes in SI units or meaningful scales.
                - All calculations and plots shall consider and account for missing values as well as string and Boolean values in the dataset.
                - The returned code shall be UTF-8 encoded.

                For better understanding:
                This is the result of `df.head(10)` on the dataset:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Generate an R script that performs an exploratory data analysis (EDA) of the dataset and implements the visualization goal.
                Use the information from the previous message for this purpose.

                The dataset can be loaded from the following file:
                - File path: `'{dataset_path}'`
                - Separator: `'{dataset_sep}'`

                Requirements for the code:
                - The code shall be directly executable without syntax errors.
                - All visualizations shall be visually appealing, well-labeled (in German), readable, and saved as PNG files at:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Choose chart types according to the hints in the visualization goal.
                - Also perform short statistical analyses if needed to achieve the visualization goal. Examples of short, meaningful analyses:
                  - Proportion of missing values per column,
                  - Correlations of numerical variables,
                  - Overview tables of central measures (mean, standard deviation, etc.).
                - Use colors, labels, and titles meaningfully:
                  - Titles should describe what is shown (in German),
                  - Legends and axis labels should not cut off information,
                  - Label axes in units or meaningful scales.
                - All calculations and plots shall consider and account for missing values as well as string and Boolean values in the dataset.
                - The returned code shall be UTF-8 encoded.

                For better understanding:  
                This is the result of `df.head(10)` on the dataset:
                '{df_head_markdown}'
            """,
    }
)

Goal = import_language_dto(AGENT_LANGUAGE, GoalBase)
Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_generate_python_code(state: AgentState) -> AgentState:
    """Generates Python code for data visualization."""

    for index, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        description_user_message, temp_agent = _get_generate_code_agent(state, vis.goal)

        code_user_message = prompt.get_prompt(
            AGENT_LANGUAGE,
            "generate_python_code",
            dataset_path=state["dataset_path"],
            dataset_sep=";",
            df_head_markdown=str(state["dataset_df"].head(10).to_markdown()),
            output_path=state["output_path"],
            goal_index=index
        )

        code_user_message = HumanMessage(content=code_user_message)

        messages = [description_user_message, code_user_message]
        llm_response = temp_agent.invoke({"messages": messages})

        code: Code = llm_response["structured_response"]
        vis.code = code

        print_color(f"LLM generated visualization code: ", Color.OK_GREEN)
        print_color(vis.code.code, Color.OK_BLUE)
        print_color(f"visualization goal: ", Color.OK_GREEN)
        print_color(vis.goal.question, Color.OK_BLUE)

        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name)
                )
                break

    return state


def _get_generate_code_agent(state: AgentState, goal: Goal):
    """Helper-function to create the code generation agent and description message."""
    programming_language = state["programming_language"]

    # get programming language specific prompt
    lib_instruction_key = (
        "generate_code_python_lib_instruction"
        if programming_language == ProgrammingLanguage.PYTHON
        else "generate_code_r_lib_instruction"
    )
    library_instruction = prompt.get_prompt(AGENT_LANGUAGE, lib_instruction_key)

    # get system prompt
    system_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "generate_code_system_prompt",
        programming_language=programming_language.value,
        library_instruction=library_instruction
    )

    # create the agent
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code,
        system_prompt=system_prompt
    )

    # load the instructions
    description_user_message = prompt.get_prompt(
        AGENT_LANGUAGE,
        "generate_code_description_user_prompt",
        summary=str(getattr(state.get("summary", None), "summary", "")),
        visualization_goal=goal
    )
    description_user_message = HumanMessage(content=description_user_message)

    # clear output directory before generating new plots / code
    clear_output_dir(state["output_path"])
    return description_user_message, temp_agent


@track_duration
def llm_generate_r_code(state: AgentState) -> AgentState:
    """Generates R code for data visualization."""

    for index, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        description_user_message, temp_agent = _get_generate_code_agent(state, vis.goal)

        code_user_message = prompt.get_prompt(
            AGENT_LANGUAGE,
            "generate_r_code",
            dataset_path=state["dataset_path"],
            dataset_sep=";",
            df_head_markdown=str(state["dataset_df"].head(10).to_markdown()),
            output_path=state["output_path"],
            goal_index=index
        )

        code_user_message = HumanMessage(content=code_user_message)

        messages = [description_user_message, code_user_message]
        llm_response = temp_agent.invoke({"messages": messages})

        code: Code = llm_response["structured_response"]
        vis.code = CodeWrapper(
            code=code.code,
            std_out=None,
            std_err=None,
            needs_regeneration=None,
            regeneration_attempts=None,
            refactoring_attempts=None,
            judge_result=None
        )
        print_color(f"LLM generated visualization code: ", Color.OK_GREEN)
        print_color(vis.code.code, Color.OK_BLUE)
        print_color(f"visualization goal: ", Color.OK_GREEN)
        print_color(vis.goal.question, Color.OK_BLUE)

        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
            break
    return state


def decide_programming_language(state: AgentState):
    """Decides the programming language, which should be used for code generation."""
    programming_language: ProgrammingLanguage = state["programming_language"]
    if programming_language == ProgrammingLanguage.PYTHON:
        return "python"
    elif programming_language == ProgrammingLanguage.R:
        return "r"
    else:
        raise ValueError("Unsupported programming language.")
