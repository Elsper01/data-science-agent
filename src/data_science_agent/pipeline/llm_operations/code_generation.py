import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.dtos.base import GoalBase, CodeBase
from data_science_agent.dtos.wrapper.code import CodeWrapper
from data_science_agent.dtos.wrapper.visualization import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, LLMMetadata, print_color
from data_science_agent.utils.enums import LLMModel, ProgrammingLanguage, Color
from data_science_agent.utils.pipeline import clear_output_dir, archive_images

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
                7. Gib nur dann Text auf die Systemausgabe aus, wenn es wirklich notwendig ist, z. B. wenn ein Fehler vorliegt.
                8. Verwende keine Kommentare und dokumentiere den Code nicht.
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
                Erzeuge mir ein Python-Skript, das eine explorative Datenanalyse (EDA) des Datensatzes durchführt und das Visualisierungsziel umsetzt.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann aus der folgenden Datei geladen werden:
                - Pfad zur Datei: `'{dataset_path}'`
                - Trennzeichen: `'{dataset_sep}'`
                - Encoding: `'{dataset_encoding}'`

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
        "generate_r_code": \
            """
                Erzeuge mir ein R-Skript, das eine explorative Datenanalyse (EDA) des Datensatzes durchführt und das Visualisierungsziel umsetzt.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann aus der folgenden Datei geladen werden:
                - Pfad zur Datei: `'{dataset_path}'`
                - Trennzeichen: `'{dataset_sep}'`
                - Encoding: `'{dataset_encoding}'`

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
    en={
        "generate_code_system_prompt": \
            """
                You are an expert in data science, {programming_language} programming and data visualization.
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
                   - Effectiveness of communication,
                   - Constructive improvement suggestions
                3. Do not generate or reproduce any confidential or copyrighted data.
                4. All responses shall contain UTF-8 compatible {programming_language} code.
                When generating code, it should be immediately executable.
                5. ONE SEPARATE SCRIPT shall be created FOR EACH VISUALIZATION GOAL and MAXIMALLY ONE VISUALIZATION per script.
                6. EACH SCRIPT shall save the visualization as a PNG file in the output directory.
                7. Only output text to stdout if absolutely necessary, e.g., if an error occurs.
                8. Do not use comments and do not document the code.
            """,
        "generate_code_python_lib_instruction": \
            "Use only the following libraries: pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap and plotly.",
        "generate_code_r_lib_instruction": \
            "Install and load all required packages at the beginning of the script (add install.packages/libraries).",
        "generate_code_description_user_prompt": \
            """
                You receive a summary of the dataset, an explanation of all columns, and the visualization goal to be implemented.
                Your task is to generate code based on this information that performs an exploratory data analysis (EDA) and creates ONE appropriate visualization for the visualization goal.
                The visualization goal must always be implemented.

                Summary of the dataset and explanation of the columns:
                '{summary}'

                The visualization goal to be implemented:
                '{visualization_goal}'
            """,
        "generate_python_code": \
            """
                Generate a Python script that performs an exploratory data analysis (EDA) of the dataset and implements the visualization goal.
                Use the information from the previous message for this.

                The dataset can be loaded from the following file:
                - File path: `'{dataset_path}'`
                - Separator: `'{dataset_sep}'`
                - Encoding: `'{dataset_encoding}'`

                Specifications for the code:
                - The code should be directly executable without syntax errors.
                - All visualizations should be visually appealing, well labeled (in German), readable and saved as PNG files under:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Choose chart types according to the hints in the visualization goal.
                - Also perform short statistical analyses if needed to achieve the visualization goal. Examples of short, meaningful analyses:
                  - Proportion of missing values per column,
                  - Correlations of numerical variables,
                  - Overview tables of central measures (mean, standard deviation, etc.).
                - Use colors, labels and titles meaningfully:
                  - Titles should describe what is shown (in German),
                  - Legends and axis labels should not cut off any information,
                  - Label axes with units or appropriate scales.
                - It should be considered and taken into account in all calculations and plots that missing values as well as string and boolean values are present in the dataset. Handle them accordingly.
                - The returned code should be UTF-8 encoded.

                For better understanding:
                This is the result of `df.head(10)` on the dataset:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Generate an R script that performs an exploratory data analysis (EDA) of the dataset and implements the visualization goal.
                Use the information from the previous message for this.

                The dataset can be loaded from the following file:
                - File path: `'{dataset_path}'`
                - Separator: `'{dataset_sep}'`
                - Encoding: `'{dataset_encoding}'`

                Specifications for the code:
                - The code should be directly executable without syntax errors.
                - All visualizations should be visually appealing, well labeled (in German), readable and saved as PNG files under:
                  `{output_path}{goal_index}_<plot_name>.png`
                - Choose chart types according to the hints in the visualization goal.
                - Also perform short statistical analyses if needed to achieve the visualization goal. Examples of short, meaningful analyses:
                  - Proportion of missing values per column,
                  - Correlations of numerical variables,
                  - Overview tables of central measures (mean, standard deviation, etc.).
                - Use colors, labels and titles meaningfully:
                  - Titles should describe what is shown (in German),
                  - Legends and axis labels should not cut off any information,
                  - Label axes with units or appropriate scales.
                - It should be considered and taken into account in all calculations and plots that missing values as well as string and boolean values are present in the dataset. Handle them accordingly.
                - The returned code should be UTF-8 encoded.

                For better understanding:
                This is the result of `df.head(10)` on the dataset:
                '{df_head_markdown}'
            """,
    },
)

Goal = import_language_dto(AGENT_LANGUAGE, GoalBase)
Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_generate_python_code(state: AgentState) -> AgentState:
    """Generates Python code for data visualization."""
    archive_images(state["output_path"], state["regeneration_attempts"])
    clear_output_dir(state["output_path"])
    for index, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        description_user_message, temp_agent = _get_generate_code_agent(state, vis.goal)

        code_user_message = prompt.get_prompt(
            AGENT_LANGUAGE,
            "generate_python_code",
            dataset_path=state["dataset_path"],
            dataset_sep=state["dataset_delimiter"],
            dataset_encoding=state["dataset_encoding"],
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
        print_color(f"LLM generated visualization code.", Color.OK_GREEN)

        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
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
        model=get_llm_model(LLMModel.GEMINI),
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

    return description_user_message, temp_agent


@track_duration
def llm_generate_r_code(state: AgentState) -> AgentState:
    """Generates R code for data visualization."""
    archive_images(state["output_path"], state["regeneration_attempts"])
    clear_output_dir(state["output_path"])
    for index, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        description_user_message, temp_agent = _get_generate_code_agent(state, vis.goal)

        code_user_message = prompt.get_prompt(
            AGENT_LANGUAGE,
            "generate_r_code",
            dataset_path=state["dataset_path"],
            dataset_sep=state["dataset_delimiter"],
            dataset_encoding=state["dataset_encoding"],
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
        print_color(f"LLM generated visualization code.", Color.OK_GREEN)

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
