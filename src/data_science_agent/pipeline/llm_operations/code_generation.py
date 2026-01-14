import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from data_science_agent.dtos.base.responses.visualization_container_base import VisualizationContainerBase
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
                Deine Hauptaufgabe ist Code zu Generieren, um die Visualierungsziele für den Datensatz zu berechnen und zu visualisieren.:
                1. Schreibe stets korrekten und ausführungssicheren {programming_language}‑Code.
                2. Beachte bei der Erstellung oder Bewertung von Visualisierungen die Qualitätskriterien für gute Datenvisualisierung:
                   - Angemessenheit des Diagrammtyps,
                   - Klarheit und Lesbarkeit,
                   - Daten-Treue,
                   - Ästhetische Gestaltung,
                   - Technische Korrektheit,
                   - Effektivität der Kommunikation,
                   - Konstruktive Verbesserungsvorschläge.
                3. Du darfst keinerlei vertrauliche oder urheberrechtlich geschützte Daten erzeugen oder wiedergeben.
                4. Alle Antworten sollen UTF‑8‑kompatiblen {programming_language}‑Code enthalten.
                Wenn du Code generierst, soll dieser sofort lauffähig, sauber strukturiert, modular und kommentiert sein.
                5. PRO VISUALISIERUNGSZIEL soll EIN SEPARATES SKRIPT erstellt werden und PRO SKRIPT MAXIMAL EINE VISUALISIERUNG umgesetzt werden.
                6. JEDES SKRIPT soll die Visualisierung als PNG-Datei im Output Verzeichnis speichern.
            """,
        "generate_code_python_lib_instruction": \
            "Nutze ausschließlich die folgenden Bibliotheken: pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap.",
        "generate_code_r_lib_instruction": \
            "Installiere und lade alle benötigten Pakete am Anfang des Skripts (füge install.packages/libraries hinzu).",
        "generate_code_description_user_prompt": \
            """
                Du erhältst eine Zusammenfassung des Datensatzes, eine Erklärung aller Spalten und eine Liste von Visualisierungszielen.
                Deine Aufgabe ist es, basierend darauf Code zu generieren, der eine explorative Datenanalyse (EDA) durchführt und jeweils EINE passende Visualisierungen PRO Visualisierungsziel erstellt.
                Es soll für jedes Visualisierungsziel ein separates Skript erstellt werden.
                ALLE Visualisierungsziele sollen umgesetzt werden.
                
                Beschreibung bzw. Zusammenfassung des Datensatzes:
                '{summary}'
                
                Beschreibung der relevanten Spalten:
                '{columns}'
                
                Liste der Visualisierungsziele, welche umgesetzt werden sollen:
                '{visualization_goals}' 
            """,
        "generate_python_code": \
            """
                Erzeuge mir basierend auf den vorherigen Informationen Python-Code,
                der eine explorative Datenanalyse (EDA) des Datensatzes durchführt und die Visualierungsziele umsetzt.
                Es soll pro Visualisierungsziel ein separates Skript erstellt werden.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann mit folgendem Befehl geladen werden:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Vorgaben für den Code:
                - Verwende ausschließlich `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, `basemap`.
                - Der Code soll modular, gut kommentiert und direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `{output_path}<plot_name>.png`
                - Wähle Diagrammtypen basierend auf die Hinweise im Visualisierungsziel. 
                - Führe auch kurze statistische Analysen durch, wenn diese benötigt werden um das Visualisierungsziel zu erreichen. Beispiele für kurze, sinnvolle Analysen:
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
                Das ist das Ergebnis von `df.head(10)` auf den Datensatz:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Erzeuge mir ein R-Skript, das eine explorative Datenanalyse (EDA) des Datensatzes durchführt und die Visualisierungsziele umsetzt.
                Verwende hierfür die Informationen aus der vorherigen Nachricht.

                Der Datensatz kann aus der folgenden Datei geladen werden:
                - Pfad zur Datei: `'{dataset_path}'`
                - Trennzeichen: `'{dataset_sep}'`

                Vorgaben für den Code:
                - Der Code soll modular und direkt ausführbar sein, ohne syntaktische Fehler.
                - Alle Diagramme sollen optisch ansprechend, gut beschriftet (in Deutsch), lesbar und in PNG-Dateien gespeichert werden unter:
                  `{output_path}<goal_index>_<plot_name>.png`
                - Wähle Diagrammtypen entsprechend der Hinweise im Visualisierungsziel.
                - Führe auch kurze statistische Analysen durch, falls diese benötigt werden im das Visualisierungsziel zu erreichen. Beispiele für kurze, sinnvolle Analysen:
                  - Anteil fehlender Werte je Spalte,
                  - Korrelationen numerischer Variablen,
                  - Übersichtstabellen zu zentralen Kennwerten (Mittelwert, Standardabweichung etc.).
                - Verwende Farben, Beschriftungen und Titel sinnvoll:
                  - Titel sollen beschreiben, was gezeigt wird (auf Deutsch),
                  - Legenden und Achsenbeschriftungen sollen keine Information abschneiden,
                  - Achsen in Einheiten oder sinnvollen Skalen beschriften.
                - Es soll bei allen Berechnungen und Plots beachtet und berücksichtigt werden, dass fehlende Werte und auch String und Boolean Werte im Datensatz vorhanden sind. Also entsprechend damit umgehen.
                - Der zurückgegebene Code soll bitte in UTF-8 kodiert sein.
                - Für jede Visualisierung soll eine separate Methode erstellt werden, welche eine passende Fehlerbehandlung hat. Die Fehler Meldung soll ausgegeben werden, aber die Ausführung des restlichen Codes soll nicht abgebrochen werden.

                Zum besseren Verständnis:  
                Das ist das Ergebnis von `df.head(10)` auf den Datensatz:
                '{df_head_markdown}'
            """,
    },
    en={  # TODO: wir müssen, encoding und seperator direkt beim load_data in den agentstate packen
        # TODO: die deutschen prompts wurden alle angepasst, die englischen müssen noch angepasst werden
        "generate_python_code": \
            """
                Based on the previous summary and the data structure, generate Python code
                that performs an exploratory data analysis (EDA) of the dataset and produces suitable visualizations.

                The data can be loaded with:
                `df = pd.read_csv("'{dataset_path}'", sep="'{dataset_sep}'")`

                Code requirements:
                - Use only `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `geopandas`, and `basemap`.
                - The code must be modular, well‑commented, directly executable, and free of syntax errors.
                - All plots must be visually appealing, clearly labeled (in English), readable, and saved as PNG files under:
                  `{output_path}<plot_name>.png`
                - Choose chart types based on data meaning:
                  - Geographic variables → spatial distribution (e.g., map with point markers)
                  - Temporal variables → investigate whether temporal trends relate to other variables
                  - Numerical variables → histograms, boxplots, and scatterplots
                  - Categorical variables → bar plots of frequency distributions (top-10 for long lists if needed)
                - Include brief statistical analyses when appropriate, possibly with visualizations:
                  - proportion of missing values per column
                  - correlations of numerical variables
                  - summary tables for key metrics (mean, std-dev, etc.)
                - Use colors, labels, and titles meaningfully:
                  - Titles should describe what is shown (in English)
                  - Legends and axis labels must not truncate information
                  - Axes should be labeled in SI units or meaningful scales
                - Add concise explanatory comments about **why** certain plots are used.
                - Prioritize visualizations consistent with a data‑science workflow (data quality, distribution, relationships, geography, time).
                - Make sure missing values, strings, and booleans in the dataset are handled appropriately.
                - The returned code must be UTF‑8 encoded.
                - Create a separate function for each visualization and execute them at the end of the script within try/except blocks so that errors are logged but do not stop execution of the rest of the code.

                For reference, this is the output of `df.head(10)`:
                '{df_head_markdown}'
            """,
        "generate_r_code": \
            """
                Based on the analysis, generate an R script that performs an exploratory data analysis (EDA)
                of the dataset and produces meaningful visualizations.

                The dataset can be loaded from the following CSV:
                - File path: `'{dataset_path}'`
                - Separator: `'{dataset_sep}'`

                Code requirements:
                - The code must be modular, well‑commented, directly runnable, and free of syntax errors.
                - All plots must be visually appealing, clearly labeled (in English), readable, and saved as PNG files under:
                  `{output_path}<plot_name>.png`
                - Choose chart types according to the data’s meaning. The following hints may help, but apply your judgment:
                  - Geographic variables → spatial distribution (e.g., map with points)
                  - Temporal variables → study if variable values change over time
                  - Numerical variables → histograms, boxplots, scatterplots
                  - Categorical variables → bar charts for frequency distributions (top-10 when appropriate)
                - Add light statistical summaries if they make sense, optionally with visualization:
                  - proportion of missing values per column
                  - correlations between numeric variables
                  - summary tables of central measures (mean, standard deviation, etc.)
                - Use colors, labels, and titles sensibly:
                  - Titles should clearly state what is shown (in English)
                  - Legends and axes must not cut off information
                  - Axes in meaningful units or scales
                - Handle missing values, string, and boolean types gracefully in all computations and plots.
                - The returned code must be UTF‑8 encoded.
                - Each visualization should have its own function with proper error handling.
                  Errors should be printed but must not interrupt the execution of the rest of the script.

                For reference:  
                This is the output of `df.head(10)` on the dataset:
                '{df_head_markdown}'

                This is the result of the previous column‑wise analysis — use this information to decide which visualizations are meaningful:
                '{summary_columns}'

                This is a general description of the dataset:
                '{summary}'
            """,
        "generate_code_system_prompt": \
            """
                You are an expert in data science, {programming_language} programming and data visualization.
                {library_instruction}
                Your primary task is to produce code that computes and visualizes data‑science analyses for a dataset.
                General principles:
                1. Always write correct and executable {programming_language} code.
                2. Consider visualization quality: appropriate chart types, clarity, data‑fidelity, aesthetics, correctness, communication effectiveness, and constructive suggestions.
                3. Do not produce or reproduce confidential or copyrighted data.
                4. All answers should contain UTF‑8 compatible {programming_language} code.
                Generated code must be immediately runnable, well structured, modular and commented.
            """,
        "generate_code_python_lib_instruction": \
            "Use only the following libraries: pandas, numpy, matplotlib.pyplot, seaborn, geopandas, basemap.",
        "generate_code_r_lib_instruction": \
            "Install and load required packages at the start of the script (include install.packages/library calls).",
        "generate_code_description_user_prompt": \
            """
                The following information was generated previously:
                Dataset description / summary:
                '{summary}'
                Relevant column descriptions:
                '{columns}'
            """,
    }
)

VisualizationContainer = import_language_dto(AGENT_LANGUAGE, VisualizationContainerBase)


@track_duration
def llm_generate_python_code(state: AgentState) -> AgentState:
    """Generates python code for data visualization."""
    # TODO: bis jetzt wurde nur der r code funktion angepasst, diese muss noch nachgezogen werden
    description_user_message, temp_agent = _get_generate_code_agent(state)

    code_user_message = prompt.get_prompt(
        AGENT_LANGUAGE,
        "generate_python_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head().to_markdown()),
        output_path=state["output_path"]
    )

    code_user_message = HumanMessage(content=code_user_message)

    messages = [description_user_message, code_user_message]

    llm_response = temp_agent.invoke({"messages": messages})

    vis_container: VisualizationContainer = llm_response["structured_response"]

    for vis in vis_container.visualizations:
        print_color(f"LLM generated visualization code: ", Color.OK_GREEN)
        print_color(vis.code.code, Color.OK_BLUE)
        print_color(f"visualization goal: ", Color.OK_GREEN)
        print_color(vis.goal.question, Color.OK_BLUE)

    state["visualizations"] = vis_container

    state["llm_metadata"].append(
        LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))
    return state


def _get_generate_code_agent(state: AgentState):
    """Helper-function to create the code generation agent and description message."""
    programming_language = state["programming_language"]

    # programming language specific prompts
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

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=VisualizationContainer,
        system_prompt=system_prompt
    )

    description_user_message = prompt.get_prompt(
        AGENT_LANGUAGE,
        "generate_code_description_user_prompt",
        summary=str(getattr(state.get("summary", None), "summary", "")),
        columns=str(getattr(state.get("summary", None), "columns", "")),
        visualization_goals=str(getattr(state.get("goals", None), "goals", ""))
    )
    description_user_message = HumanMessage(content=description_user_message)

    # clear output directory before generating new plots / code
    clear_output_dir(state["output_path"])
    return description_user_message, temp_agent


@track_duration
def llm_generate_r_code(state: AgentState) -> AgentState:
    """Generates R code for data visualization."""
    description_user_message, temp_agent = _get_generate_code_agent(state)

    code_user_message = prompt.get_prompt(
        AGENT_LANGUAGE,
        "generate_r_code",
        dataset_path=state["dataset_path"],
        dataset_sep=";",
        df_head_markdown=str(state["dataset_df"].head(10).to_markdown()),
        output_path=state["output_path"]
    )

    code_user_message = HumanMessage(content=code_user_message)

    messages = [description_user_message, code_user_message]
    llm_response = temp_agent.invoke({"messages": messages})

    vis_container: VisualizationContainer = llm_response["structured_response"]

    for vis in vis_container.visualizations:
        print_color(f"LLM generated visualization code: ", Color.OK_GREEN)
        print_color(vis.code.code, Color.OK_BLUE)
        print_color(f"visualization goal: ", Color.OK_GREEN)
        print_color(vis.goal.question, Color.OK_BLUE)

    state["visualizations"] = vis_container

    state["llm_metadata"].append(
        LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))
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
