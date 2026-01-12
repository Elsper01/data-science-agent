import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from data_science_agent.dtos.base.responses.summary_base import SummaryBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color
from data_science_agent.utils.enums import LLMModel, Color
from data_science_agent.utils.llm_metadata import LLMMetadata

prompt: Prompt = Prompt(
    de={
        "summary_system_prompt": \
            """
                Du bist ein Data Science-Experte und analysierst einen tabellarische Datensatz zusammen mit seinen Metadaten.
                Alle Aussagen sollen sich strikt auf den Datensatz beziehen. Wichtig sind Auffälligkeiten, Trends, fehlende Werte, Datentypen, Zusammenhänge. 
                Antwortformat soll wissenschaftlich sein. Sprich für Laien verständlich, aber dennoch informativ und verhalte dich als Experte.
                Auch ein Experte soll aus den bereitgestellten Daten und Informationen einen Nutzen haben.
                Sei dir bei sämtlichen Aussagen sicher und beziehe dich immer auf den Datensatz oder andere durch den Prompt bereitgestellte Informationen.

                Folgende Daten sollen bei der Analyse helfen:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
                - Metadata: '{metadata}'
            """,
        "summary_user_prompt": \
            """
                Fasse mir (Experte) den Datensatz passend zusammen.
                Ich möchte als Ergebnis eine Zusammenfassung bzw. Erklärung des Datensatzes und eine Erklärung / Beschreibung der einzelnen Spalten des Datensatzes.
                Die verwendete Sprache soll Deutsch sein.
                Für mich sind vor allem Visualisierungen wichtig, die mir helfen den Datensatz zu verstehen und zu analysieren. 
            """
    },
    en={
        "summary_system_prompt": \
            """
                You are a data science expert helping me analyze a tabular dataset.
                Please provide a summary of the dataset: what it is about, any notable aspects, trends, or missing values. 
                The response should be easy to understand for general users,
                while still informative enough for an expert to judge whether the dataset is suitable for their work.

                Here is all relevant information:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
                - Metadata: '{metadata}'
            """,
        "summary_user_prompt": \
            """
                Please summarize the dataset appropriately.
                I want a summary or explanation of the dataset and a description of each column.
                The response language should be English.
            """,
    }
)

Summary = import_language_dto(AGENT_LANGUAGE, SummaryBase)


@track_duration
def llm_generate_summary(state: AgentState) -> AgentState:
    """This node generates the dataset summary by using a LLM."""

    system_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_system_prompt",
        column_names=str(state.get("column_names", [])),
        descriptions=str(state.get("descriptions", [])),
        metadata=str(state.get("metadata", []))
    )

    user_content = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_user_prompt"
    )
    user_msg = HumanMessage(content=user_content)

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.MISTRAL),
        response_format=Summary,
        system_prompt=system_prompt
    )

    llm_response = temp_agent.invoke({"messages": [user_msg]})

    state["llm_metadata"].append(
        LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))

    print_color(f"LLM result: ", Color.HEADER)
    summary: Summary = llm_response["structured_response"]
    print_color(f"Dataset Summary: ", Color.OK_GREEN)
    print(summary.summary)
    print_color(f"\nColumn Descriptions:", Color.OK_BLUE)
    print(summary.columns)

    state["summary"] = summary

    return state
