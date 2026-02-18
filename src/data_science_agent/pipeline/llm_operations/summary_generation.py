import inspect
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from assert_llm_tools import evaluate_summary, LLMConfig

from data_science_agent.dtos.base.responses.summary_base import SummaryBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, OPENROUTER_API_KEY, BASE_URL
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
                - Dataset: '{dataset}'
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
                You are a data science expert and analyze a tabular dataset along with its metadata.
                All statements should strictly relate to the dataset. Important are anomalies, trends, missing values, data types, correlations.
                The response format should be scientific. Speak understandably for laypeople, but still informative and behave as an expert.
                Even an expert should benefit from the provided data and information.
                Be certain about all statements and always refer to the dataset or other information provided by the prompt.
        
                The following data shall assist in the analysis:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
                - Metadata: '{metadata}'
                - Dataset: '{dataset}'
            """,
        "summary_user_prompt": \
            """
                Summarize the dataset appropriately for me (expert).
                As a result, I want a summary or explanation of the dataset and an explanation/description of each column of the dataset.
                The language used should be English.
                For me, visualizations are especially important that help me understand and analyze the dataset.
            """
    }
)

Summary = import_language_dto(AGENT_LANGUAGE, SummaryBase)


@track_duration
def llm_generate_summary(state: AgentState) -> AgentState:
    """This node generates the dataset summary by using a LLM."""

    llm_response = _get_agent_and_messages(state)

    for message in reversed(llm_response["messages"]):
        if isinstance(message, AIMessage):
            state["llm_metadata"].append(
                LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name)
            )
            break

    summary: Summary = llm_response["structured_response"]
    print_color(f"Generated Dataset Summary.", Color.OK_GREEN)

    state["summary"] = summary

    # TODO: das kann später wieder raus; wir überlegen uns noch ob das sinnvoll ist
    output_path = state["output_path"]
    summary_file_path = os.path.join(output_path, "summary.txt")

    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(str(summary))

    print_color(f"Summary saved to: {summary_file_path}", Color.OK_BLUE)

    # evaluate(state.get("dataset_df", []).to_markdown() ,summary)

    return state


def _get_agent_and_messages(state: AgentState) -> dict[str, Any] | Any:
    system_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_system_prompt",
        column_names=str(state.get("column_names", [])),
        descriptions=str(state.get("descriptions", [])),
        metadata=str(state.get("metadata", [])),
        dataset=str(state.get("dataset_df", []).to_markdown()),
    )

    user_content = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_user_prompt"
    )
    user_msg = HumanMessage(content=user_content)

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GEMINI),
        response_format=Summary,
        system_prompt=system_prompt
    )

    llm_response = temp_agent.invoke({"messages": [user_msg]})
    return llm_response

def evaluate(table, summary):
    # TODO: das kann raus; wir können diese LLM-as-a-Judge's auch selber bauen und integrieren
    # Configure LLM for evaluation
    config = LLMConfig(
        provider="openai",
        model_id="entwicklung",#LLMModel.GPT_4o,
        api_key=OPENROUTER_API_KEY,
        proxy_url="https://ki-api.scw.ext.seitenbau.net/v1",#BASE_URL
    )

    # Evaluate the summary
    results = evaluate_summary(
        full_text=table,
        summary=summary.summary,
        metrics=["rouge", "bleu", "bert_score", "bart_score", "faithfulness", "hallucination"],
        llm_config=config
    )

    print(results)
