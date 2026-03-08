import csv
import inspect
import os
from typing import Any

import nltk
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from rouge_score import rouge_scorer

from data_science_agent.dtos.base.responses.summary_base import SummaryBase
from data_science_agent.dtos.wrapper import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, DurationMetadata, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color, ProgrammingLanguage
from data_science_agent.utils.llm_metadata import LLMMetadata

nltk.download('wordnet')

prompt: Prompt = Prompt(
    de={
        "summary_system_prompt": \
            """
                Du bist ein Data Science-Experte und analysierst einen tabellarische Datensatz.
                Alle Aussagen sollen sich strikt auf den Datensatz beziehen. Wichtig sind Auffälligkeiten, Trends, fehlende Werte, Datentypen, Zusammenhänge. 
                Antwortformat soll wissenschaftlich sein. Sprich für Laien verständlich, aber dennoch informativ und verhalte dich als Experte.
                Auch ein Experte soll aus den bereitgestellten Daten und Informationen einen Nutzen haben.
                Sei dir bei sämtlichen Aussagen sicher und beziehe dich immer auf den Datensatz oder andere durch den Prompt bereitgestellte Informationen.

                Folgende Daten sollen bei der Analyse helfen:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
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
                You are a data science expert and analyze a tabular dataset.
                All statements should strictly relate to the dataset. Important are anomalies, trends, missing values, data types, correlations.
                The response format should be scientific. Speak understandably for laypeople, but still informative and behave as an expert.
                Even an expert should benefit from the provided data and information.
                Be certain about all statements and always refer to the dataset or other information provided by the prompt.

                The following data shall assist in the analysis:
                - Columns: '{column_names}'
                - Descriptions: '{descriptions}'
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

    gpt5   = _get_agent_and_messages(state, LLMModel.GPT_5)
    gpt4o  = _get_agent_and_messages(state, LLMModel.GPT_4o)
    gemini = _get_agent_and_messages(state, LLMModel.GEMINI)
    grok   = _get_agent_and_messages(state, LLMModel.GROK)
    claude4= _get_agent_and_messages(state, LLMModel.CLAUDE_4)

    results = []
    for model_name, response in [
        ("GPT-5",   gpt5["structured_response"]),
        ("GPT-4o",  gpt4o["structured_response"]),
        ("Gemini",  gemini["structured_response"]),
        ("Grok",    grok["structured_response"]),
        ("Claude",  claude4["structured_response"]),
    ]:
        scores = evaluate_summary_by_model(state, response, model_name)
        row = {"model": model_name}
        row.update(scores)
        results.append(row)

    output_path = state["output_path"]
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "summary_evaluation.csv")

    fieldnames = ["model", "bleu", "meteor", "rouge1", "rouge2", "rougeL"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print_color(f"Evaluation metrics saved to: {csv_path}", Color.OK_BLUE)

    llm_response = gemini
    for message in reversed(llm_response["messages"]):
        if isinstance(message, AIMessage):
            state["llm_metadata"].append(
                LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name)
            )
            break

    summary: Summary = llm_response["structured_response"]
    print_color("Generated Dataset Summary.", Color.OK_GREEN)
    state["summary"] = summary

    summary_file_path = os.path.join(output_path, "summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(str(summary))

    print_color(f"Summary saved to: {summary_file_path}", Color.OK_BLUE)
    print_color("Evaluation done.", Color.OK_BLUE)

    return state


def evaluate_summary_by_model(state: AgentState, summary: BaseModel, model_name:str) -> dict[str, float]:
    dataset_text = get_dataset_preview(state.get("dataset_df", []), 25).to_markdown()
    metadata_text = str(state.get("metadata", ""))
    descriptions_text = str(state.get("descriptions", ""))
    evaluation_scores = evaluate_summary(
        summary=summary,
        dataset_text=dataset_text,
        metadata=metadata_text,
        column_descriptions=descriptions_text,
    )
    print_color(f"  Evaluation Metrics {model_name}: {evaluation_scores}", Color.OK_CYAN)

    return evaluation_scores


def _get_agent_and_messages(state: AgentState, model: LLMModel) -> dict[str, Any] | Any:
    system_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_system_prompt",
        column_names=str(state.get("column_names", [])),
        descriptions=str(state.get("descriptions", [])),
        metadata=str(state.get("metadata", [])),
        dataset=str(get_dataset_preview(state["dataset_df"]).to_markdown())
    )

    user_content = prompt.get_prompt(
        AGENT_LANGUAGE,
        "summary_user_prompt"
    )
    user_msg = HumanMessage(content=user_content)

    temp_agent = create_agent(
        model=get_llm_model(model),
        response_format=Summary,
        system_prompt=system_prompt
    )

    llm_response = temp_agent.invoke({"messages": [user_msg]})
    return llm_response


def get_dataset_preview(df, n=25):
    """Robust sampling."""
    if df is None:
        return None
    sample_size = min(n, len(df))
    if sample_size == 0:
        return df
    return df.sample(n=sample_size, random_state=42, replace=False)


def evaluate_summary(summary: Summary, dataset_text: str, metadata: str, column_descriptions: str) -> dict[str, float]:
    """
    Evaluates the generated summary using ROUGE, BLEU and METEOR against a reference text constructed from the dataset,
    metadata and column descriptions.
    """

    reference_text = f"{column_descriptions}\n{metadata}\n{dataset_text}"
    summary_text = f"{summary.summary}\n{summary.columns}"

    reference_tokens = word_tokenize(reference_text)
    summary_tokens = word_tokenize(summary_text)

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], summary_tokens, smoothing_function=smoothie)

    meteor = meteor_score([reference_tokens], summary_tokens)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = rouge.score(reference_text, summary_text)
    rouge1 = scores["rouge1"].fmeasure
    rouge2 = scores["rouge2"].fmeasure
    rougeL = scores["rougeL"].fmeasure

    results = {
        "bleu": bleu_score,
        "meteor": meteor,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL
    }

    return results
