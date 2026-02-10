import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from data_science_agent.dtos.base.responses.judge_base import JudgeBase
from data_science_agent.dtos.base.responses.judge_verdict_base import JudgeVerdictBase
from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

prompt: Prompt = Prompt(
    de={
        "judge_system_prompt":
            """
                Du bist eine Expertin bzw. ein Experte für Datenvisualisierung und analytische Kommunikation.
                Deine Aufgabe ist es, Code zu überprüfen und zu bewerten, der einzelne Diagramme oder Visualisierungen erzeugt.
                Erstelle für JEDE Visualisierung eine detaillierte Kritik auf Grundlage von Angemessenheit, Klarheit, Treue zu den Daten, Ästhetik, technischer Korrektheit und Verbesserungsvorschlägen.
                Bewerte ob der Code das angegebene Visualisierungsziel korrekt umsetzt.
            """,
        "judge_user_prompt":
            """
                Mit dem folgenden Code wurde eine Visualisierung für das angegebene Ziel erzeugt.
                Bitte bewerte diese Visualisierung anhand der genannten Kriterien und gib eine ausführliche Kritik ab.

                VISUALISIERUNGSZIEL:
                {goal_description}

                Generierter Code:
                {code}
            """
    },
    en={
        "judge_system_prompt":
            """
                You are an expert in data visualization and analytical communication.
                Your task is to review and evaluate code that generates individual diagrams or visualizations.
                Create a detailed critique for EACH visualization based on appropriateness, clarity, fidelity to the data, aesthetics, technical correctness, and improvement suggestions.
                Evaluate whether the code correctly implements the specified visualization goal.
            """,
        "judge_user_prompt":
            """
                The following code was used to create a visualization for the specified goal.
                Please evaluate this visualization based on the mentioned criteria and provide a detailed critique.

                VISUALIZATION GOAL:
                {goal_description}

                Generated Code:
                {code}
            """
    },
)

JudgeVerdict = import_language_dto(AGENT_LANGUAGE, JudgeVerdictBase)
Judge = import_language_dto(AGENT_LANGUAGE, JudgeBase)


@track_duration
def llm_judge_code(state: AgentState) -> AgentState:
    """Ask the LLM to judge the generated code for each visualization."""
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "judge_system_prompt")

    for i, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper

        temp_agent = create_agent(
            model=get_llm_model(LLMModel.MISTRAL),
            system_prompt=SystemMessage(content=system_prompt),
            response_format=JudgeVerdict
        )

        code_content = prompt.get_prompt(
            AGENT_LANGUAGE,
            "judge_user_prompt",
            goal_description=vis.goal,
            code=vis.code.code
        )

        llm_response = temp_agent.invoke({"messages": [HumanMessage(content=code_content)]})
        judge_result: JudgeVerdict = llm_response["structured_response"]

        vis.code.judge_result = JudgeVerdict(
            figure_name=judge_result.figure_name,
            file_name=judge_result.file_name,
            critic_notes=judge_result.critic_notes,
            suggestion_code=judge_result.suggestion_code,
            needs_regeneration=judge_result.needs_regeneration,
            can_be_deleted=judge_result.can_be_deleted
        )

        print_color(f"LLM Judge for vis#{i}", Color.WARNING)
        print(f"Figure: {judge_result.figure_name}, File: {judge_result.file_name}")
        print(f"Critic notes: {judge_result.critic_notes}")
        print(f"Suggested code: {judge_result.suggestion_code}")
        print(f"Needs regeneration: {judge_result.needs_regeneration}")
        print(f"Can be deleted: {judge_result.can_be_deleted}")
        print("-----")

        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
                break

    return state