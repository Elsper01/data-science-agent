from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from data_science_agent.dtos.base import JudgeBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color
from data_science_agent.utils.enums import LLMModel, Color

prompt: Prompt = Prompt(
    de={
        "judge_system_prompt": \
            """
                Du bist eine Expertin bzw. ein Experte für Datenvisualisierung und analytische Kommunikation.
                Deine Aufgabe ist es, Code zu überprüfen und zu bewerten, der Diagramme oder andere Visualisierungen erzeugt.
                Erstelle eine detaillierte Kritik auf Grundlage von Angemessenheit, Klarheit, Treue zu den Daten, Ästhetik, technischer Korrektheit und Verbesserungsvorschlägen.
            """,
        "judge_user_prompt": \
            """
                Mit dem folgenden Code wurden Diagramme zur Visualisierung eines Datensatzes erzeugt.
                Bitte bewerte die erzeugten Visualisierungen anhand der genannten Kriterien und gib eine ausführliche Kritik ab.
                Generierter Code:
                {code}
            """
    },
    en={
        "judge_system_prompt": \
            """
                You are an expert in data visualization and analytical communication.
                Your task is to review and evaluate code that produces charts or other visualizations,
                producing a detailed critique based on appropriateness, clarity, data-fidelity, aesthetics, technical correctness and improvement suggestions.
            """,
        "judge_user_prompt": \
            """
                The following code was used to create visualizations of a dataset.
                Please evaluate the visualizations according to the criteria and provide a detailed critique.
                Generated code:
                {code}
            """,
    }
)

Judge = import_language_dto(AGENT_LANGUAGE, JudgeBase)


def llm_judge_code(state: AgentState) -> AgentState:
    """Ask the LLM to judge the generated code."""
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "judge_system_prompt")
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        system_prompt=SystemMessage(content=system_prompt),
        response_format=Judge
    )

    code_content = prompt.get_prompt(AGENT_LANGUAGE, "judge_user_prompt", code=state["code"].code)

    llm_response = temp_agent.invoke({"messages": [HumanMessage(content=code_content)]})
    judge_result: Judge = llm_response["structured_response"]
    state["judge_messages"] = judge_result.verdicts
    print_color(f"LLM Judge", Color.WARNING)
    for x in judge_result.verdicts:
        print(f"Figure: {x.figure_name}, File: {x.file_name}")
        print(f"Critic notes: {x.critic_notes}")
        print(f"Suggested code: {x.suggestion_code}")
        print(f"Needs regeneration: {x.needs_regeneration}")
        print(f"Can be deleted: {x.can_be_deleted}")
        print("-----")

    # TODO: step two - we judge the generated plots as well -> das lagern wir direkt in einen evaluate agent aus, der uns alle erzeugten plots bewertet
    return state
