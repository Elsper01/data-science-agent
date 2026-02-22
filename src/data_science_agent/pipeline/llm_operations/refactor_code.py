import inspect
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from data_science_agent.dtos.base.responses.code_base import CodeBase
from data_science_agent.dtos.wrapper.visualization import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

prompt: Prompt = Prompt(
    de={
        "refactor_system_prompt": """
            Du bist ein erfahrener Entwickler und Code‑Refactoring‑Agent.
            Du erhältst Bewertungen (Verdicts) vom Judge‑Agenten und sollst
            den betroffenen Code überarbeiten. Gib am Ende den vollständigen,
            überarbeiteten Code zurück und kommentiere Änderungen kurz.
        """,
        "refactor_user_prompt": """
            Hier ist der aktuelle Code, der überarbeitet werden soll:

            --- CODE START ---
            {code}
            --- CODE END ---

            Hier sind die vom Judge‑Agenten zurückgegebenen Bewertungen (Verdicts):
            {judge_messages}

            Bitte überarbeite den Code basierend auf diesen Bewertungen
            und liefere den vollständig überarbeiteten Quellcode zurück.
        """
    },
    en={
        "refactor_system_prompt": """
            You are an experienced developer and code refactoring agent.
            You receive evaluations (verdicts) from the Judge agent and should
            revise the affected code. At the end, return the complete,
            revised code and briefly comment on the changes.
        """,
        "refactor_user_prompt": """
            Here is the current code to be refactored:

            --- CODE START ---
            {code}
            --- CODE END ---

            Here are the evaluations (verdicts) returned by the Judge agent:
            {judge_messages}

            Please refactor the code based on these evaluations
            and return the completely revised source code.
        """
    },
)

Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_refactor_visualizations(state: AgentState) -> AgentState:
    """Refactors all visualizations based on judge feedback."""

    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "refactor_system_prompt")

    base_agent = create_agent(
        model=get_llm_model(LLMModel.MISTRAL),
        system_prompt=SystemMessage(content=system_prompt),
        response_format=Code,
    )

    for vis in state["visualizations"]:
        vis: VisualizationWrapper
        instructions = prompt.get_prompt(
            AGENT_LANGUAGE,
            "refactor_user_prompt",
            code=vis.code.code,
            judge_messages=vis.code.judge_result
        )

        llm_response = base_agent.invoke({"messages": [HumanMessage(content=instructions)]})
        code: Code = llm_response["structured_response"]

        # TODO: speichere ab wie oft eine abbildung refactored werden musste
        vis.code.code = code.code
        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
                break

    state["is_refactoring"] = True
    state["regeneration_attempts"] = 0

    return state