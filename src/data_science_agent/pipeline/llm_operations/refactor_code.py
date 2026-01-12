import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from data_science_agent.dtos.base.responses.code_base import CodeBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

prompt: Prompt = Prompt(
    de={
        "refactor_system_prompt": \
            """
                Du bist ein erfahrener Entwickler und Code-Refactoring-Agent.
                Du erhältst Bewertungen (Verdicts) vom Judge-Agenten und sollst den betroffenen Code überarbeiten.
                Gib am Ende den vollständigen, überarbeiteten Code zurück und kommentiere Änderungen kurz.
            """,
        "refactor_user_prompt": \
            """
                Hier ist der aktuelle Code, der überarbeitet werden soll:

                --- CODE START ---
                {code}
                --- CODE END ---

                Hier sind die vom Judge-Agenten zurückgegebenen Bewertungen (Verdicts):
                {judge_messages}

                Bitte überarbeite den Code basierend auf diesen Bewertungen und liefere den vollständig überarbeiteten Quellcode zurück.
            """
    },
    en={
        "refactor_system_prompt": \
            """
                You are an experienced developer and code-refactoring agent.
                You receive judgments from a Judge agent and should refactor the affected code accordingly.
                Return the full, refactored code and briefly comment changes.
            """,
        "refactor_user_prompt": \
            """
                Here is the current code to be refactored:
    
                --- CODE START ---
                {code}
                --- CODE END ---
    
                Here are the verdicts from the Judge agent:
                {judge_messages}
    
                Please refactor the code based on these verdicts and return the full source code.
            """
    }
)

Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_refactor_plots(state: AgentState) -> AgentState:
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "refactor_system_prompt")
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        system_prompt=SystemMessage(content=system_prompt),
        response_format=Code
    )

    instructions = prompt.get_prompt(
        AGENT_LANGUAGE,
        "refactor_user_prompt",
        code=state["code"].code,
        judge_messages=str(state.get("judge_messages", []))
    )

    # TODO: refactoring -> da can write_code_to_file util methode verwendet werden
    llm_response = temp_agent.invoke({"messages": [HumanMessage(content=instructions)]})
    code: Code = llm_response["structured_response"]
    state["code"] = code

    # reset the regeneration attempts
    state["regeneration_attempts"] = 0

    # set flag for decide_regenerate_code to determine if agent is allowed to stop
    state["is_refactoring"] = True

    state["llm_metadata"].append(
        LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))

    print_color(f"LLM Refactor", Color.WARNING)
    with open(state["script_path"], "w", encoding="UTF-8") as file:
        file.write(code.code)
    return state
