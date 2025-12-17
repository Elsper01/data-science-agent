import os

from data_science_agent.dtos.base import CodeBase
from data_science_agent.graph import AgentState
from data_science_agent.language import import_language_dto
from data_science_agent.utils import print_color, AGENT_LANGUAGE, LLMMetadata
from data_science_agent.utils.enums import Color

Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


def generate_and_write_code(state: AgentState, temp_agent, messages, calling_method_name: str) -> AgentState:
    """Helper function to generate code and write it to a file."""
    llm_response = temp_agent.invoke({"messages": messages})
    file_name = "generated_plots"

    state["script_path"] = os.path.join(state["output_path"], file_name + "." + state["programming_language"].value)
    with open(state["script_path"], "w", encoding="UTF-8") as file:
        print_color(f"LLM regenerated code: ", Color.HEADER)
        code: Code = llm_response["structured_response"]
        state["code"] = code
        file.write(code.code)
    state["messages"] = llm_response["messages"]
    state["llm_metadata"].append(LLMMetadata.from_ai_message(llm_response["messages"][-1], calling_method_name))
    return state
