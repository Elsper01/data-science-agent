import os

from data_science_agent.dtos.base import VisualizationContainerBase
from data_science_agent.graph import AgentState
from data_science_agent.language import import_language_dto
from data_science_agent.utils import print_color, AGENT_LANGUAGE, LLMMetadata
from data_science_agent.utils.enums import Color

VisualizationContainer = import_language_dto(AGENT_LANGUAGE, VisualizationContainerBase)

def generate_and_write_code(state: AgentState, temp_agent, messages, calling_method_name: str) -> AgentState:
    """Helper function to generate code and write it to a file."""
    llm_response = temp_agent.invoke({"messages": messages})
    file_name = "generated_plots"

    # with open(state["script_path"], "w", encoding="UTF-8") as file:
    #     print_color(f"LLM regenerated code: ", Color.HEADER)
    #     visualization_container:
    #     state["code"] = code
    #     file.write(code.code)
    vis_container: VisualizationContainer = llm_response["structured_response"]
    for vis in vis_container.visualizations:
        print_color(f"LLM generated visualization code: ", Color.OK_GREEN)
        print_color(vis.code.code, Color.OK_BLUE)
        print_color(f"visualization goal: ", Color.OK_GREEN)
        print_color(vis.goal.question, Color.OK_BLUE)
    state["visualizations"] = vis_container
    state["messages"] = llm_response["messages"]
    state["llm_metadata"].append(LLMMetadata.from_ai_message(llm_response["messages"][-1], calling_method_name))
    return state
