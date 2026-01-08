from langgraph.graph import StateGraph, START, END

from data_science_agent.graph import AgentState
from data_science_agent.pipeline import (
    load_dataset,
    analyse_dataset,
    load_metadata,
    analyse_metadata,
    llm_generate_summary,
    llm_generate_python_code,
    llm_generate_r_code,
    decide_programming_language,
    test_generated_code,
    decide_regenerate_code,
    llm_regenerate_code,
    llm_judge_code,
    llm_refactor_plots,
    llm_generate_goals,
)
from data_science_agent.utils import print_color
from data_science_agent.utils.enums import Color


def __print_statistics(state: AgentState) -> AgentState:
    """Prints the statistics of the graph."""
    print_color("Graph Duration Statistics:", Color.HEADER)
    for duration in state["durations"]:
        print_color(f"{duration.method_name}", Color.OK_BLUE)
        print_color(f"  Duration: {duration.get_total_duration()} seconds", Color.OK_GREEN)
    print_color("\n\nGraph LLM Cost & Token Statistics:", Color.HEADER)
    for metadata in state["llm_metadata"]:
        print_color(f"{metadata.method_name}", Color.OK_BLUE)
        print_color(f"  Token total: {metadata.token_usage.total_tokens} tokens", Color.OK_GREEN)
        print_color(f"  Costs total: {metadata.cost_details.total_cost} $", Color.OK_GREEN)
    return state

def build_graph():
    """Builds the state graph for the data science agent."""

    graph = StateGraph(AgentState)
    graph.add_node("load_data", load_dataset)
    graph.add_edge(START, "load_data")
    graph.add_node("analyse_data", analyse_dataset)
    graph.add_edge("load_data", "analyse_data")
    graph.add_edge("analyse_data", "analyse_metadata")
    graph.add_node("analyse_metadata", analyse_metadata)
    graph.add_node("LLM generate_summary", llm_generate_summary)
    graph.add_edge("analyse_metadata", "LLM generate_summary")
    graph.add_node("LLM generate_goals", llm_generate_goals)
    graph.add_edge("LLM generate_summary", "LLM generate_goals")
    graph.add_node("LLM generate_python_code", llm_generate_python_code)
    graph.add_node("LLM generate_r_code", llm_generate_r_code)
    graph.add_conditional_edges(
        "LLM generate_goals",
        decide_programming_language, {
            "python": "LLM generate_python_code",
            "r": "LLM generate_r_code"
        })
    graph.add_node("test_generated_code", test_generated_code)
    graph.add_edge("LLM generate_python_code", "test_generated_code")
    graph.add_edge("LLM generate_r_code", "statistics") # test_generated_code
    graph.add_conditional_edges(
        "test_generated_code",
        decide_regenerate_code,
        {
            "regenerate_code": "LLM regenerate_code",
            "judge": "LLM judge_plots",
            "end": "statistics"
        }
    )
    graph.add_node("LLM regenerate_code", llm_regenerate_code)
    graph.add_node("LLM judge_plots", llm_judge_code)
    graph.add_node("LLM refactor_plots", llm_refactor_plots)
    graph.add_edge("LLM judge_plots", "LLM refactor_plots")
    graph.add_edge("LLM refactor_plots", "test_generated_code")
    graph.add_edge("LLM regenerate_code", "test_generated_code")
    graph.add_node("statistics", __print_statistics)
    graph.add_edge("statistics", END)
    return graph.compile()
