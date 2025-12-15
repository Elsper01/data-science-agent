from langgraph.graph import StateGraph, START, END

from data_science_agent.graph import AgentState
from data_science_agent.pipeline import (
    load_dataset,
    analyse_dataset,
    load_metadata,
    llm_generate_summary,
    llm_generate_python_code,
    llm_generate_r_code,
    decide_programming_language
)


def build_graph():
    """Builds the state graph for the data science agent."""

    graph = StateGraph(AgentState)
    graph.add_node("load_data", load_dataset)
    graph.add_edge(START, "load_data")
    graph.add_node("analyse_data", analyse_dataset)
    graph.add_edge("load_data", "analyse_data")
    graph.add_node("load_metadata", load_metadata)
    graph.add_edge("analyse_data", "load_metadata")
    graph.add_node("LLM generate_summary", llm_generate_summary)
    graph.add_edge("load_metadata", "LLM generate_summary")
    graph.add_node("LLM generate_python_code", llm_generate_python_code)
    graph.add_node("LLM generate_r_code", llm_generate_r_code)
    graph.add_conditional_edges(
        "LLM generate_summary",
        decide_programming_language, {
            "python": "LLM generate_python_code",
            "r": "LLM generate_r_code"
        })
    # graph.add_node("test_generated_code", test_generated_code)
    graph.add_edge("LLM generate_python_code", END)
    graph.add_edge("LLM generate_r_code", END)
    # graph.add_edge("LLM generate_python_code", "test_generated_code")
    # graph.add_edge("LLM generate_r_code", "test_generated_code")
    # graph.add_conditional_edges(
    #     "test_generated_code",
    #     decide_regenerate_code,
    #     {
    #         "regenerate_code": "LLM regenerate_code",
    #         "judge": "LLM judge_plots",
    #         "end": END
    #     }
    # )
    # graph.add_node("LLM regenerate_code", llm_regenerate_code)
    # graph.add_node("LLM judge_plots", llm_judge_plots)
    # graph.add_node("LLM refactor_plots", llm_refactor_plots)
    # graph.add_edge("LLM judge_plots", "LLM refactor_plots")
    # graph.add_edge("LLM refactor_plots", "test_generated_code")
    # graph.add_edge("LLM regenerate_code", "test_generated_code")
    return graph.compile()
