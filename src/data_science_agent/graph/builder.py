from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from data_science_agent.graph import AgentState
from data_science_agent.pipeline import (
    load_dataset,
    analyse_dataset,
    analyse_metadata,
    llm_generate_summary,
    llm_generate_python_code,
    llm_generate_r_code,
    decide_programming_language,
    test_generated_code,
    decide_regenerate_code,
    llm_regenerate_code,
    llm_judge_code,
    llm_refactor_visualizations,
    llm_generate_goals,
    llm_evaluate_visualizations,
)
from data_science_agent.utils import print_color
from data_science_agent.utils.enums import Color


def __print_statistics(state: AgentState) -> AgentState:
    """Prints and saves the statistics of the graph to a file."""

    # Total values
    total_duration = sum(d.get_total_duration() or 0 for d in state["durations"])
    total_tokens = sum(m.token_usage.total_tokens or 0 for m in state["llm_metadata"])
    total_cost = sum(m.cost_details.total_cost or 0.0 for m in state["llm_metadata"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_lines = [
        "=" * 60,
        "Graph Statistics Report",
        "=" * 60,
        f"Timestamp: {timestamp}",
        f"Total Duration: {total_duration:.2f} seconds",
        f"Total Token Usage: {total_tokens} tokens",
        f"Total Costs: ${total_cost:.4f}",
        "=" * 60,
        "",
    ]

    stats_dir = Path(state["statistics_path"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    timestamp_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    statistics_path = stats_dir / f"statistics_{timestamp_filename}.txt"

    with open(statistics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))

        # Duration Statistics
        f.write("Duration Statistics:\n")
        f.write("-" * 40 + "\n")
        for duration in state["durations"]:
            f.write(f"Method: {duration.method_name}\n")
            f.write(f"  Duration: {duration.get_total_duration():.4f} seconds\n\n")

        f.write("\n")

        # LLM Cost & Token Statistics
        f.write("LLM Cost & Token Statistics:\n")
        f.write("-" * 40 + "\n")
        for metadata in state["llm_metadata"]:
            f.write(f"Method: {metadata.method_name}\n")
            f.write(f"  Model: {metadata.model_name}\n")
            f.write(f"  Token total: {metadata.token_usage.total_tokens} tokens\n")
            f.write(f"  Costs total: ${metadata.cost_details.total_cost:.4f}\n\n")

    print_color(f"Statistics saved to: {statistics_path}", Color.OK_GREEN)

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
    graph.add_edge("LLM generate_r_code", "test_generated_code")
    graph.add_conditional_edges(
        "test_generated_code",
        decide_regenerate_code,
        {
            "regenerate_code": "LLM regenerate_code",
            "evaluate": "LLM evaluate_visualizations #1",
            "end": "LLM evaluate_visualizations #2"
        }
    )
    graph.add_node("LLM evaluate_visualizations #1", llm_evaluate_visualizations)
    graph.add_edge("LLM evaluate_visualizations #1", "LLM judge_plots")
    graph.add_node("LLM regenerate_code", llm_regenerate_code)
    graph.add_node("LLM judge_plots", llm_judge_code)
    graph.add_node("LLM refactor_visualizations", llm_refactor_visualizations)
    graph.add_edge("LLM judge_plots", "LLM refactor_visualizations")
    graph.add_edge("LLM refactor_visualizations", "test_generated_code")
    graph.add_edge("LLM regenerate_code", "test_generated_code")
    graph.add_edge("LLM evaluate_visualizations #2", "statistics")
    graph.add_node("LLM evaluate_visualizations #2", llm_evaluate_visualizations)
    graph.add_node("statistics", __print_statistics)
    graph.add_edge("statistics", END)
    return graph.compile()
