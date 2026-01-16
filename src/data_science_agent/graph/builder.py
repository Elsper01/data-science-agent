from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
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
        f"Dataset Path: {state['dataset_path']}",
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
    state["stats_file_path"] = statistics_path

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

        # Evaluation Results
        f.write("\n")
        f.write("Visualization Results:\n")
        f.write("-" * 40 + "\n")

        visualizations: [VisualizationWrapper] = state.get("visualizations")
        if visualizations:
            for index, vis in enumerate(visualizations):
                # print all visualization information
                vis_index = index
                vis_desc = getattr(getattr(vis, "goal", None), "description", None) or str(getattr(vis, "goal", ""))

                f.write(f"Visualization #{vis_index if vis_index is not None else 'n/a'} - Goal: {vis_desc}\n")

                # helper to handle an evaluation object
                def _write_eval(prefix, eval_obj):
                    if not eval_obj:
                        return
                    try:
                        bugs = eval_obj.bugs
                        transformation = eval_obj.transformation
                        compliance = eval_obj.compliance
                        vtype = eval_obj.type
                        encoding = eval_obj.encoding
                        aesthetics = eval_obj.aesthetics
                    except Exception:
                        f.write(f"  {prefix}: {repr(eval_obj)}\n")
                        return

                    f.write(f"  {prefix}:\n")
                    f.write(f"    - bugs: {bugs.score} / 10\n")
                    f.write(f"      -> {bugs.rationale}\n")
                    f.write(f"    - transformation: {transformation.score} / 10\n")
                    f.write(f"      -> {transformation.rationale}\n")
                    f.write(f"    - compliance: {compliance.score} / 10\n")
                    f.write(f"      -> {compliance.rationale}\n")
                    f.write(f"    - type: {vtype.score} / 10\n")
                    f.write(f"      -> {vtype.rationale}\n")
                    f.write(f"    - encoding: {encoding.score} / 10\n")
                    f.write(f"      -> {encoding.rationale}\n")
                    f.write(f"    - aesthetics: {aesthetics.score} / 10\n")
                    f.write(f"      -> {aesthetics.rationale}\n")
                    total_score = bugs.score + transformation.score + compliance.score + vtype.score + encoding.score + aesthetics.score
                    avg = float(total_score) / 6
                    f.write(f"    Total Score: {total_score:.2f}/60\n")
                    f.write(f"    Average Score: {avg:.2f}/10\n")

                _write_eval("pre_refactoring_evaluation", getattr(vis, "pre_refactoring_evaluation", None))
                _write_eval("post_refactoring_evaluation", getattr(vis, "post_refactoring_evaluation", None))

                f.write("\n")

    visualizations = state.get("visualizations")
    # TODO: das kann direkt im oberen Loop mit ausgegeben werden
    if visualizations:
        for i, vis in enumerate(visualizations):
            vis_index = i
            print_color(f"LIDA Judge for vis#{vis_index if vis_index is not None else 'n/a'}", Color.WARNING)

            # prefer post if exists, else pre if exists; also print both if both present
            for label in ("pre_refactoring_evaluation", "post_refactoring_evaluation"):
                lida_result = getattr(vis, label, None)
                if not lida_result:
                    continue

                print_color(f"   - {label}:", Color.OK_BLUE)
                try:
                    print_color(f"      - bugs: {lida_result.bugs.score} / 10", Color.OK_BLUE)
                    print_color(f"      - transformation: {lida_result.transformation.score} / 10", Color.OK_BLUE)
                    print_color(f"      - compliance: {lida_result.compliance.score} / 10", Color.OK_BLUE)
                    print_color(f"      - type: {lida_result.type.score} / 10", Color.OK_BLUE)
                    print_color(f"      - encoding: {lida_result.encoding.score} / 10", Color.OK_BLUE)
                    print_color(f"      - aesthetics: {lida_result.aesthetics.score} / 10", Color.OK_BLUE)

                    total_score = (
                        lida_result.bugs.score
                        + lida_result.transformation.score
                        + lida_result.compliance.score
                        + lida_result.type.score
                        + lida_result.encoding.score
                        + lida_result.aesthetics.score
                    )
                    print_color(f"  Total Score: {total_score:.2f}/60", Color.OK_GREEN)
                    print_color(f"  Average Score: {round(float(total_score) / 6, 1)}/10", Color.OK_GREEN)
                except Exception:
                    print_color(f"  Could not read structured evaluation for {label}", Color.WARNING)

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
    graph.add_edge("LLM evaluate_visualizations #1", "statistics")
    graph.add_node("LLM regenerate_code", llm_regenerate_code)
    # graph.add_node("LLM judge_plots", llm_judge_code)
    # graph.add_node("LLM refactor_visualizations", llm_refactor_visualizations)
    # graph.add_edge("LLM judge_plots", "LLM refactor_visualizations")
    # graph.add_edge("LLM refactor_visualizations", "test_generated_code")
    graph.add_edge("LLM regenerate_code", "test_generated_code")
    graph.add_edge("LLM evaluate_visualizations #2", "statistics")
    graph.add_node("LLM evaluate_visualizations #2", llm_evaluate_visualizations)
    graph.add_node("statistics", __print_statistics)
    graph.add_edge("statistics", END)
    return graph.compile()
