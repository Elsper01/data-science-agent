from datetime import datetime
from pathlib import Path

from data_science_agent.dtos.wrapper.visualization import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.utils import print_color, MAX_REGENERATION_ATTEMPTS, AGENT_LANGUAGE
from data_science_agent.utils.enums import Color


def print_statistics(state: AgentState) -> AgentState:
    """Prints and saves the statistics of the graph to a file."""
    # VER lines

    VER_lines = [
        "VER Statistics",
        "=" * 60,
        "",
    ]
    # Calculate VER per generation / regeneration attempt
    for generation_attempt in range(0, MAX_REGENERATION_ATTEMPTS):
        key = str(generation_attempt)
        no_vis = 0
        one_vis = 0
        multiple_vis = 0
        for i, vis in enumerate(state["visualizations"]):
            vis: VisualizationWrapper
            # check if genration is already done
            ver = vis.VER_values.get(key)
            if ver is None:
                VER_lines.append(f"VER for generation attempt #{generation_attempt}: already done.")
                break
            # if the value exists, update the counter to calculate VER
            else:
                if ver == 0:
                    no_vis += 1
                elif ver == 1:
                    one_vis += 1
                elif ver >= 2:
                    multiple_vis += 1
                else:
                    raise ValueError("Invalid VER result value. Values must be positive integers.")
        ver_exactly_one = one_vis / max(1, (no_vis + one_vis + multiple_vis))
        ver_one_or_more = (one_vis + multiple_vis) / max(1, (no_vis + one_vis + multiple_vis))
        VER_lines.append(f"VER for generation attempt #{generation_attempt}:")
        VER_lines.append(f"  - No Visualization: {no_vis}")
        VER_lines.append(f"  - Exactly One Visualization: {one_vis} / {max(1, (no_vis + one_vis + multiple_vis))} = ({ver_exactly_one:.2%})")
        VER_lines.append(f"  - One or More Visualization: {one_vis + multiple_vis} / {max(1, (no_vis + one_vis + multiple_vis))} =  ({ver_one_or_more:.2%})")

    VER_lines.append("=" * 60)

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
        f"Programming Language: {state['programming_language'].value}",
        f"Language: {AGENT_LANGUAGE}",
        f"Numer of maximum Regeneration Attempts: {MAX_REGENERATION_ATTEMPTS}",
        f"Timestamp: {timestamp}",
        f"Total Duration: {total_duration:.2f} seconds",
        f"Total Token Usage: {total_tokens} tokens",
        f"Total Costs: ${total_cost:.4f}",
        "=" * 60,
    ]

    stats_dir = Path(state["statistics_path"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    timestamp_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    statistics_path = stats_dir / f"statistics_{timestamp_filename}.txt"
    state["stats_file_path"] = statistics_path

    with open(statistics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write("\n".join(VER_lines))

        # Duration Statistics
        f.write("\n"+"=" * 60+"\n")
        f.write("Duration Statistics:\n")
        f.write("=" * 60 + "\n")
        for duration in state["durations"]:
            f.write(f"Method: {duration.method_name}\n")
            f.write(f"  Duration: {duration.get_total_duration():.4f} seconds\n\n")

        f.write("\n")

        # LLM Cost & Token Statistics
        f.write("=" * 60 + "\n")
        f.write("LLM Cost & Token Statistics:\n")
        f.write("=" * 60 + "\n")
        for metadata in state["llm_metadata"]:
            f.write(f"Method: {metadata.method_name}\n")
            f.write(f"  Model: {metadata.model_name}\n")
            f.write(f"  Token total: {metadata.token_usage.total_tokens} tokens\n")
            f.write(f"  Costs total: ${metadata.cost_details.total_cost:.4f}\n\n")

        # Evaluation Results
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("Visualization Results:\n")
        f.write("=" * 60 + "\n")

        visualizations: [VisualizationWrapper] = state.get("visualizations")
        if visualizations:
            for index, vis in enumerate(visualizations):
                # print all visualization information
                vis_index = index
                vis_desc = getattr(getattr(vis, "goal", None), "description", None) or str(getattr(vis, "goal", ""))

                f.write(f"Visualization #{vis_index if vis_index is not None else 'n/a'}\n")
                f.write(f"-> Goal:\n")
                f.write(f"{vis.goal}")
                f.write(f"\n\n")

                f.write(f"-> Code:\n")
                f.write(f"{vis.code.code}")
                f.write(f"\n\n")

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

                f.write(f"-> SEVQ:\n")
                _write_eval("pre_refactoring_evaluation", getattr(vis, "pre_refactoring_evaluation", None))
                f.write(f"\n\n")
                # _write_eval("post_refactoring_evaluation", getattr(vis, "post_refactoring_evaluation", None))


        print_color(f"Statistics saved to: {statistics_path}", Color.OK_GREEN)

        # TODO: we have to temporariy

        return state
