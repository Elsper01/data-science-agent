import subprocess
import sys
import os
import tempfile
import os

from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import print_color
from data_science_agent.utils.enums import ProgrammingLanguage, Color


def _count_generated_imgs(state: AgentState, vis_index: int) -> int:
    all_output_files = os.listdir(state["output_path"])
    img_files = [f for f in all_output_files if f.lower().endswith(".png")]
    img_count = 0
    for img in img_files:
        if img.startswith(f"{vis_index}_"):
            img_count += 1
    return img_count

@track_duration
def test_generated_code(state: AgentState) -> AgentState:
    """Test the generated code by executing it and capturing its output and errors."""
    language: ProgrammingLanguage = state["programming_language"]

    project_root = state.get("project_root", os.getcwd())
    working_dir = project_root

    for i, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        code = vis.code.code

        if language is ProgrammingLanguage.R:
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.R',
                    delete=False,
                    encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                cmd = ["Rscript", temp_file]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=working_dir
                )
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        else:
            cmd = [sys.executable, "-c", code]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=working_dir
            )

        img_count = _count_generated_imgs(state, i)
        key = f"{state["regeneration_attempts"]}"
        vis.VER_values[key] = img_count

        if result.stdout:
            vis.code.std_out = result.stdout
        else:
            vis.code.std_out = "No output from generated code."
        if result.stderr:
            vis.code.std_err = result.stderr
        else:
            vis.code.std_err = "No errors from generated code."
        print_color(f"Testing generated code ({language.value}) for vis#{i}: ", Color.HEADER)

    return state