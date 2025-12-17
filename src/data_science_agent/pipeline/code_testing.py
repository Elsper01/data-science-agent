import subprocess
import sys

from data_science_agent.graph import AgentState
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import print_color
from data_science_agent.utils.enums import ProgrammingLanguage, Color


@track_duration
def test_generated_code(state: AgentState) -> AgentState:
    """Test the generated code by executing it and capturing its output and errors."""
    language: ProgrammingLanguage = state["programming_language"]
    script_path = state["script_path"]
    if language is ProgrammingLanguage.R:
        cmd = ["Rscript", script_path]
    else:  # default to python
        cmd = [sys.executable, script_path]

    generated_code_test_result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    # we save both stdout and stderr to see what the LLM produced and to determine if code must be regenerated
    if generated_code_test_result.stdout:
        state["code_test_stdout"] = generated_code_test_result.stdout
    else:
        state["code_test_stdout"] = "No output from generated code."
    if generated_code_test_result.stderr:
        state["code_test_stderr"] = generated_code_test_result.stderr
    else:
        state["code_test_stderr"] = "No errors from generated code."
    print_color(f"Testing generated code ({language.value}): ", Color.HEADER)
    print("output: ", generated_code_test_result.stdout)
    print("error: ", generated_code_test_result.stderr)
    return state
