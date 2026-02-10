import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from data_science_agent.dtos.base.responses.code_base import CodeBase
from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import get_llm_model, AGENT_LANGUAGE, print_color, MAX_REGENERATION_ATTEMPTS, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color
from data_science_agent.utils.pipeline import clear_output_dir, archive_images

prompt = Prompt(
    de={
        "system_prompt":
            """
                Du bist ein erfahrener Datenanalyst, der Experte darin ist existierenden Code zu verbessern, refactoren und Fehler zu beheben.
                Du bekommst als Input ein Visualisierungsziel, das mit dem Code umgesetzt werden soll, sowie den Code selbst und die Fehlermeldungen, die bei der Ausführung des Codes entstanden sind.
            """,
        "user_prompt":
            """
                Der vorherige Code hatte folgende Fehler:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'
                
                Bitte generiere den Code erneut und behebe die oben genannten Fehler.
                Das ist der vorherige Code:
                '{code}'
                Das ist das Visualisierungsziel, das mit dem Code umgesetzt werden soll:
                '{visualization_goal}'
            """,
    },
    en={
        "system_prompt":
            """
                You are an experienced data analyst, expert at improving, refactoring, and fixing bugs in existing code.
                You receive as input a visualization goal to be implemented with the code, the code itself, and the error messages that occurred during code execution.
            """,
        "user_prompt":
            """
                The previous code had the following errors:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'

                Please regenerate the code and fix the errors mentioned above.
                This is the previous code:
                '{code}'
                This is the visualization goal to be implemented with the code:
                '{visualization_goal}'
            """,
    }
)

Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_regenerate_code(state: AgentState) -> AgentState:
    """Regenerates code using an LLM based on previous test results for each visualization."""

    # clean output directory before regenerating plots
    archive_images(state["output_path"], state["regeneration_attempts"])
    clear_output_dir(state["output_path"])

    # increment global regeneration attempt counter
    state["regeneration_attempts"] += 1

    # we iterate over all visualizations and regenerate code for those that need it
    for i, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper
        # check the current number of regeneration attempts for this visualization, default is None so we set it to 0
        current_attempts = vis.code.regeneration_attempts
        if current_attempts is None:
            current_attempts = 0
        # if the code needs regeneration, and we haven't exceeded the max attempts, regenerate
        if vis.code.needs_regeneration[-1] and current_attempts < MAX_REGENERATION_ATTEMPTS:
            vis.code.regeneration_attempts = current_attempts + 1

            print_color(f"Regenerating code for vis#{i}, attempt {current_attempts}", Color.WARNING)

            messages = [
                SystemMessage(
                    content=prompt.get_prompt(
                        AGENT_LANGUAGE,
                        "system_prompt"
                    )
                ),
                HumanMessage(
                    content=prompt.get_prompt(
                        AGENT_LANGUAGE,
                        "user_prompt",
                        test_stdout=vis.code.std_out or "",
                        test_stderr=vis.code.std_err or "",
                        code=vis.code.code or "",
                        visualization_goal=vis.goal
                    )
                )
            ]

            temp_agent = create_agent(
                model=get_llm_model(LLMModel.GPT_5),
                response_format=Code
            )

            llm_response = temp_agent.invoke({"messages": messages})

            # only update code.code, other fields remain
            regenerated_code: Code = llm_response["structured_response"]

            vis.code.code = regenerated_code.code

            for message in reversed(llm_response["messages"]):
                if isinstance(message, AIMessage):
                    state["llm_metadata"].append(
                        LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
                    break

            vis.code.needs_regeneration.append(False)

    return state
