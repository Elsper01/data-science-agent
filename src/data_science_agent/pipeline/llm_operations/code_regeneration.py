import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from data_science_agent.dtos.base import CodeBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.pipeline.utils import generate_and_write_code
from data_science_agent.utils import get_llm_model, AGENT_LANGUAGE
from data_science_agent.utils.enums import LLMModel
from data_science_agent.utils.pipeline import clear_output_dir

# TODO: we dont we have any system prompt here?
prompt = Prompt(
    de={
        "regenerate_code_user_prompt":
            """
                Der vorherige Code hatte folgende Fehler:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'
                Bitte generiere den Code erneut und behebe die oben genannten Fehler.
                Das ist die Beschreibung des Codes:
                '{code_explanation}'
                Das ist der vorherige Code:
                '{code}'
            """,
    },
    en={
        "regenerate_code_user_prompt":
            """
                The previous code produced the following errors:
                stdout:
                '{test_stdout}'
                stderr:
                '{test_stderr}'

                Please regenerate the code and fix the errors above.
                This is the description of the code:
                '{code_explanation}'
                This is the previous code:
                '{code}'
            """,
    }
)

Code = import_language_dto(AGENT_LANGUAGE, CodeBase)


@track_duration
def llm_regenerate_code(state: AgentState) -> AgentState:
    """Regenerates code using an LLM based on previous test results."""
    state["messages"].append(
        HumanMessage(
            content=prompt.get_prompt(
                AGENT_LANGUAGE,
                "regenerate_code_user_prompt",
                test_stdout=state.get("code_test_stdout", ""),
                test_stderr=state.get("code_test_stderr", ""),
                code_explanation=getattr(state.get("code"), "explanation", ""),
                code=getattr(state.get("code"), "code", "")
            )
        )
    )

    # clean output directory before regenerating plots
    clear_output_dir(state["output_path"])

    state["regeneration_attempts"] += 1
    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=Code
    )
    return generate_and_write_code(state, temp_agent, state["messages"], inspect.currentframe().f_code.co_name)
