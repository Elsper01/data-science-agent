import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.dtos.base.responses.regeneration_base import RegenerationBase
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.utils import get_llm_model, AGENT_LANGUAGE, print_color, MAX_REGENERATION_ATTEMPTS, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

prompt = Prompt(
    de={
        "decide_regenerate_code_system_prompt": \
            """
                Du bist ein Experte darin Code Output zu interpretieren, der entscheidet, ob der gegebene Text Fehler enthält, die eine erneute Generierung des Codes erforderlich machen.
                Antworte mit einer bool Antwort, welche true ist, genau dann wenn der Code Fehler enthält, die eine erneute Erzeugung des Codes zwingend notwendig machen.
                Ansonsten antworte mit false.
                Wichtig, du bekommst den Text von stdout und stderr des Codes. Das heißt gegebenenfalls sind dort auch nur Infos oder Deprecated Warnings enthalten, diese musst du von wahren Fehlern bzw. Exceptions unterscheiden, welche unbedingt korrigiert werden müssen damit ein Diagramm erzeugt werden kann und den restlichen Ablauf des Skriptes nicht behindern.
            """,
        "decide_regenerate_code_user_prompt": \
            """
                Hier ist die Ausgabe (stdout) und die Fehlerausgabe (stderr) des Codes:
                stdout:
                '{test_stdout}'

                stderr:
                '{test_stderr}'

                Bitte entscheide, ob der Code unbedingt neu generiert werden muss.
            """,
    },
    en={
        "decide_regenerate_code_system_prompt": \
            """
                You are an expert in interpreting code output.
                Determine whether the given text contains errors that make regeneration of the code necessary.
                Answer with a boolean: true if the code contains critical errors that require regeneration,
                false otherwise.
                Note: You receive the content of stdout and stderr.
                Some lines may contain informational or deprecated warnings — these must be distinguished from real exceptions
                that prevent a plot from being created or block the rest of the script.
            """,
        "decide_regenerate_code_user_prompt": \
            """
                Here is the program output (stdout) and the error output (stderr):

                stdout:
                '{test_stdout}'

                stderr:
                '{test_stderr}'

                Please decide whether the code absolutely needs to be regenerated.
            """,
    }
)

Regeneration = import_language_dto(AGENT_LANGUAGE, RegenerationBase)
Visualization = import_language_dto(AGENT_LANGUAGE, VisualizationBase)

def decide_regenerate_code(state: AgentState) -> AgentState:
    """Decides whether the code should be regenerated based on test results for each visualization."""
    model = get_llm_model(LLMModel.GPT_4o)
    any_needs_regeneration = False

    for vis in state["visualizations"].visualizations:
        vis: Visualization

        test_stdout = vis.code.std_out or ""
        test_stderr = vis.code.std_err or ""

        if test_stdout or test_stderr:
            user_prompt = HumanMessage(
                content=prompt.get_prompt(
                    AGENT_LANGUAGE,
                    "decide_regenerate_code_user_prompt",
                    test_stdout=test_stdout,
                    test_stderr=test_stderr
                )
            )

            decide_agent = create_agent(
                model=model,
                response_format=Regeneration,
                system_prompt=prompt.get_prompt(AGENT_LANGUAGE, "decide_regenerate_code_system_prompt")
            )

            messages = [user_prompt]
            llm_response = decide_agent.invoke({"messages": messages})

            for message in reversed(llm_response["messages"]):
                if isinstance(message, AIMessage):
                    state["llm_metadata"].append(
                        LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
                    break

            regeneration_response: Regeneration = llm_response["structured_response"]

            vis.code.needs_regeneration = regeneration_response.should_be_regenerated
            print_color(f"Vis#{vis.goal.index} - Regeneration decision: {regeneration_response.should_be_regenerated}",
                        Color.OK_CYAN)

            if regeneration_response.should_be_regenerated:
                any_needs_regeneration = True
        else:
            vis.code.needs_regeneration = False

    if any_needs_regeneration and state["regeneration_attempts"] < MAX_REGENERATION_ATTEMPTS:
        print_color(f"Regenerating code, attempt {state['regeneration_attempts']}", Color.WARNING)
        return "regenerate_code"
    elif state["regeneration_attempts"] == MAX_REGENERATION_ATTEMPTS:
        print_color(f"Max attempts limit ({MAX_REGENERATION_ATTEMPTS}) reached.", Color.OK_GREEN)
        if not state["is_refactoring"]:
            return "evaluate"
        else:
            return "end"
    else:
        print_color(f"All visualizations are functionally working.", Color.OK_GREEN)
        if not state["is_refactoring"]:
            return "evaluate"
        else:
            return "end"
