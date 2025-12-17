import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from data_science_agent.dtos.base import RegenerationBase
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


def decide_regenerate_code(state: AgentState) -> AgentState:
    """Decides whether the code should be regenerated based on test results."""
    # LLM decides if the text in stdout and stderr are actual errors or just infos / deprecated warnings.
    if state["code_test_stdout"] or state["code_test_stderr"]:
        model = get_llm_model(LLMModel.GPT_4o)
        user_prompt = HumanMessage(
            content=prompt.get_prompt(
                AGENT_LANGUAGE,
                "decide_regenerate_code_user_prompt",
                test_stdout=state.get("code_test_stdout", ""),
                test_stderr=state.get("code_test_stderr", "")
            )
        )

        decide_agent = create_agent(
            model=model,
            response_format=Regeneration,
            system_prompt=prompt.get_prompt(AGENT_LANGUAGE, "decide_regenerate_code_system_prompt")
        )

        messages = [user_prompt]

        llm_response = decide_agent.invoke({"messages": messages})
        state["llm_metadata"].append(
            LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))
        regeneration_response: Regeneration = llm_response["structured_response"]
        print_color(f"Regeneration decision: {regeneration_response.should_be_regenerated}", Color.OK_CYAN)
        print(regeneration_response.should_be_regenerated)
        if regeneration_response.should_be_regenerated and state["regeneration_attempts"] < MAX_REGENERATION_ATTEMPTS:
            print_color(f"Regenerating code, attempt {state['regeneration_attempts']}", Color.WARNING)
            return "regenerate_code"
        elif state["regeneration_attempts"] == MAX_REGENERATION_ATTEMPTS:
            print_color(f"Max attempts limit ({MAX_REGENERATION_ATTEMPTS}) succeeded.", Color.OK_GREEN)
            if not state["is_refactoring"]:
                return "judge"
            else:
                return "end"
        else:
            print_color(f"Code is functionally working.", Color.OK_GREEN)
            if not state["is_refactoring"]:
                return "judge"
            else:
                return "end"
    else:
        if not state["is_refactoring"]:
            return "judge"
        else:
            return "end"
