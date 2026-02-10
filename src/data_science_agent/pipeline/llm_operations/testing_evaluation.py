import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from data_science_agent.dtos.base.responses.regeneration_base import RegenerationBase
from data_science_agent.dtos.wrapper.VisualizationWrapper import VisualizationWrapper
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
                You are an expert at interpreting code output and deciding whether the given text contains errors that require code regeneration.
                Answer with a boolean value, which is true only if the code contains errors that make regeneration absolutely necessary.
                Otherwise, respond with false.
                Important: You receive the text from stdout and stderr of the code. This means there may only be informational messages or deprecated warnings included, which you must distinguish from true errors or exceptions that absolutely need to be corrected in order for a chart to be generated and not hinder the remaining execution of the script.
            """,
        "decide_regenerate_code_user_prompt": \
            """
                Here is the standard output (stdout) and error output (stderr) of the code:
                stdout:
                '{test_stdout}'

                stderr:
                '{test_stderr}'

                Please decide whether the code absolutely needs to be regenerated.
            """,
    },
)

Regeneration = import_language_dto(AGENT_LANGUAGE, RegenerationBase)

def decide_regenerate_code(state: AgentState) -> AgentState:
    """Decides whether the code should be regenerated based on test results for each visualization."""
    model = get_llm_model(LLMModel.GPT_4o)
    any_needs_regeneration = False

    for i, vis in enumerate(state["visualizations"]):
        vis: VisualizationWrapper

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

            if vis.code.needs_regeneration is None:
                vis.code.needs_regeneration = []

            vis.code.needs_regeneration.append(regeneration_response.should_be_regenerated)
            print_color(f"Vis#{i} - Regeneration decision: {regeneration_response.should_be_regenerated}",
                        Color.OK_CYAN)

            if regeneration_response.should_be_regenerated:
                any_needs_regeneration = True
        else:
            vis.code.needs_regeneration.append(False)

    if any_needs_regeneration and state["regeneration_attempts"] < MAX_REGENERATION_ATTEMPTS:
        print_color(f"Regenerating code, attempt {state['regeneration_attempts'] + 1}", Color.WARNING)
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
