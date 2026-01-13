import inspect
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from data_science_agent.dtos.base.responses.judge_base import JudgeBase
from data_science_agent.dtos.base.responses.judge_verdict_base import JudgeVerdictBase
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

from data_science_agent.dto.lida_evaluation import LidaeEvaluation

prompt: Prompt = Prompt(
    de={
        "judge_lida_system_prompt": "...",  # Inhalt aus YAML oben
        "judge_lida_user_prompt":
            """
            Bewerte den folgenden Visualisierungscode für das angegebene Ziel.

            VISUALISIERUNGSZIEL:
            {goal_description}

            Generierter Code:
            ```{language}
            {code}
            ```

            Evaluiere den Code basierend auf den 6 Dimensionen.
            """
    },
    en={
        "judge_lida_system_prompt": "...",  # Inhalt aus YAML oben
        "judge_lida_user_prompt":
            """
            Evaluate the following visualization code for the specified goal.

            VISUALIZATION GOAL:
            {goal_description}

            Generated code:
            ```{language}
            {code}
            ```

            Evaluate the code based on the 6 dimensions.
            """
    }
)

JudgeVerdict = import_language_dto(AGENT_LANGUAGE, JudgeVerdictBase)
Visualization = import_language_dto(AGENT_LANGUAGE, VisualizationBase)
Judge = import_language_dto(AGENT_LANGUAGE, JudgeBase)


@track_duration
def llm_judge_code_lida(state: AgentState) -> AgentState:
    """Ask the LLM to judge the generated code using LIDA criteria."""
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "judge_lida_system_prompt")
    language = "python"  # Oder dynamisch basierend auf state

    for vis in state["visualizations"].visualizations:
        vis: Visualization

        temp_agent = create_agent(
            model=get_llm_model(LLMModel.GPT_5),
            system_prompt=SystemMessage(content=system_prompt),
            response_format=LidaeEvaluation
        )

        code_content = prompt.get_prompt(
            AGENT_LANGUAGE,
            "judge_lida_user_prompt",
            goal_description=vis.goal,
            code=vis.code.code,
            language=language
        )

        llm_response = temp_agent.invoke({"messages": [HumanMessage(content=code_content)]})
        lida_result: LidaeEvaluation = llm_response["structured_response"]

        print_color(f"LIDA Judge for vis#{vis.goal.index}", Color.WARNING)

        # Ergebnisse ausgeben
        for eval_item in lida_result.evaluations:
            print(f"  {eval_item.dimension}: {eval_item.score}/10")
            print(f"    Rationale: {eval_item.rationale}")

        # Gesamtscore berechnen
        avg_score = sum(e.score for e in lida_result.evaluations) / len(lida_result.evaluations)
        print(f"\n  Average Score: {avg_score:.2f}/10")
        print("-----")

        # Optional: Als JudgeVerdict speichern (konvertieren)
        needs_regeneration = any(e.score < 5 for e in lida_result.evaluations)

        critic_notes = "\n".join(
            f"{e.dimension.upper()} ({e.score}/10): {e.rationale}"
            for e in lida_result.evaluations
        )

        vis.code.judge_result = JudgeVerdict(
            figure_name=f"figure_{vis.goal.index}",
            file_name=vis.code.file_name,
            critic_notes=critic_notes,
            suggestion_code=None,
            needs_regeneration=needs_regeneration,
            can_be_deleted=False
        )

        state["llm_metadata"].append(
            LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))

    return state