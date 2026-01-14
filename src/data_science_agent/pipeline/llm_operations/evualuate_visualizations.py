import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from data_science_agent.dtos.base import LidaEvaluationBase
from data_science_agent.dtos.base.responses.visualization_base import VisualizationBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt, import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color, LLMMetadata
from data_science_agent.utils.enums import LLMModel, Color

prompt: Prompt = Prompt(
    de={
        "system_prompt":
            """
            Du bist ein erfahrener Assistent für Datenvisualisierung und bewertest Code, der Visualisierungen erstellt. 
            Bewerte den gegebenen Code ausschließlich danach, wie gut er das angegeben Ziel der Visualisierung erfüllt. 
            Analysiere den Code sachlich, beachte Visualisierungs-Best-Practices und begründe jede Bewertung klar.
            
            Verwende die folgenden sechs LIDA-Bewertungsdimensionen:
            
            1. **bugs** – Enthält der Code syntaktische oder logische Fehler, die die Ausführung verhindern oder das Ergebnis verfälschen?
               Wenn ein Fehler den Code unbrauchbar macht, muss der Wert unter 5 liegen.
            
            2. **transformation** – Ist die Datenvorbereitung (Filtern, Aggregieren, Gruppieren etc.) korrekt und sinnvoll für das Ziel?
            
            3. **compliance** – Wie gut erfüllt der Code das formulierte Visualisierungsziel und zeigt die gewünschten Zusammenhänge?
            
            4. **type** – Ist der gewählte Visualisierungstyp (z.B. Balken-, Linien-, Streudiagramm) für Ziel und Daten geeignet?
            
            5. **encoding** – Werden Daten und Variablen korrekt und effektiv über visuelle Kanäle (Achsen, Farben, Formen, Größen) dargestellt?
            
            6. **aesthetics** – Ist das visuelle Erscheinungsbild (Layout, Farben, Lesbarkeit) ansprechend und verständlich?
            
            Bewerte jede Dimension auf einer Skala von **1 (schlecht)** bis **10 (ausgezeichnet)**.  
            Liefere für jede Bewertung eine **präzise Begründung**, warum du diesen Wert gewählt hast.
            """,
        "user_prompt":
            """
            Bewerte den folgenden Python-Visualisierungscode in Hinblick auf das angegebene Ziel. 
            Beziehe dich bei deiner Evaluation auf die sechs LIDA-Dimensionen (bugs, transformation, compliance, type, encoding, aesthetics).


            VISUALISIERUNGSZIEL:
            {goal_description}

            Generierter Code:
            ```{programming_language}
            {code}
            ```

            Evaluiere den Code basierend auf den 6 Dimensionen.
            """
    },
    en={
        "system_prompt":
            """
            You are an experienced data visualization assistant responsible for evaluating code that creates visualizations.
            Assess the given code solely on how well it fulfills the stated visualization goal. 
            Analyze the code objectively, follow data visualization best practices, and clearly justify every rating you assign.

            Use the following six LIDA evaluation dimensions:

            1. **bugs** – Does the code contain syntax or logic errors that would prevent execution or distort the result?  
               If such an error makes the code unusable, the score must be below 5.

            2. **transformation** – Is the data preparation (filtering, aggregation, grouping, etc.) correct and meaningful for the given goal?

            3. **compliance** – How well does the code fulfill the defined visualization goal and reveal the intended insights?

            4. **type** – Is the chosen visualization type (e.g., bar, line, scatter plot) appropriate for the objective and data?

            5. **encoding** – Are data and variables properly and effectively represented through visual channels (axes, color, form, size, etc.)?

            6. **aesthetics** – Is the visual appearance (layout, colors, readability) clear, appealing, and easy to interpret?

            Rate each dimension on a scale from **1 (poor)** to **10 (excellent)**.  
            Provide a **precise explanation** for each score, describing why you chose that rating.
            """,
        "user_prompt":
            """
            Evaluate the following Python visualization code in relation to the stated goal.  
            Base your evaluation on the six LIDA dimensions (bugs, transformation, compliance, type, encoding, aesthetics).

            VISUALIZATION GOAL:
            {goal_description}

            Generated Code:
            ```{programming_language}
            {code}
            ```

            Evaluate the code thoroughly along all six dimensions.
            """
    }
)

LidaEvaluation = import_language_dto(AGENT_LANGUAGE, LidaEvaluationBase)
Visualization = import_language_dto(AGENT_LANGUAGE, VisualizationBase)


@track_duration
def llm_evaluate_visualizations(state: AgentState) -> AgentState:
    """Evaluate the generated code for each visualization using LIDA evaluation metric."""
    system_prompt = prompt.get_prompt(AGENT_LANGUAGE, "system_prompt")
    programming_language = state["programming_language"].value

    for vis in state["visualizations"].visualizations:
        vis: Visualization

        temp_agent = create_agent(
            model=get_llm_model(LLMModel.MISTRAL),
            system_prompt=SystemMessage(content=system_prompt),
            response_format=LidaEvaluation
        )

        user_prompt = prompt.get_prompt(
            AGENT_LANGUAGE,
            "user_prompt",
            goal_description=vis.goal,
            code=vis.code.code,
            programming_language=programming_language
        )

        llm_response = temp_agent.invoke({"messages": [HumanMessage(content=user_prompt)]})
        lida_result: LidaEvaluation = llm_response["structured_response"]

        print_color(f"LIDA Judge for vis#{vis.goal.index}", Color.WARNING)

        print_color(f"   - bugs: {lida_result.bugs.score} / 10", Color.OK_BLUE)
        print_color(f"   - transformation: {lida_result.transformation.score} / 10", Color.OK_BLUE)
        print_color(f"   - compliance: {lida_result.compliance.score} / 10", Color.OK_BLUE)
        print_color(f"   - type: {lida_result.type.score} / 10", Color.OK_BLUE)
        print_color(f"   - encoding: {lida_result.encoding.score} / 10", Color.OK_BLUE)
        print_color(f"   - aesthetics: {lida_result.aesthetics.score} / 10", Color.OK_BLUE)

        total_score = lida_result.bugs.score + lida_result.transformation.score + \
                      lida_result.compliance.score + lida_result.type.score + \
                      lida_result.encoding.score + lida_result.aesthetics.score
        print_color(f"\n  Total Score: {total_score:.2f}/60", Color.OK_GREEN)
        print_color(f"\n  Average Score: {round(float(total_score) / 6, 1)}/10", Color.OK_GREEN)

        # TODO: vielleicht gehen wir nur in refactoring für eine Visualisierung wenn der Wert unter 5 ist?
        if state["is_before_refactoring"]:
            vis.pre_refactoring_evaluation = lida_result
        else:
            vis.post_refactoring_evaluation = lida_result

        for message in reversed(llm_response["messages"]):
            if isinstance(message, AIMessage):
                state["llm_metadata"].append(
                    LLMMetadata.from_ai_message(message, inspect.currentframe().f_code.co_name))
                break

    state["is_before_refactoring"] = False

    return state
