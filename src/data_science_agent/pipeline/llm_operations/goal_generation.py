import inspect

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from data_science_agent.dtos.base.responses import GoalContainerBase
from data_science_agent.graph import AgentState
from data_science_agent.language import Prompt
from data_science_agent.language import import_language_dto
from data_science_agent.pipeline.decorator.duration_tracking import track_duration
from data_science_agent.utils import AGENT_LANGUAGE, get_llm_model, print_color
from data_science_agent.utils.enums import LLMModel, Color
from data_science_agent.utils.llm_metadata import LLMMetadata

"""
Parts of this code are adopted from the Microsoft LIDA project:
    https://github.com/microsoft/lida

You will find in every method / function docstring a note when it was copied / adopted from LIDA.

Citation:
    cff-version: 1.2.0
    message: "If you use this software, please cite it as below."
    authors:
      - family-names: Dibia
        given-names: Victor
    title: "LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models"
    version: 1.0.0
    date-released: 2023-07-01
    url: "https://aclanthology.org/2023.acl-demo.11"
    doi: "10.18653/v1/2023.acl-demo.11"
    conference:
      name: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)"
      month: jul
      year: 2023
      address: "Toronto, Canada"
      publisher: "Association for Computational Linguistics"
"""

prompt: Prompt = Prompt(
    de={
        "system_prompt": \
            """
                Du bist ein erfahrener Datenanalyst, der anhand einer Datenzusammenfassung und einer festgelegten Persona eine vorgegebene Anzahl an aussagekräftigen ZIELEN generieren kann. DIE EMPFOHLENEN VISUALISIERUNGEN MÜSSEN DEN BESTEN PRAKTIKEN FÜR VISUALISIERUNGEN FOLGEN (z.B. Balkendiagramme statt Kreisdiagramme zur Vergleich von Mengen) UND SINNVOLL SEIN (z.B. Längen- und Breitengrade nur auf Karten darstellen, wo es angemessen ist). Sie müssen außerdem auf die angegebene Persona zugeschnitten sein. Jedes Ziel muss eine Frage, eine Visualisierung (DIE VISUALISIERUNG MUSS DIE EXAKTEN SPALTENFELDER AUS DER ZUSAMMENFASSUNG NENNEN) und eine Begründung enthalten (RECHTFERTIGUNG, WARUM GENAU DIESE DATENFELDER VERWENDET WERDEN UND WAS DURCH DIE VISUALISIERUNG GELERNT WIRD). Jedes Ziel MUSST die exakten Felder aus der obigen Datensummary erwähnen.
            """,
        "user_prompt": \
            """
                Die Anzahl der zu generierenden ZIELE beträgt {n}. Die Ziele sollen auf der folgenden Datensummary basieren:
                {summary}
            """
    },
    en={
        "system_prompt": \
            """
                You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
            """,
        "user_prompt": \
            """
                The number of GOALS to generate is {n}. The goals should be based on the data summary below:
                {summary}
            """,
    }
)

GoalContainerBase = import_language_dto(AGENT_LANGUAGE, GoalContainerBase)


@track_duration
def llm_generate_goals(state: AgentState) -> AgentState:
    """
        This node generates the visualization goals by using a LLM. It uses the before generated summary of the dataset.

        ** Modified copy from LIDA project **
    """

    system_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "system_prompt",
    )

    user_prompt = prompt.get_prompt(
        AGENT_LANGUAGE,
        "user_prompt",
        n=10,
        summary=state["summary"].summary
    )
    user_msg = HumanMessage(content=user_prompt)

    temp_agent = create_agent(
        model=get_llm_model(LLMModel.GPT_5),
        response_format=GoalContainerBase,
        system_prompt=system_prompt
    )

    llm_response = temp_agent.invoke({"messages": [user_msg]})

    state["llm_metadata"].append(
        LLMMetadata.from_ai_message(llm_response["messages"][-1], inspect.currentframe().f_code.co_name))

    print_color(f"LLM result: ", Color.HEADER)
    goals: GoalContainerBase = llm_response["structured_response"]
    print_color(f"Dataset Summary: ", Color.OK_GREEN)
    for goal in goals.goals:
        print_color(f"Goal Nr. {goal.index}", Color.OK_GREEN)
        print_color(f"- Goal Question: {goal.question}", Color.OK_BLUE)
        print_color(f"- Visualization: {goal.visualization}", Color.OK_BLUE)
        print_color(f"- Rationale: {goal.rationale}", Color.OK_BLUE)
    state["goals"] = goals

    return state
