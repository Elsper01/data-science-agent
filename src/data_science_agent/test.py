import os
from time import time

from data_science_agent.graph import build_graph
from data_science_agent.utils import print_color, enums

datasets = os.listdir("./src/resources/data/")

for dataset in datasets[:1]:
    start = time()

    agent = build_graph()

    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("./src/resources/output/graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved graph.png successfully.")

    path = os.path.join("./src/resources/data/", "pegelstände")

    result = agent.invoke(
        {
            "dataset_path": os.path.join(path, "data.csv"),
            "metadata_path": os.path.join(path, "metadata.rdf"),
            "regeneration_attempts": 2, # TODO: später wieder auf 0 setzen
            "programming_language": enums.ProgrammingLanguage.R,
            "is_refactoring": False,
            "output_path": "./src/resources/output/",
            "durations": [],
            "llm_metadata": [],
            "statistics_path": "./src/resources/statistics/",
            "is_before_refactoring": True,
            "number_visualization_goals": 3,
        }
    )

    print_color(f"Agent finished in {time() - start} seconds.", enums.Color.WARNING)

# from data_science_agent.graph.agent_state import AgentState
#
# agent = AgentState({
#     "dataset_path": os.path.join("spass123", "data.csv"),
#     "metadata_path": os.path.join("spass123", "metadata.rdf"),
#     "regeneration_attempts": 0,
#     "programming_language": enums.ProgrammingLanguage.R,
#     "is_refactoring": False,
#     "output_path": "./src/resources/output/",
#     "durations": [],
#     "llm_metadata": [],
# })
# print(agent)
