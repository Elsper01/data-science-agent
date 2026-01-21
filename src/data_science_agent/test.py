import os
import shutil
import traceback
from time import time

from data_science_agent.graph import build_graph, AgentState
from data_science_agent.utils import print_color, enums

datasets = os.listdir("./src/resources/data/")

for i, dataset in enumerate(datasets[:1]):
    start = time()

    agent = build_graph()

    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("./src/resources/output/graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved graph.png successfully.")

    path = os.path.join("./src/resources/data/", dataset)

    os.makedirs(f"./src/resources/output/{i}", exist_ok=True)

    state: AgentState = {
        "dataset_path": os.path.join(path, "data.csv"),
        "metadata_path": os.path.join(path, "metadata.rdf"),
        "regeneration_attempts": 0,
        "programming_language": enums.ProgrammingLanguage.R,
        "is_refactoring": False,
        "output_path": f"./src/resources/output/{i}/",
        "durations": [],
        "llm_metadata": [],
        "statistics_path": "./src/resources/statistics/",
        "is_before_refactoring": True,
        "number_visualization_goals": 3,
    }

    try:
        result = agent.invoke(state)
    except Exception as e:
        print_color(f"Agent failed with error: {e}", enums.Color.WARNING)
        print(e)
        print(traceback.format_exc())
        continue

    print_color(f"Agent finished in {time() - start} seconds.", enums.Color.WARNING)

    # copy the summary to the output folder
    files = os.listdir("./src/resources/statistics/")
    latest_file = max(
        [os.path.join("./src/resources/statistics/", f) for f in files],
        key=os.path.getctime,
    )
    shutil.copy(latest_file, f"./src/resources/output/{i}/")
