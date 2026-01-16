import os
import shutil
from datetime import datetime
from time import time

from data_science_agent.graph import build_graph, AgentState
from data_science_agent.utils import print_color, enums

datasets = os.listdir("./src/resources/data/")

for i, dataset in enumerate(datasets):
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
        continue

    print_color(f"Agent finished in {time() - start} seconds.", enums.Color.WARNING)

    # we save all outputs to output folder that we archive

    # first we copy the summary to the output folder
    files = os.listdir("./src/resources/statistics/")
    latest_file = max(
        [os.path.join("./src/resources/statistics/", f) for f in files],
        key=os.path.getctime,
    )
    shutil.copy(latest_file, f"./src/resources/output/{i}/")

    # archive_path = r"C:\Users\jelsper\Desktop\archive"
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # archive_path = os.path.join(archive_path, f"run_{timestamp}")
    # # create a new archive dir for every run
    # os.makedirs(archive_path, exist_ok=False)
    #
    # output_dir_files = os.listdir("./src/resources/output/")
    # excluded_files = ["graph.png", ".gitkeep", ".gitignore"]
    # files_to_archive = [f for f in output_dir_files if f not in excluded_files]
    #
    # for file in files_to_archive:
    #     file_path = f"./src/resources/output/{file}"
    #     shutil.copy(file_path, archive_path)
    #     # delete file
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    #     else:
    #         shutil.rmtree(file_path)
