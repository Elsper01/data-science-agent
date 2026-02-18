import os
import shutil
import traceback
from time import time

from data_science_agent.graph import build_graph, AgentState
from data_science_agent.utils import print_color, enums

datasets = os.listdir("./src/resources/data/study")

# random.seed(42)
selected_datasets = datasets  # random.sample(datasets, k=20)

counter_success = 0
counter_fail = 0
start = time()

from tqdm import tqdm

total_datasets = len(selected_datasets)
start_time_total = time()

running = True

for i, dataset in tqdm(enumerate(selected_datasets), total=total_datasets, desc="Processing datasets"):
    if not running:
        break
    iteration_start = time()
    print(f"\nDataset: {dataset}")

    agent = build_graph()

    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("./src/resources/output/graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved graph.png successfully.")

    try:
        path = os.path.join("./src/resources/data/study/", dataset)

        os.makedirs(f"./src/resources/output/{dataset}", exist_ok=True)

        state: AgentState = {
            "dataset_dir": path,
            "dataset_path": os.path.join(path, "dataset.csv"),
            "metadata_path": os.path.join(path, "metadata.rdf"),
            "regeneration_attempts": 0,
            "programming_language": enums.ProgrammingLanguage.R,
            "is_refactoring": False,
            "output_path": f"./src/resources/output/{dataset}/",
            "durations": [],
            "llm_metadata": [],
            "statistics_path": "./src/resources/statistics/",
            "is_before_refactoring": True,
            "number_visualization_goals": 3,
        }
        result = agent.invoke(state)
        counter_success += 1
        running = True#False
    except Exception as e:
        print_color(f"Agent failed with error: {e}", enums.Color.WARNING)
        counter_fail += 1
        continue

    elapsed_this = time() - iteration_start
    elapsed_total = time() - start_time_total

    # Estimate remaining time
    avg_per_dataset = elapsed_total / (i + 1)
    remaining = avg_per_dataset * (total_datasets - i - 1)

    print(f"Dataset {i}/{total_datasets} | Elapsed: {elapsed_total:.1f}s | Remaining: ~{remaining:.1f}s")

    # copy the summary to the output folder
    files = os.listdir("./src/resources/statistics/")
    latest_file = max(
        [os.path.join("./src/resources/statistics/", f) for f in files],
        key=os.path.getctime,
    )
    shutil.copy(latest_file, f"./src/resources/output/{dataset}/")

print_color(f"Generations failed: {counter_fail}", enums.Color.WARNING)
print_color(f"Generations {counter_success} successful", enums.Color.WARNING)
