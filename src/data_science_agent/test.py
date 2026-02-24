import os
import shutil
import traceback
from time import time

from data_science_agent.graph import build_graph, AgentState
from data_science_agent.utils import print_color, enums

datasets = os.listdir("./src/resources/data/study")

selected_datasets = ["13473"]#datasets  # random.sample(datasets, k=20)

counter_success = 0
counter_fail = 0
start = time()

from tqdm import tqdm

total_datasets = len(selected_datasets)
start_time_total = time()

for i, dataset in tqdm(enumerate(selected_datasets),
                       total=total_datasets,
                       desc="Processing datasets"):
    iteration_start = time()
    print(f"\nDataset: {dataset}")

    output_dir = f"./src/resources/output/{dataset}"

    # checks whether output dir exists and is empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"➡️  Output directory for '{dataset}' is not empty. Skipping...")
        continue

    # create dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    agent = build_graph()
    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("./src/resources/output/graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved graph.png successfully.")

    try:
        path = os.path.join("./src/resources/data/study/", dataset)

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

    except Exception as e:
        print_color(f"Agent failed with error: {e}", enums.Color.WARNING)
        traceback.print_exc()
        counter_fail += 1
        continue

    elapsed_this = time() - iteration_start
    elapsed_total = time() - start_time_total

    # Estimate remaining time
    avg_per_dataset = elapsed_total / (i + 1)
    remaining = avg_per_dataset * (total_datasets - i - 1)

    print(f"Dataset {i}/{total_datasets} | Elapsed: {elapsed_total:.1f}s | Remaining: ~{remaining:.1f}s")

    # Copy latest summary to output
    files = os.listdir("./src/resources/statistics/")
    latest_file = max(
        [os.path.join("./src/resources/statistics/", f) for f in files],
        key=os.path.getctime,
    )
    shutil.copy(latest_file, output_dir)

print_color(f"Generations failed: {counter_fail}", enums.Color.WARNING)
print_color(f"Generations {counter_success} successful", enums.Color.WARNING)
