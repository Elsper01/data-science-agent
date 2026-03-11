import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from tqdm import tqdm

from data_science_agent.graph import build_graph, AgentState
from data_science_agent.utils import print_color, enums

STUDY_DIR = "./src/resources/data/study"
OUTPUT_BASE = "./src/resources/output"
STATISTICS_DIR = "./src/resources/statistics"

random.seed(42)


def process_dataset(dataset: str) -> tuple[str, bool, str]:
    """Worker-Funktion, die einen einzelnen Datensatz verarbeitet."""
    iteration_start = time()
    dataset_path = os.path.join(STUDY_DIR, dataset)
    output_dir = os.path.join(OUTPUT_BASE, dataset)

    try:
        if os.path.exists(output_dir) and os.listdir(output_dir):
            return dataset, True, f"➡️  Output directory for '{dataset}' is not empty. Skipping..."

        os.makedirs(output_dir, exist_ok=True)

        agent = build_graph()

        # einmal pro Prozess mermaid Graph rendern (optional)
        png_bytes = agent.get_graph().draw_mermaid_png()
        with open(os.path.join(output_dir, "graph.png"), "wb") as f:
            f.write(png_bytes)

        state: AgentState = {
            "dataset_dir": dataset_path,
            "dataset_path": os.path.join(dataset_path, "dataset.csv"),
            "metadata_path": os.path.join(dataset_path, "metadata.rdf"),
            "regeneration_attempts": 0,
            "programming_language": enums.ProgrammingLanguage.R,
            "is_refactoring": False,
            "output_path": output_dir,
            "durations": [],
            "llm_metadata": [],
            "statistics_path": output_dir,
            "is_before_refactoring": True,
            "number_visualization_goals": 3,
        }

        agent.invoke(state)

        elapsed = time() - iteration_start
        return dataset, True, f"✅ Finished in {elapsed:.1f}s"

    except Exception as e:
        tb = traceback.format_exc()
        return dataset, False, f"❌ Failed: {e}\n{tb}"


def main():
    datasets = os.listdir(STUDY_DIR)
    selected_datasets = datasets#random.sample(datasets, k=10)

    start = time()
    counter_success = 0
    counter_fail = 0

    MAX_WORKERS = min(8, os.cpu_count())

    print_color(f"Starting parallel processing with {MAX_WORKERS} workers …", enums.Color.OK_BLUE)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_dataset, ds): ds for ds in selected_datasets}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            dataset = futures[future]
            ok = False
            msg = ""
            try:
                dataset, ok, msg = future.result()
            except Exception as e:
                msg = f"❌ Future raised exception: {e}"
                ok = False
            print(msg)

            if ok:
                counter_success += 1
            else:
                counter_fail += 1

    elapsed_total = time() - start
    print_color(f"\nTotal time: {elapsed_total/60:.1f}min", enums.Color.OK_BLUE)
    print_color(f"Generations successful: {counter_success}", enums.Color.OK_GREEN)
    print_color(f"Generations failed: {counter_fail}", enums.Color.WARNING)


if __name__ == "__main__":
    main()