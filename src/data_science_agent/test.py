from time import time

from data_science_agent.graph import build_graph
from data_science_agent.utils import print_color, enums

start = time()

agent = build_graph()

png_bytes = agent.get_graph().draw_mermaid_png()

with open("./src/resources/output/graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved graph.png successfully.")

result = agent.invoke(
    {
        "dataset_path": "./src/resources/data/pegel.csv",
        "metadata_path": "./src/resources/data/pegel.rdf",
        "regeneration_attempts": 0,
        "programming_language": enums.ProgrammingLanguage.R,
        "is_refactoring": False,
        "output_path": "./src/resources/output/"
    }
)

print_color(f"Agent finished in {time() - start} seconds.", enums.Color.WARNING)
