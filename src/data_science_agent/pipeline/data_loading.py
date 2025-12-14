from data_science_agent.graph import AgentState

import pandas as pd

# TODO: hier muss noch auf dynamisches Laden beliebiger tabellarischer Datensätze umgebaut werden

def load_dataset(state: AgentState) -> AgentState:
    """Lädt den Datensatz und speichert ihn in einem pandas DataFrame."""
    state["dataset_df"] = pd.read_csv(
        state["dataset_path"],
        sep=";"
    )
    return state