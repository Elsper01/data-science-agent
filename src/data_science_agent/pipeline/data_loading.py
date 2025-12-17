from data_science_agent.graph import AgentState

import pandas as pd

from data_science_agent.pipeline.decorator.duration_tracking import track_duration


# TODO: hier muss noch auf dynamisches Laden beliebiger tabellarischer Datensätze umgebaut werden

@track_duration
def load_dataset(state: AgentState) -> AgentState:
    """Lädt den Datensatz und speichert ihn in einem pandas DataFrame."""
    state["dataset_df"] = pd.read_csv(
        state["dataset_path"],
        sep=";"
    )
    return state