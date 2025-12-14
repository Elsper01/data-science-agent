import pandas as pd
from data_science_agent.graph import AgentState
from data_science_agent.language import import_language_dto
from data_science_agent.dtos.base import DescriptionBase
from data_science_agent.utils import AGENT_LANGUAGE

Description = import_language_dto(AGENT_LANGUAGE, DescriptionBase)

def __convert_nan_to_none(record: dict) -> dict:
    """Converts a dictionary's NaN values to None."""
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}

def analyse_dataset(state: AgentState) -> AgentState:
    """Analysiert den Datensatz und speichert das Analyseergebnis im Zustand."""
    df: pd.DataFrame = state["dataset_df"]
    state["column_names"] = list(df.columns)

    desc_df = df.describe(include="all").T.reset_index().rename(columns={"index": "column_name"})
    desc = [
        Description(**__convert_nan_to_none(row.to_dict()))
        for _, row in desc_df.iterrows()
    ]
    state["descriptions"] = desc
    return state