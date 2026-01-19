import csv
import re
from typing import Any

import chardet
from pandas import DataFrame

from data_science_agent.graph import AgentState

import pandas as pd

from data_science_agent.pipeline.decorator.duration_tracking import track_duration

"""
Parts of this code are adopted from the Microsoft LIDA project:
    https://github.com/microsoft/lida
    
You will find in every method / function docstring a note when it was copied / adopted from LIDA.

Citation:
    cff-version: 1.2.0
    message: "If you use this software, please cite it as below."
    authors:
      - family-names: Dibia
        given-names: Victor
    title: "LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models"
    version: 1.0.0
    date-released: 2023-07-01
    url: "https://aclanthology.org/2023.acl-demo.11"
    doi: "10.18653/v1/2023.acl-demo.11"
    conference:
      name: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)"
      month: jul
      year: 2023
      address: "Toronto, Canada"
      publisher: "Association for Computational Linguistics"
"""


@track_duration
def load_dataset(state: AgentState) -> AgentState:
    """Loads dataset and saves it as a dataframe in the agent state."""
    state["dataset_df"], state["dataset_delimiter"], state["dataset_encoding"] = read_dataframe(state["dataset_path"])
    return state


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.

    ** Copy from LIDA project **

    :param col_name: The name of the column to be cleaned.
    :return: A sanitized string valid as a column name.
    """
    return re.sub(r'[^0-9a-zA-Z_]', '_', col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.

    ** Copy from LIDA project **

    :param df: The DataFrame with possibly dirty column names.
    :return: A copy of the DataFrame with clean column names.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]
    return cleaned_df


def read_dataframe(file_location: str, encoding: str = 'utf-8') -> tuple[DataFrame, str, Any]:
    """
    Read a dataframe from a given file location and clean its column names.

    ** Modified copy from LIDA project **

    :param file_location: The path to the file containing the data.
    :param encoding: Encoding to use for the file reading.
    :return: A cleaned DataFrame.
    """

    with open(file_location, 'rb') as f:
        data = f.read()
        dialect = csv.Sniffer().sniff(data[:1024].decode(errors='ignore'))
    encoding_result = chardet.detect(data)
    encoding = encoding_result['encoding']
    delimiter = dialect.delimiter

    file_extension = file_location.split('.')[-1]

    read_funcs = {
        'json': lambda: pd.read_json(file_location, orient='records', encoding=encoding),
        'csv': lambda: pd.read_csv(file_location, sep=None, engine='python', encoding=encoding),
        'xls': lambda: pd.read_excel(file_location, encoding=encoding),
        'xlsx': lambda: pd.read_excel(file_location, encoding=encoding),
        'parquet': pd.read_parquet,
        'feather': pd.read_feather,
        'tsv': lambda: pd.read_csv(file_location, sep="\t", encoding=encoding)
    }

    if file_extension not in read_funcs:
        raise ValueError('Unsupported file type')

    try:
        df = read_funcs[file_extension]()
    except Exception as e:
        print(f"Failed to read file: {file_location}. Error: {e}")
        raise

    # Clean column names
    cleaned_df = clean_column_names(df)

    if cleaned_df.columns.tolist() != df.columns.tolist():
        write_funcs = {
            'csv': lambda: cleaned_df.to_csv(file_location, index=False, encoding=encoding),
            'xls': lambda: cleaned_df.to_excel(file_location, index=False),
            'xlsx': lambda: cleaned_df.to_excel(file_location, index=False),
            'parquet': lambda: cleaned_df.to_parquet(file_location, index=False),
            'feather': lambda: cleaned_df.to_feather(file_location, index=False),
            'json': lambda: cleaned_df.to_json(file_location, orient='records', index=False, default_handler=str),
            'tsv': lambda: cleaned_df.to_csv(file_location, index=False, sep='\t', encoding=encoding)
        }

        if file_extension not in write_funcs:
            raise ValueError('Unsupported file type')

        try:
            write_funcs[file_extension]()
        except Exception as e:
            print(f"Failed to write file: {file_location}. Error: {e}")
            raise

    return cleaned_df, delimiter, encoding


def file_to_df(file_location: str):
    """
    Get summary of data from file location
    ** Copy from LIDA project **
    """
    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df