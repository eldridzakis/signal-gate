"""Helper functions for AutoML log parsing and feature discretization."""

import re
from typing import List
import pandas as pd


def operational_log_number_of_features(s: str) -> int:
    """Extracts the total number of candidate features evaluated from execution logs."""
    number_of_features = 0
    for line in s.split("\n"):
        match = re.search(r".+(Best feature).+\) of (\d+)", line)
        if match:
            number_of_features += int(match.group(2))
    return number_of_features


def get_regression_bins(df: pd.DataFrame) -> List[int]:
    """Gets regression discretization bins from a feature lift table.

    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe containing Lift columns.

    Returns
    -------
    list
        List of bin edge values.
    """
    columns = df.columns[df.columns.str.contains("Lift")]

    bins = [
        int(re.search(pattern=r"\d+", string=column).group())
        for column in columns
        if re.search(pattern=r"\d+", string=column)
    ]

    if len(columns) > 0 and re.search(string=columns[-1], pattern=r"\d+$"):
        bins.append(int(re.search(string=columns[-1], pattern=r"\d+$").group()))

    return bins


def get_binned_target(df_features: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Discretizes a continuous target variable using regression bin edges."""
    bins = get_regression_bins(df_features)
    return pd.cut(y, bins=bins)


def clean_enriched_data(
    df: pd.DataFrame,
    df_features: pd.DataFrame,
    target: str
) -> pd.DataFrame:
    """Takes an enriched dataframe and returns target and generated features."""
    df_cleaned = df[[target] + df.columns[-df_features.shape[0]:].tolist()]
    return df_cleaned
