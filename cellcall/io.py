"""I/O utilities for loading R data and reference outputs."""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd


def load_rdata(path: str) -> dict:
    """Load an R .Rdata or .rda file.

    Requires the 'rdata' package: pip install rdata

    Parameters
    ----------
    path : str
        Path to .Rdata file.

    Returns
    -------
    dict
        Dictionary of variable names to values.
    """
    import rdata
    parsed = rdata.read_rda(path)
    result = {}
    for key, val in parsed.items():
        result[key] = val
    return result


def load_example_data(extdata_dir: str | None = None) -> dict:
    """Load the example_Data.Rdata test fixture.

    Returns
    -------
    dict
        Dictionary with loaded R objects.
    """
    if extdata_dir is None:
        extdata_dir = os.path.join(os.path.dirname(__file__), "extdata")
    path = os.path.join(extdata_dir, "example_Data.Rdata")
    return load_rdata(path)


def load_reference_json(path: str) -> dict:
    """Load reference output JSON produced by R driver.

    Parameters
    ----------
    path : str
        Path to reference JSON file.

    Returns
    -------
    dict
        Reference output data.
    """
    with open(path, "r") as f:
        return json.load(f)


def matrix_to_json(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serializable format."""
    return {
        "index": df.index.tolist(),
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
    }


def json_to_matrix(data: dict) -> pd.DataFrame:
    """Convert JSON format back to DataFrame."""
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
