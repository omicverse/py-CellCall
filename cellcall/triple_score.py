"""Transform L-R-TF triple relations to scores."""

from __future__ import annotations
import pandas as pd


def trans2triple_score(df: pd.DataFrame) -> pd.DataFrame:
    """Transform sankey dataframe to triple scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Ligand, Receptor, TF, weight1, weight2.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Ligand, Receptor, TF, value.
    """
    df = df.copy()
    df.columns = ["Ligand", "Receptor", "TF", "weight1", "weight2"]
    df = df[(df["weight1"] != 0) & (df["weight2"] != 0)]

    df["Ligand"] = "sender:" + df["Ligand"].astype(str)
    df["Receptor"] = "receiver:" + df["Receptor"].astype(str)
    df["TF"] = "TF:" + df["TF"].astype(str)

    # Normalize weight2 by group
    sum_tf = df.groupby(["Ligand", "Receptor"])["weight2"].sum().reset_index()
    sum_tf.columns = ["Ligand", "Receptor", "Score"]

    normalized_w2 = []
    for _, row in df.iterrows():
        score = sum_tf.loc[
            (sum_tf["Ligand"] == row["Ligand"]) & (sum_tf["Receptor"] == row["Receptor"]),
            "Score",
        ].values[0]
        normalized_w2.append(float(row["weight2"]) / score if score != 0 else 0)

    df["weight2"] = normalized_w2
    df["value"] = df["weight1"] * df["weight2"]
    result = df[["Ligand", "Receptor", "TF", "value"]].copy()
    return result
