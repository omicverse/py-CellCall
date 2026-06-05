"""KEGG pathway distance computation between ligand-receptor pairs and TFs.

Vectorized implementation for speed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def get_distance_kegg(
    triple_relation: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """Compute distance between LR pairs and TFs from KEGG pathway data.

    Parameters
    ----------
    triple_relation : pd.DataFrame
        Must contain columns at indices 4, 5 for Ligand_Symbol, Receptor_Symbol,
        TF_Symbol, and a column with KEGG path distances (index 3, comma-separated
        path_distance pairs like "path1_3,path2_5").
    method : str
        "mean" or "median" to aggregate multiple path distances.

    Returns
    -------
    pd.DataFrame
        Distance matrix (LR pairs x TFs).
    """
    lr_inter = triple_relation.iloc[:, [4, 5]].drop_duplicates()
    lr_labels = (lr_inter.iloc[:, 0].astype(str) + "-" + lr_inter.iloc[:, 1].astype(str)).values

    tf_symbols = triple_relation["TF_Symbol"].unique()

    dist_matrix = pd.DataFrame(0.0, index=lr_labels, columns=tf_symbols)

    # Vectorized: group by LR-TF pair and compute mean distance
    triple_relation = triple_relation.copy()
    triple_relation["_lr_key"] = (
        triple_relation.iloc[:, 4].astype(str) + "-" + triple_relation.iloc[:, 5].astype(str)
    )

    # Parse distances vectorized
    def _parse_distances(path_str: str, method: str = "mean") -> float:
        if not path_str or path_str == "nan":
            return 0.0
        parts = str(path_str).split(",")
        distances = []
        for part in parts:
            if "_" in part:
                try:
                    d = float(part.rsplit("_", 1)[1])
                    distances.append(d)
                except (ValueError, IndexError):
                    continue
        if not distances:
            return 0.0
        return float(np.mean(distances)) if method == "mean" else float(np.median(distances))

    # Apply vectorized
    path_col = triple_relation.iloc[:, 3]
    distances = path_col.apply(lambda x: _parse_distances(x, method))

    for i, row in triple_relation.iterrows():
        lr_key = row["_lr_key"]
        tf_key = row["TF_Symbol"]
        d = distances[i]
        if lr_key in dist_matrix.index and tf_key in dist_matrix.columns:
            dist_matrix.loc[lr_key, tf_key] = d

    return dist_matrix
