"""Log2 fold change calculation per cell type."""

from __future__ import annotations
import numpy as np
import pandas as pd


def mylog2foldchange(
    in_data: pd.DataFrame,
    cell_types: list[str],
    method: str = "median",
    probs: float = 0.75,
) -> dict[str, pd.DataFrame]:
    """Calculate log2 fold change for each cell type vs all others.

    Parameters
    ----------
    in_data : pd.DataFrame
        Expression matrix (genes x cells).
    cell_types : list[str]
        List of cell type names.
    method : str
        "median" or "mean".
    probs : float
        Quantile probability for median method.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping cell type to DataFrame with columns [gene_id, log2fc].
    """
    result = {}
    for ct in cell_types:
        case_mask = in_data.columns == ct
        control_mask = ~case_mask

        if method == "mean":
            case_mean = in_data.loc[:, case_mask].mean(axis=1)
            control_mean = in_data.loc[:, control_mask].mean(axis=1)
            fc = (case_mean + 1) / (control_mean + 1)
        elif method == "median":
            case_vals = in_data.loc[:, case_mask]
            control_vals = in_data.loc[:, control_mask]
            case_q = case_vals.quantile(probs, axis=1)
            control_q = control_vals.quantile(probs, axis=1)
            fc = (case_q + 1) / (control_q + 1)
        else:
            raise ValueError(f"method must be 'median' or 'mean', got {method}")

        log2fc = np.log2(fc)
        log2fc = log2fc.replace([np.inf, -np.inf], 0).fillna(0)

        res = pd.DataFrame({"gene_id": in_data.index, "log2fc": log2fc.values})
        result[ct] = res
    return result
