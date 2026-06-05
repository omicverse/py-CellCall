"""TF-target gene correlation filtering using Spearman correlation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def get_correlated_genes(
    data: pd.DataFrame,
    cell_type: str,
    tf: str,
    target_list: list[str],
    p_value: float = 0.05,
    cor_value: float = 0.0,
    top_gene: float = 0.1,
) -> list[str]:
    """Filter target genes of a TF by Spearman correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Expression matrix (genes x cells).
    cell_type : str
        Cell type to subset.
    tf : str
        Transcription factor gene name.
    target_list : list[str]
        Candidate target genes.
    p_value : float
        P-value threshold.
    cor_value : float
        Correlation threshold.
    top_gene : float
        Fraction of top correlated genes to keep (1.0 = keep all).

    Returns
    -------
    list[str]
        Filtered target gene names.
    """
    if tf not in data.index:
        return []

    ct_cols = data.columns[data.columns == cell_type]
    if len(ct_cols) == 0:
        return []

    expr_tmp = data[ct_cols]

    tf_expr = expr_tmp.loc[tf].values.astype(float)
    target_expr = expr_tmp.loc[expr_tmp.index.isin(target_list)]

    if target_expr.empty:
        return []

    # Compute Spearman correlation: tf vs each target
    valid_targets = []
    valid_corr = []
    valid_p = []

    for gene in target_expr.index:
        gene_expr = target_expr.loc[gene].values.astype(float)
        if np.std(gene_expr) == 0 or np.std(tf_expr) == 0:
            continue
        corr, p = stats.spearmanr(tf_expr, gene_expr)
        if np.isnan(corr):
            continue
        valid_targets.append(gene)
        valid_corr.append(corr)
        valid_p.append(p)

    if not valid_targets:
        return []

    corr_df = pd.DataFrame({"gene": valid_targets, "corr": valid_corr, "p": valid_p})

    # Filter by p-value and correlation
    filtered = corr_df[(corr_df["p"] < p_value) & (corr_df["corr"] > cor_value)]

    if len(filtered) <= 1:
        return []

    # Sort by correlation descending
    filtered = filtered.sort_values("corr", ascending=False)

    # Take top fraction (skip the first one which is TF itself if present)
    n_top = int(np.floor(top_gene * len(filtered)))
    genes = filtered["gene"].tolist()

    if len(genes) > n_top + 1:
        genes = genes[1 : n_top + 1]
    return genes
