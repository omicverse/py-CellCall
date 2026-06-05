"""Visualization functions: bubble plot, pheatmap, getForBubble."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_for_bubble(
    pathway_hyper_list: list[pd.DataFrame],
    cella_cellb: list[str],
    extdata_dir: str | None = None,
) -> pd.DataFrame:
    """Melt enrichment dataframes for bubble graph.

    Parameters
    ----------
    pathway_hyper_list : list[pd.DataFrame]
        List of pathway enrichment results (from get_hyper_pathway).
    cella_cellb : list[str]
        List of cell-cell pair labels.
    extdata_dir : str, optional
        Path to extdata directory.

    Returns
    -------
    pd.DataFrame
        Melted DataFrame with columns: pathway, cc, pvalue, NES.
    """
    if extdata_dir is None:
        extdata_dir = os.path.join(os.path.dirname(__file__), "extdata")

    # Merge all pathway results
    merged = pathway_hyper_list[0].copy()
    for i in range(1, len(pathway_hyper_list)):
        merged = merged.merge(
            pathway_hyper_list[i],
            on="pathway",
            how="outer",
            suffixes=("", f"_{i}"),
        )

    merged.index = merged["pathway"]

    # Extract pvalue columns (every 3rd starting from 0)
    pvalue_cols = [c for c in merged.columns if c.startswith("Pvalue")]
    nes_cols = [c for c in merged.columns if c.startswith("NES")]

    pvalue_df = merged[pvalue_cols].copy()
    pvalue_df.columns = cella_cellb[:len(pvalue_cols)]
    pvalue_df = pvalue_df.fillna(1)

    nes_df = merged[nes_cols].copy()
    nes_df.columns = cella_cellb[:len(nes_cols)]
    nes_df = nes_df.fillna(0)

    # Filter out zero-NES rows
    nonzero_mask = nes_df.sum(axis=1) != 0
    nes_df = nes_df[nonzero_mask]
    pvalue_df = pvalue_df[nonzero_mask]

    # Load pathway names
    kegg_path = os.path.join(extdata_dir, "KEGG_SYMBOL_ID.txt")
    pathway_info = pd.read_csv(kegg_path, sep="\t", header=None)
    pathway_info.columns = ["id", "name", "main_object_name", "sub_object_name"]
    pathway_lookup = dict(zip(pathway_info["id"], pathway_info["name"]))

    # Melt pvalue
    pvalue_df = pvalue_df.reset_index()
    pvalue_df["pathway"] = pvalue_df["pathway"].map(pathway_lookup).fillna(pvalue_df["pathway"])
    pvalue_melted = pvalue_df.melt(id_vars=["pathway"], var_name="cc", value_name="pvalue")

    # Melt NES
    nes_df = nes_df.reset_index()
    nes_df["pathway"] = nes_df["pathway"].map(pathway_lookup).fillna(nes_df["pathway"])
    nes_melted = nes_df.melt(id_vars=["pathway"], var_name="cc", value_name="NES")
    nes_melted["NES"] = nes_melted["NES"].clip(0, 2)

    # Merge
    result = pvalue_melted.merge(nes_melted, on=["pathway", "cc"])
    return result


def plot_bubble(dat: pd.DataFrame, figsize: tuple = (8, 6)) -> plt.Figure:
    """Plot bubble graph of pathway enrichment.

    Parameters
    ----------
    dat : pd.DataFrame
        DataFrame from get_for_bubble with columns: pathway, cc, pvalue, NES.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    df = dat.copy()
    df["logPvalue"] = -np.log10(df["pvalue"].clip(lower=1e-10))
    df.loc[df["logPvalue"] >= 2, "logPvalue"] = 2
    df.loc[(df["logPvalue"] >= 1.3) & (df["logPvalue"] < 2), "logPvalue"] = 1.3
    df.loc[df["logPvalue"] < 1.3, "logPvalue"] = 1

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        df["cc"], df["pathway"],
        c=df["NES"], s=df["logPvalue"] * 50,
        cmap="RdYlBu_r", edgecolors="black", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="NES")
    ax.set_xlabel("Cell-Cell pair")
    ax.set_ylabel("Pathway")
    ax.set_title("Pathway Enrichment")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def view_pheatmap(
    data: pd.DataFrame,
    show_rownames: bool = True,
    show_colnames: bool = True,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    figsize: tuple = (10, 8),
    cmap: str = "YlOrRd",
    title: str = "score",
) -> plt.Figure:
    """Plot heatmap of communication scores.

    Parameters
    ----------
    data : pd.DataFrame
        Score matrix.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_data = data.copy()
    if cluster_rows:
        from scipy.cluster.hierarchy import linkage, leaves_list
        if len(plot_data) > 1:
            Z = linkage(plot_data.values, method="average")
            order = leaves_list(Z)
            plot_data = plot_data.iloc[order]
    if cluster_cols:
        from scipy.cluster.hierarchy import linkage, leaves_list
        if len(plot_data.columns) > 1:
            Z = linkage(plot_data.values.T, method="average")
            order = leaves_list(Z)
            plot_data = plot_data.iloc[:, order]

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=cmap,
        xticklabels=show_colnames,
        yticklabels=show_rownames,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig
