"""Normalization functions: CPM for 10X, TPM/RPKM for Smart-seq2."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


def _load_gene_lengths(org: str, extdata_dir: str | None = None) -> pd.DataFrame:
    """Load gene length table for the given organism."""
    if extdata_dir is None:
        extdata_dir = os.path.join(os.path.dirname(__file__), "extdata")
    if org == "Homo sapiens":
        path = os.path.join(extdata_dir, "homo", "transcriptmaxLength.Rdata")
    elif org == "Mus musculus":
        path = os.path.join(extdata_dir, "mmus", "transcriptmaxLength.Rdata")
    else:
        raise ValueError(f"Unsupported organism: {org}")
    # Rdata files need rpy2 or rdata package; try rdata first
    try:
        import rdata
        parsed = rdata.read_rda(path)
        # Rdata returns a dict; the variable name varies
        for key, val in parsed.items():
            if isinstance(val, pd.DataFrame):
                return val
            # sometimes wrapped in OrderedDict
            return list(parsed.values())[0]
    except ImportError:
        raise ImportError(
            "Install 'rdata' package to load .Rdata files: pip install rdata"
        )


def counts2normalized_10x(
    data: pd.DataFrame,
    to_type: str = "CPM",
    scale_factor: float = 1e6,
) -> pd.DataFrame:
    """Normalize count matrix to CPM (for 10X data).

    Parameters
    ----------
    data : pd.DataFrame
        Count matrix with genes as rows, cells as columns.
    to_type : str
        Normalization type, only "CPM" supported.
    scale_factor : float
        Scaling factor (default 1e6).

    Returns
    -------
    pd.DataFrame
        Normalized expression matrix.
    """
    if to_type != "CPM":
        raise ValueError(f"Only CPM supported for 10X, got {to_type}")

    total_count = data.sum(axis=0)
    cpm = data.div(total_count, axis=1) * scale_factor
    return cpm


def counts2normalized_smartseq2(
    data: pd.DataFrame,
    org: str = "Homo sapiens",
    to_type: str = "TPM",
    scale_factor: float = 1e6,
    extdata_dir: str | None = None,
) -> pd.DataFrame:
    """Normalize count matrix to TPM or RPKM (for Smart-seq2 data).

    Parameters
    ----------
    data : pd.DataFrame
        Count matrix with genes as rows, cells as columns.
    org : str
        Organism name.
    to_type : str
        "TPM" or "RPKM".
    scale_factor : float
        Scaling factor.
    extdata_dir : str, optional
        Path to extdata directory.

    Returns
    -------
    pd.DataFrame
        Normalized expression matrix.
    """
    g_l = _load_gene_lengths(org, extdata_dir)
    common_genes = data.index.intersection(g_l.iloc[:, 0])
    expr = data.loc[common_genes]
    lengths = g_l.set_index(g_l.columns[0]).loc[common_genes, g_l.columns[1]].values

    if to_type == "RPKM":
        total_count = expr.sum(axis=0)
        rpkm = expr.div(lengths, axis=0).div(total_count, axis=1) * 1e9
        return rpkm
    elif to_type == "TPM":
        rpk = expr.div(lengths, axis=0)
        total_rpk = rpk.sum(axis=0)
        tpm = rpk.div(total_rpk, axis=1) * scale_factor
        return tpm
    else:
        raise ValueError(f"to_type must be 'RPKM' or 'TPM', got {to_type}")
