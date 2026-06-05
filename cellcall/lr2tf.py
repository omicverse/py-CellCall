"""LR2TF - Score ligand-receptor to transcription factor triple relations."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .object import CellInter


def _load_triple_relation(
    org: str = "Homo sapiens",
    is_core: bool = True,
    extdata_dir: str | None = None,
) -> pd.DataFrame:
    """Load L-R-TF triple relation data."""
    if extdata_dir is None:
        extdata_dir = os.path.join(os.path.dirname(__file__), "extdata")

    if org == "Homo sapiens":
        triple_file = "new_ligand_receptor_TFs.txt"
        triple_ext_file = "new_ligand_receptor_TFs_extended.txt"
    elif org == "Mus musculus":
        triple_file = "new_ligand_receptor_TFs_homology.txt"
        triple_ext_file = "new_ligand_receptor_TFs_homology_extended.txt"
    else:
        raise ValueError(f"Unsupported organism: {org}")

    triple = pd.read_csv(os.path.join(extdata_dir, triple_file), sep="\t", header=0)
    if not is_core:
        triple_ext = pd.read_csv(os.path.join(extdata_dir, triple_ext_file), sep="\t", header=0)
        triple = pd.concat([triple, triple_ext], ignore_index=True)
    return triple


def lr2tf(
    object: CellInter,
    sender_cell: str,
    receiver_cell: str,
    slot: str = "expr_l_r_log2_scale",
    org: str = "Homo sapiens",
    is_core: bool = True,
    extdata_dir: str | None = None,
) -> CellInter:
    """Score L-R-TF triple relations for a specific sender-receiver pair.

    Parameters
    ----------
    object : CellInter
        Object with communication profile data.
    sender_cell : str
        Sender cell type name.
    receiver_cell : str
        Receiver cell type name.
    slot : str
        Which data slot to use for L-R scores.
    org : str
        Organism name.
    is_core : bool
        Use core LR data only.
    extdata_dir : str, optional
        Path to extdata directory.

    Returns
    -------
    CellInter
        Updated object with sankey data in reductions.
    """
    my_data = object.data[slot]
    detect_genes = set(object.data["expr_mean"].index)
    regulons_matrix = object.data["regulons_matrix"]
    distance_kegg = object.data["DistanceKEGG"]
    expr_mean = object.data["expr_mean"]

    triple_relation = _load_triple_relation(org, is_core, extdata_dir)

    # Get L-R pairs with non-zero scores for this sender-receiver
    col_key = f"{sender_cell}-{receiver_cell}"
    if col_key not in my_data.columns:
        raise ValueError(f"Column '{col_key}' not found in {slot}")

    interest_df = my_data[[col_key]].copy()
    interest_df = interest_df.sort_values(col_key, ascending=False)
    interest_df = interest_df[interest_df[col_key] != 0]

    if interest_df.empty:
        object.reductions["sankey"] = pd.DataFrame(
            columns=["Ligand", "Receptor", "TF", "weight1", "weight2"]
        )
        return object

    # Parse L-R pairs
    lr_pairs = [idx.split("-", 1) for idx in interest_df.index]

    # Collect all relevant TFs
    all_tfs = set()
    for sender, receiver in lr_pairs:
        mask = (
            (triple_relation["Ligand_Symbol"] == sender)
            & (triple_relation["Receptor_Symbol"] == receiver)
        )
        tfs = triple_relation.loc[mask, "TF_Symbol"].values
        for tf in tfs:
            if tf in detect_genes:
                if tf in regulons_matrix.index and regulons_matrix.loc[tf, receiver_cell] > 0:
                    all_tfs.add(tf)

    all_tfs = sorted(all_tfs)

    # Build sankey matrix
    sankey_rows = []
    for sender, receiver in lr_pairs:
        lr_key = f"{sender}-{receiver}"

        mask = (
            (triple_relation["Ligand_Symbol"] == sender)
            & (triple_relation["Receptor_Symbol"] == receiver)
        )
        info = triple_relation.loc[mask, ["Ligand_Symbol", "Receptor_Symbol", "TF_Symbol"]]
        tfs = info["TF_Symbol"].values
        tfs = [t for t in tfs if t in detect_genes]
        tfs = [t for t in tfs if t in regulons_matrix.index and regulons_matrix.loc[t, receiver_cell] > 0]

        if not tfs:
            continue

        tf_vals = regulons_matrix.loc[tfs, receiver_cell]
        lr_val = interest_df.loc[lr_key, col_key] if lr_key in interest_df.index else 0

        # Distance-based weights
        if lr_key in distance_kegg.index:
            valid_tfs = [t for t in tfs if t in distance_kegg.columns]
            if valid_tfs:
                dist_vals = distance_kegg.loc[lr_key, valid_tfs]
                dist_vals = dist_vals.replace(0, np.inf)
                dist_inv = 1.0 / dist_vals
                w = dist_inv / dist_inv.sum()

                for tf in valid_tfs:
                    sankey_rows.append({
                        "Ligand": sender,
                        "Receptor": receiver,
                        "TF": tf,
                        "weight1": float(lr_val),
                        "weight2": float(tf_vals[tf] * w[tf]),
                    })

    if sankey_rows:
        sankey_df = pd.DataFrame(sankey_rows)
    else:
        sankey_df = pd.DataFrame(
            columns=["Ligand", "Receptor", "TF", "weight1", "weight2"]
        )

    object.reductions["sankey"] = sankey_df
    return object
