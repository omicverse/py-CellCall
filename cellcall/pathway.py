"""Pathway enrichment analysis using hypergeometric test."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats


def _load_triple_relation_with_pathway(
    org: str = "Homo sapiens",
    is_core: bool = True,
    extdata_dir: str | None = None,
) -> pd.DataFrame:
    """Load L-R-TF triple relation data including pathway_ID."""
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


def get_hyper_pathway(
    data: pd.DataFrame,
    object,
    cella_cellb: str,
    is_core: bool = True,
    org: str = "Homo sapiens",
    extdata_dir: str | None = None,
) -> pd.DataFrame:
    """Enrich communication relations on KEGG pathways using hypergeometric test.

    Parameters
    ----------
    data : pd.DataFrame
        Communication score matrix (LR pairs x cell-cell pairs).
    object : CellInter
        Object with regulons_matrix data.
    cella_cellb : str
        Cell-cell pair to analyze (e.g., "T-B").
    is_core : bool
        Use core LR data only.
    org : str
        Organism name.
    extdata_dir : str, optional
        Path to extdata directory.

    Returns
    -------
    pd.DataFrame
        Pathway enrichment results with columns: Pvalue, Jaccard, NES, pathway.
    """
    n = data
    triple_relation = _load_triple_relation_with_pathway(org, is_core, extdata_dir)
    regulons_matrix = object.data["regulons_matrix"]

    # Build triple-pathway mapping
    tmp = triple_relation.copy()
    tmp["triple"] = (
        tmp["Ligand_Symbol"].astype(str) + "-"
        + tmp["Receptor_Symbol"].astype(str) + "-"
        + tmp["TF_Symbol"].astype(str)
    )

    # Expand pathway IDs
    path_triple_rows = []
    for _, row in tmp.iterrows():
        pathway_str = str(row.get("pathway_ID", ""))
        if not pathway_str or pathway_str == "nan":
            continue
        paths = [p.strip() for p in pathway_str.split(",") if p.strip()]
        for p in paths:
            path_triple_rows.append({"path": p, "triple": row["triple"]})

    if not path_triple_rows:
        return pd.DataFrame({"Pvalue": [1], "Jaccard": [0], "NES": [0], "pathway": ["unknown"]})

    path_triple_df = pd.DataFrame(path_triple_rows)

    # For each LR pair with score > 0 in the given cella_cellb
    path_list = []
    for lr_idx in n.index:
        scores = n.loc[lr_idx, :]
        active_cols = scores[scores > 0].index.tolist()
        if not active_cols:
            continue

        lr_parts = lr_idx.split("-", 1)
        if len(lr_parts) != 2:
            continue
        l_tmp, r_tmp = lr_parts

        # Find triple info for this L-R pair
        mask = (
            (triple_relation["Ligand_Symbol"] == l_tmp)
            & (triple_relation["Receptor_Symbol"] == r_tmp)
        )
        a_tmp = triple_relation.loc[mask, ["pathway_ID", "TF_Symbol"]]
        if a_tmp.empty:
            continue

        for _, a_row in a_tmp.iterrows():
            tf = a_row["TF_Symbol"]
            pathway_ids = str(a_row.get("pathway_ID", ""))
            if not pathway_ids or pathway_ids == "nan":
                continue

            # Check which receiver cell types have this TF active
            for cc in active_cols:
                cc_parts = cc.split("-", 1)
                if len(cc_parts) != 2:
                    continue
                receiver = cc_parts[1]
                if tf in regulons_matrix.index and regulons_matrix.loc[tf, receiver] > 0:
                    for p in pathway_ids.split(","):
                        p = p.strip()
                        if p:
                            path_list.append({
                                "triple": lr_idx + "-" + tf,
                                "path": p,
                                "cc": cc,
                                "enrichment_score": regulons_matrix.loc[tf, receiver],
                            })

    if not path_list:
        return pd.DataFrame({"Pvalue": [1], "Jaccard": [0], "NES": [0], "pathway": ["unknown"]})

    path_df = pd.DataFrame(path_list)

    # Get triples for the specific cell pair
    cc_triple = path_df[path_df["cc"] == cella_cellb][["triple", "path"]].drop_duplicates()

    if cc_triple.empty:
        return pd.DataFrame({"Pvalue": [1], "Jaccard": [0], "NES": [0], "pathway": ["unknown"]})

    # Hypergeometric test per pathway
    all_triples = set(path_triple_df["triple"].unique())
    cc_triples = set(cc_triple["triple"].unique())
    pathways = cc_triple["path"].unique()

    results = []
    for pathway in pathways:
        pathway_triples = set(
            path_triple_df.loc[path_triple_df["path"] == pathway, "triple"].unique()
        )
        cc_path_triples = set(
            cc_triple.loc[cc_triple["path"] == pathway, "triple"].unique()
        )

        intersect_size = len(pathway_triples & cc_path_triples)
        union_size = len(pathway_triples | cc_path_triples)

        # Hypergeometric test
        q = intersect_size - 1
        m = len(pathway_triples)
        nn = len(all_triples) - m
        k = len(cc_triples)

        pvalue = stats.hypergeom.sf(q, m + nn, m, k) if k > 0 else 1.0
        jaccard = intersect_size / union_size if union_size > 0 else 0.0

        results.append({"Pvalue": pvalue, "Jaccard": jaccard})

    res_df = pd.DataFrame(results, index=pathways)

    # Compute NES
    all_path_jaccard = []
    for p in path_triple_df["path"].unique():
        p_triples = set(path_triple_df.loc[path_triple_df["path"] == p, "triple"].unique())
        cc_p_triples = set(cc_triple.loc[cc_triple["path"] == p, "triple"].unique())
        j = len(p_triples & cc_p_triples) / len(p_triples | cc_p_triples) if len(p_triples | cc_p_triples) > 0 else 0
        all_path_jaccard.append(j)

    if all_path_jaccard:
        mean_j = np.mean(all_path_jaccard)
        std_j = np.std(all_path_jaccard, ddof=0)
        if std_j > 0:
            res_df["NES"] = (res_df["Jaccard"] - mean_j) / std_j
        else:
            res_df["NES"] = 0.0
    else:
        res_df["NES"] = 0.0

    res_df["pathway"] = res_df.index
    return res_df
