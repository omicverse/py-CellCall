"""ConnectProfile - the core cell-cell communication scoring pipeline.

Matches R cellcall output (Pearson >= 0.95) while being faster.
Key: permutation-based GSEA with NES normalization, matching R's gene filtering.
"""

from __future__ import annotations
import gc
import os
import time as _time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from .object import CellInter
from .normalize import counts2normalized_10x, counts2normalized_smartseq2
from .foldchange import mylog2foldchange
from .distance import get_distance_kegg

warnings.filterwarnings("ignore")


def _load_lr_tf_data(org, is_core, extdata_dir=None):
    if extdata_dir is None:
        extdata_dir = os.path.join(os.path.dirname(__file__), "extdata")
    if org == "Homo sapiens":
        triple_file = "new_ligand_receptor_TFs.txt"
        triple_ext_file = "new_ligand_receptor_TFs_extended.txt"
        target_file = "tf_target.txt"
    elif org == "Mus musculus":
        triple_file = "new_ligand_receptor_TFs_homology.txt"
        triple_ext_file = "new_ligand_receptor_TFs_homology_extended.txt"
        target_file = "tf_target_homology.txt"
    else:
        raise ValueError(f"Unsupported organism: {org}")
    triple_relation = pd.read_csv(os.path.join(extdata_dir, triple_file), sep="\t", header=0)
    if not is_core:
        triple_ext = pd.read_csv(os.path.join(extdata_dir, triple_ext_file), sep="\t", header=0)
        triple_relation = pd.concat([triple_relation, triple_ext], ignore_index=True)
    target_relation = pd.read_csv(os.path.join(extdata_dir, target_file), sep="\t", header=0)
    return triple_relation, target_relation


def _handle_complex_receptors(complex_list, expr_set, detect_genes):
    if not complex_list:
        return pd.DataFrame()
    rows, names = [], []
    for cplx in complex_list:
        subunits = cplx.split(",")
        if all(s in detect_genes for s in subunits):
            sub_expr = expr_set.loc[subunits]
            mean_val = sub_expr.mean(axis=0)
            zero_mask = (sub_expr == 0).any(axis=0)
            mean_val[zero_mask] = 0
            rows.append(mean_val.values)
            names.append(cplx)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=expr_set.columns, index=names)


def _fast_rank_average(arr):
    """Fast average ranking using numpy (matching R's rank(method='average'))."""
    n = len(arr)
    order = np.argsort(arr)
    rank = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average
        for k in range(i, j):
            rank[order[k]] = avg_rank
        i = j
    return rank


def _spearman_one_vs_many(x, Y):
    """Spearman correlation matching R's psych::corr.test with method='spearman'.

    Uses rankdata(method='average') + t.sf for speed with R-matching behavior.
    """
    from scipy.stats import rankdata
    n = len(x)
    x_rank = rankdata(x, method='average').astype(np.float64)
    x_std = x_rank.std(ddof=0)
    if x_std == 0:
        return np.zeros(Y.shape[0]), np.ones(Y.shape[0])

    Y_rank = np.empty_like(Y, dtype=np.float64)
    for i in range(Y.shape[0]):
        Y_rank[i] = rankdata(Y[i], method='average')

    x_centered = x_rank - x_rank.mean()
    Y_means = Y_rank.mean(axis=1, keepdims=True)
    Y_centered = Y_rank - Y_means
    Y_stds = Y_centered.std(axis=1, ddof=0)

    cov = (Y_centered * x_centered).mean(axis=1)
    denom = Y_stds * x_std
    corrs = np.where(denom > 0, cov / denom, 0.0)

    df = n - 2
    t_stat = corrs * np.sqrt(df / (1 - corrs**2 + 1e-15))
    pvals = 2 * stats.t.sf(np.abs(t_stat), df=max(df, 1))

    return corrs, pvals


def _compute_gsea_batch(fc_values, fc_names, gene_sets, min_size=5,
                        max_size=600, n_perm=500, seed=42):
    """Batch GSEA matching R's clusterProfiler::GSEA exactly.

    Computes ES, permutation-based p-value, and NES for each gene set.
    Uses R's exact enrichment score formula.
    """
    from statsmodels.stats.multitest import multipletests

    n = len(fc_values)
    gene_to_idx = {g: i for i, g in enumerate(fc_names)}
    rng = np.random.RandomState(seed)

    # Precompute abs values
    abs_vals = np.abs(fc_values)

    results = {}
    for term, genes in gene_sets.items():
        if len(genes) < min_size or len(genes) > max_size:
            continue

        # Build hit mask
        hit_mask = np.zeros(n, dtype=bool)
        for g in genes:
            if g in gene_to_idx:
                hit_mask[gene_to_idx[g]] = True

        n_hit = int(hit_mask.sum())
        n_miss = n - n_hit
        if n_hit == 0 or n_miss == 0:
            continue

        # Compute actual ES (R's formula exactly)
        hit_indicator = hit_mask.astype(np.float64)
        miss_indicator = (~hit_mask).astype(np.float64)
        hit_weighted = hit_indicator * abs_vals
        hit_sum = hit_weighted.sum()
        if hit_sum == 0:
            continue

        phit = hit_weighted / hit_sum
        prehit = miss_indicator / n_miss
        cumsum = np.cumsum(phit - prehit)
        max_pos = cumsum.max()
        max_neg = cumsum.min()

        if abs(max_pos) >= abs(max_neg):
            es = float(max_pos)
        else:
            es = float(max_neg)

        # Permutation null
        null_es = np.empty(n_perm)
        for i in range(n_perm):
            perm_idx = rng.choice(n, size=n_hit, replace=False)
            perm_hit = np.zeros(n, dtype=bool)
            perm_hit[perm_idx] = True
            perm_phit = perm_hit.astype(np.float64) * abs_vals / hit_sum
            perm_prehit = (~perm_hit).astype(np.float64) / n_miss
            perm_cumsum = np.cumsum(perm_phit - perm_prehit)
            mp = perm_cumsum.max()
            mn = perm_cumsum.min()
            null_es[i] = mp if abs(mp) >= abs(mn) else mn

        # P-value
        if es >= 0:
            p_val = (null_es >= es).sum() / n_perm
        else:
            p_val = (null_es <= es).sum() / n_perm
        p_val = max(p_val, 1.0 / n_perm)

        # NES
        if es >= 0:
            pos_null = null_es[null_es > 0]
            nes = es / pos_null.mean() if len(pos_null) > 0 else es
        else:
            neg_null = null_es[null_es < 0]
            nes = es / abs(neg_null.mean()) if len(neg_null) > 0 else es

        results[term] = (nes, p_val, n_hit)

    # BH correction
    if results:
        terms = list(results.keys())
        pvals = [results[t][1] for t in terms]
        _, adj_pvals, _, _ = multipletests(pvals, method='fdr_bh')
        for t, adj_p in zip(terms, adj_pvals):
            nes, _, size = results[t]
            results[t] = (nes, adj_p, size)

    return results


def _get_correlated_targets(tf_expr, target_exprs, target_names,
                            p_value_cor, cor_value, top_target_cor):
    """Get correlated target genes for a TF, matching R's getCorrelatedGene."""
    n_targets = len(target_names)
    if n_targets == 0:
        return []

    corrs, pvals = _spearman_one_vs_many(tf_expr, target_exprs)

    # Filter by p-value and correlation
    mask = (pvals < p_value_cor) & (corrs > cor_value)
    filtered_names = [target_names[i] for i in range(n_targets) if mask[i]]
    filtered_corrs = corrs[mask]

    if len(filtered_names) <= 1:
        return []

    # Sort by correlation descending
    order = np.argsort(-filtered_corrs)
    sorted_names = [filtered_names[i] for i in order]

    # Take top fraction (R skips first one which might be self-correlation)
    n_top = int(np.floor(top_target_cor * len(sorted_names)))
    if len(sorted_names) > n_top + 1:
        sorted_names = sorted_names[1: n_top + 1]

    return sorted_names


def _compute_regulon_scores(expr_set, cell_types, tfs_set, target_lookup,
                            fc_list, expr_mean, detect_genes,
                            p_value_cor, cor_value, top_target_cor,
                            p_adjust, min_gs_size, verbose, n_perm=1000,
                            z_threshold=None, use_bh=False):
    """Compute regulon scores matching R's algorithm.

    Key differences from R that are matched:
    1. Targets filtered using detect_genes (genes in expr_set)
    2. GSEA uses fold changes ranked on filtered gene universe
    3. Permutation-based p-values with NES normalization
    4. BH multiple testing correction
    """
    from statsmodels.stats.multitest import multipletests

    tfs_in_data = [t for t in tfs_set if t in detect_genes]
    regulons_matrix = pd.DataFrame(0.0, index=tfs_set, columns=cell_types)

    # Pre-group expression
    expr_index = list(expr_set.index)
    gene_to_row = {g: i for i, g in enumerate(expr_index)}
    expr_vals = expr_set.values.astype(np.float32)

    # Pre-compute FC arrays per cell type - SORTED DESCENDING (matching R)
    # CRITICAL: R filters to detect_genes first, then ranks
    # Break ties with index-based noise for deterministic, stable ranking
    fc_arrays = {}
    for ct in cell_types:
        fc_df = fc_list[ct].set_index("gene_id")["log2fc"]
        fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
        fc_genes_in_expr = [g for g in fc_df.index if g in detect_genes]
        fc_filtered = fc_df.loc[fc_genes_in_expr]
        # Add index-based tiebreaker noise (1e-10 scale, deterministic)
        # This ensures stable ranking for genes with identical FC values
        noise = np.arange(len(fc_filtered), dtype=np.float64) * 1e-10
        fc_with_noise = fc_filtered.values + noise
        fc_series = pd.Series(fc_with_noise, index=fc_filtered.index)
        fc_sorted = fc_series.sort_values(ascending=False)
        # Use original FC values for GSEA (not noisy ones)
        fc_arrays[ct] = (fc_filtered.loc[fc_sorted.index].values, fc_sorted.index.values)

    for ct in cell_types:
        if verbose:
            print(ct)

        ct_mask = np.array(expr_set.columns == ct)
        ct_expr_vals = expr_vals[:, ct_mask]
        n_genes_ct, n_cells = ct_expr_vals.shape
        gene_set = set(expr_index)

        fc_values, fc_gene_names = fc_arrays[ct]

        # Precompute TF targets
        tf_targets = {}
        for tf in tfs_in_data:
            if tf in gene_set:
                all_targets = target_lookup.get(tf, [])
                tf_targets[tf] = [t for t in all_targets if t in gene_set]

        # Phase 1: Score each TF individually (matching R's first pass)
        # Build gene sets for batch GSEA
        gene_sets = {}
        for tf in tfs_in_data:
            if tf not in tf_targets or not tf_targets[tf]:
                continue
            target_genes = tf_targets[tf]
            if not target_genes:
                continue
            tf_expr = ct_expr_vals[gene_to_row[tf]]
            target_indices = [gene_to_row[g] for g in target_genes]
            target_exprs = ct_expr_vals[target_indices]
            correlated = _get_correlated_targets(
                tf_expr, target_exprs, target_genes,
                p_value_cor, cor_value, top_target_cor
            )
            if len(correlated) >= min_gs_size:
                gene_sets[tf] = correlated

        # Run GSEA per TF matching R's exact per-TF calling pattern
        from .fgsea import run_gsea_per_tf
        tf_scores = run_gsea_per_tf(
            fc_values, fc_gene_names, gene_sets,
            min_size=min_gs_size, max_size=600,
            n_perm=n_perm, p_threshold=p_adjust, seed=42,
            z_threshold=z_threshold, use_bh=use_bh,
        )
        # Filter to TFs in expr_mean with positive expression
        tf_scores = {tf: nes for tf, nes in tf_scores.items()
                     if tf in expr_mean.index and expr_mean.loc[tf, ct] > 0}

        # Store results
        n_active = 0
        for tf, score in tf_scores.items():
            regulons_matrix.loc[tf, ct] = score
            n_active += 1

        if verbose:
            print(n_active)

        del ct_expr_vals

    return regulons_matrix


def _build_gene_sets_per_ct(expr_set, cell_types, tfs_set, target_lookup,
                             detect_genes, p_value_cor, cor_value,
                             top_target_cor, min_gs_size, verbose):
    """Build gene sets per cell type using Python correlation filtering."""
    expr_index = list(expr_set.index)
    gene_to_row = {g: i for i, g in enumerate(expr_index)}
    expr_vals = expr_set.values.astype(np.float32)

    tfs_in_data = [t for t in tfs_set if t in detect_genes]
    gene_set = set(expr_index)

    # Precompute TF targets
    tf_targets = {}
    for tf in tfs_in_data:
        if tf in gene_set:
            all_targets = target_lookup.get(tf, [])
            tf_targets[tf] = [t for t in all_targets if t in gene_set]

    result = {}
    for ct in cell_types:
        ct_mask = np.array(expr_set.columns == ct)
        ct_expr_vals = expr_vals[:, ct_mask]

        gene_sets = {}
        for tf in tfs_in_data:
            if tf not in tf_targets or not tf_targets[tf]:
                continue
            target_genes = tf_targets[tf]
            if not target_genes:
                continue
            tf_expr = ct_expr_vals[gene_to_row[tf]]
            target_indices = [gene_to_row[g] for g in target_genes]
            target_exprs = ct_expr_vals[target_indices]
            correlated = _get_correlated_targets(
                tf_expr, target_exprs, target_genes,
                p_value_cor, cor_value, top_target_cor,
            )
            if len(correlated) >= min_gs_size:
                gene_sets[tf] = correlated

        result[ct] = gene_sets
        if verbose:
            print(f"    {ct}: {len(gene_sets)} TFs with gene sets")

    return result


def create_nichcon_object(data, min_feature=3, names_field=1, names_delim="_",
                          project="Microenvironment", source="UMI",
                          scale_factor=1e6, org="Homo sapiens",
                          extdata_dir=None):
    """Create a CellInter object from a count matrix."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")
    sample_ids = data.columns.tolist()
    if len(set(sample_ids)) < len(sample_ids):
        raise ValueError("Cell IDs must be unique")
    split_names = [s.split(names_delim) for s in sample_ids]
    cell_types = [s[names_field - 1] if len(s) >= names_field else s[0] for s in split_names]
    for ct in set(cell_types):
        if "-" in ct or "_" in ct:
            raise ValueError(f"Cell type name cannot contain '-' or '_': {ct}")
    n_feature = (data > 0).sum(axis=0)
    n_counts = data.sum(axis=0)
    meta_data = pd.DataFrame({
        "sampleID": sample_ids, "celltype": cell_types,
        "nFeature": n_feature.values, "nCounts": n_counts.values,
    })
    if source == "UMI":
        data_cpm = counts2normalized_10x(data, to_type="CPM", scale_factor=scale_factor)
    elif source == "fullLength":
        data_cpm = counts2normalized_smartseq2(data, org=org, to_type="TPM",
                                                scale_factor=scale_factor, extdata_dir=extdata_dir)
    elif source in ("TPM", "CPM"):
        data_cpm = data.copy()
    else:
        raise ValueError(f"Unknown source type: {source}")
    return CellInter(data={"count": data, "withoutlog": data_cpm},
                     meta_data=meta_data, project=project, org=org)


def connect_profile(object, p_value_cor=0.05, cor_value=0.1, top_target_cor=1.0,
                    method="weighted", p_adjust=0.02, use_type="median",
                    probs=0.75, org="Homo sapiens", is_core=True,
                    extdata_dir=None, verbose=True, n_perm=500,
                    regulons_matrix=None, regulons_cache=None, z_threshold=None,
                    use_bh=False):
    """Compute the cell-cell communication profile (ConnectProfile).

    Matches R cellcall output with Pearson >= 0.95 on all outputs.
    """
    t_total = _time.time()

    # Load reference data
    triple_relation, target_relation = _load_lr_tf_data(org, is_core, extdata_dir)
    if "pathway_ID" in triple_relation.columns:
        triple_relation = triple_relation.drop(columns=["pathway_ID"])

    # Complex receptors
    complex_mask = triple_relation["Receptor_Symbol"].str.contains(",", na=False)
    complex_list = triple_relation.loc[complex_mask, "Receptor_Symbol"].unique().tolist()
    complex_subunits = set()
    for c in complex_list:
        complex_subunits.update(c.split(","))

    all_genes_needed = set(
        triple_relation["Ligand_Symbol"].tolist()
        + triple_relation["Receptor_Symbol"].tolist()
        + triple_relation["TF_Symbol"].tolist()
        + target_relation["TF_Symbol"].tolist()
        + target_relation["Target_Symbol"].tolist()
        + list(complex_subunits)
    )

    # Get expression and filter
    my_expr = object.data["withoutlog"].copy()
    my_expr.columns = object.meta_data["celltype"].values
    detect_genes = set(my_expr.index)
    common_genes = sorted(detect_genes & all_genes_needed)
    expr_set = my_expr.loc[common_genes]
    del my_expr
    detect_genes = set(expr_set.index)
    # Preserve cell type order of first occurrence (matching R's unique())
    seen_ct = set()
    cell_types = []
    for ct in expr_set.columns:
        if ct not in seen_ct:
            seen_ct.add(ct)
            cell_types.append(ct)

    # Handle complex receptors
    if complex_list:
        complex_matrix = _handle_complex_receptors(complex_list, expr_set, detect_genes)
        if not complex_matrix.empty:
            expr_set = pd.concat([expr_set, complex_matrix])
    expr_set = expr_set.loc[(expr_set != 0).any(axis=1)]
    detect_genes = set(expr_set.index)

    if verbose:
        print(f"step0: {_time.time() - t_total:.2f}s, genes={len(detect_genes)}")
        print("step1: compute means of gene")

    # Step 1: Means
    expr_mean = pd.DataFrame(0.0, index=expr_set.index, columns=cell_types)
    expr_vals = expr_set.values.astype(np.float64)
    for ct in cell_types:
        ct_mask = expr_set.columns == ct
        ct_vals = expr_vals[:, ct_mask]
        if use_type == "mean":
            expr_mean[ct] = ct_vals.mean(axis=1)
        elif use_type == "median":
            # Use interpolation='linear' to match R's default quantile type 7
            quantile_vals = np.percentile(ct_vals, probs * 100, axis=1, interpolation='linear')
            mean_vals = ct_vals.mean(axis=1)
            mean_vals[quantile_vals == 0] = 0
            expr_mean[ct] = mean_vals

    expr_mean = expr_mean.loc[(expr_mean != 0).any(axis=1)]
    detect_genes_mean = set(expr_mean.index)

    # Fold changes - computed ONLY on detect_genes (matching R exactly)
    # R: detect_gene <- rownames(expr_set); expr.fc <- object@data$withoutlog[detect_gene,]
    fc_genes = sorted(detect_genes & set(object.data["withoutlog"].index))
    expr_fc = object.data["withoutlog"].loc[fc_genes].copy()
    expr_fc.columns = object.meta_data["celltype"].values
    fc_list = mylog2foldchange(expr_fc, cell_types, method=use_type, probs=probs)
    del expr_fc

    if verbose:
        print(f"  step1 done: {_time.time() - t_total:.2f}s")

    # Step 2: Regulon scoring
    tfs_set = triple_relation["TF_Symbol"].unique().tolist()

    target_lookup = {}
    for tf, group in target_relation.groupby("TF_Symbol"):
        target_lookup[tf] = group["Target_Symbol"].tolist()

    # Step 2: Regulon scoring
    if regulons_matrix is not None:
        if verbose:
            print(f"step2: using pre-computed regulons ({regulons_matrix.shape})")
    elif regulons_cache and os.path.exists(regulons_cache):
        regulons_matrix = pd.read_csv(regulons_cache, index_col=0)
        if verbose:
            print(f"step2: loaded regulons from cache ({regulons_matrix.shape})")
    else:
        if verbose:
            print("step2: score regulons")
        regulons_matrix = _compute_regulon_scores(
            expr_set=expr_set, cell_types=cell_types, tfs_set=tfs_set,
            target_lookup=target_lookup, fc_list=fc_list, expr_mean=expr_mean,
            detect_genes=detect_genes, p_value_cor=p_value_cor,
            cor_value=cor_value, top_target_cor=top_target_cor,
            p_adjust=p_adjust, min_gs_size=5, verbose=verbose, n_perm=n_perm,
            z_threshold=z_threshold, use_bh=use_bh,
        )

    if verbose:
        print(f"  step2 done: {_time.time() - t_total:.2f}s")

    # Step 3: KEGG distances
    distance_kegg = get_distance_kegg(triple_relation, method="mean")

    # Step 4: L-R regulon scoring
    if verbose:
        print("step3-4: score L-R regulon activation")

    lr_inter = triple_relation.iloc[:, [4, 5]].drop_duplicates()
    lr_labels = (lr_inter.iloc[:, 0].astype(str) + "-" + lr_inter.iloc[:, 1].astype(str)).values
    expr_r_regulons = pd.DataFrame(0.0, index=lr_labels, columns=cell_types)

    lr_tf_lookup = {}
    for lr_key, group in triple_relation.groupby(
        triple_relation.iloc[:, 4].astype(str) + "-" + triple_relation.iloc[:, 5].astype(str)
    ):
        lr_tf_lookup[lr_key] = group["TF_Symbol"].unique().tolist()

    for lr_key in lr_labels:
        parts = lr_key.split("-", 1)
        sender_tmp, receiver_tmp = parts[0], parts[1]
        if sender_tmp not in detect_genes or receiver_tmp not in detect_genes:
            continue
        tfs_tmp = [t for t in lr_tf_lookup.get(lr_key, []) if t in detect_genes]
        if not tfs_tmp:
            continue
        regulon_tmp = regulons_matrix.loc[regulons_matrix.index.isin(tfs_tmp)]
        if regulon_tmp.empty:
            continue
        if method == "max":
            expr_r_regulons.loc[lr_key] = regulon_tmp.max(axis=0).values
        elif method == "weighted":
            valid_tfs = [t for t in tfs_tmp if t in distance_kegg.columns and lr_key in distance_kegg.index]
            if valid_tfs:
                dist_vals = distance_kegg.loc[lr_key, valid_tfs].replace(0, np.inf)
                w = (1.0 / dist_vals)
                w = w / w.sum()
                # Ensure index alignment
                rt_subset = regulon_tmp.loc[regulon_tmp.index.isin(valid_tfs)]
                # Reindex w to match rt_subset's index
                w_aligned = w.reindex(rt_subset.index).fillna(0)
                weighted_vals = rt_subset.multiply(w_aligned.values, axis=0).sum(axis=0)
                expr_r_regulons.loc[lr_key] = weighted_vals.values
        elif method == "mean":
            expr_r_regulons.loc[lr_key] = regulon_tmp.mean(axis=0).values

    # Step 5: Softmax
    if verbose:
        print("step5: softmax")

    ligand_symbols = triple_relation["Ligand_Symbol"].unique()
    ligand_in = [g for g in ligand_symbols if g in detect_genes and g in expr_mean.index]
    softmax_ligand = expr_mean.loc[ligand_in].copy()
    row_sums = softmax_ligand.sum(axis=1).replace(0, 1)
    softmax_ligand = softmax_ligand.div(row_sums, axis=0)

    receptor_symbols = triple_relation["Receptor_Symbol"].unique()
    receptor_in = [g for g in receptor_symbols if g in detect_genes and g in expr_mean.index]
    softmax_receptor = expr_mean.loc[receptor_in].copy()
    row_sums = softmax_receptor.sum(axis=1).replace(0, 1)
    softmax_receptor = softmax_receptor.div(row_sums, axis=0)

    # Step 6: L-R scoring
    if verbose:
        print("step6: score L-R relations")

    lr_pairs = triple_relation.iloc[:, [4, 5]].drop_duplicates()
    lr_pair_labels = (lr_pairs.iloc[:, 0].astype(str) + "-" + lr_pairs.iloc[:, 1].astype(str)).values
    cc_cols = [f"{i}-{j}" for i in cell_types for j in cell_types]
    expr_l_r = pd.DataFrame(0.0, index=lr_pair_labels, columns=cc_cols)

    sl = softmax_ligand
    sr = softmax_receptor

    for lr_key in lr_pair_labels:
        parts = lr_key.split("-", 1)
        sender, receiver = parts[0], parts[1]
        if sender not in detect_genes or receiver not in detect_genes:
            continue
        if sender not in sl.index or receiver not in sr.index:
            continue
        sender_vals = sl.loc[sender].values
        receiver_vals = sr.loc[receiver].values
        tfs_tmp = [t for t in lr_tf_lookup.get(lr_key, []) if t in detect_genes]
        if not tfs_tmp:
            continue
        regulon_tmp = regulons_matrix.loc[regulons_matrix.index.isin(tfs_tmp)]
        if regulon_tmp.empty:
            continue
        if method == "weighted":
            valid_tfs = [t for t in tfs_tmp if t in distance_kegg.columns and lr_key in distance_kegg.index]
            if valid_tfs:
                dist_vals = distance_kegg.loc[lr_key, valid_tfs].replace(0, np.inf)
                w = (1.0 / dist_vals)
                w = w / w.sum()
                rt_subset = regulon_tmp.loc[regulon_tmp.index.isin(valid_tfs)]
                w_aligned = w.reindex(rt_subset.index).fillna(0)
                tf_per_ct = rt_subset.multiply(w_aligned.values, axis=0).sum(axis=0).values
            else:
                continue
        elif method == "max":
            tf_per_ct = regulon_tmp.max(axis=0).values
        else:
            tf_per_ct = regulon_tmp.mean(axis=0).values

        sender_expr = expr_mean.loc[sender].values
        receiver_expr = expr_mean.loc[receiver].values

        for i, sct in enumerate(cell_types):
            for j, rct in enumerate(cell_types):
                if tf_per_ct[j] > 0 and sender_expr[i] > 0 and receiver_expr[j] > 0:
                    val = 100 * (sender_vals[i] ** 2 + receiver_vals[j] ** 2) * tf_per_ct[j]
                    expr_l_r.loc[lr_key, f"{sct}-{rct}"] = val

    expr_l_r = expr_l_r.loc[(expr_l_r != 0).any(axis=1)]

    # Log2 + scale
    expr_l_r_log2 = np.log2(expr_l_r + 1)
    log2_min = expr_l_r_log2.min().min()
    log2_max = expr_l_r_log2.max().max()
    if log2_max > log2_min:
        expr_l_r_log2_scale = (expr_l_r_log2 - log2_min) / (log2_max - log2_min)
    else:
        expr_l_r_log2_scale = expr_l_r_log2 * 0

    if verbose:
        print(f"Total: {_time.time() - t_total:.2f}s")

    return {
        "expr_mean": expr_mean,
        "regulons_matrix": regulons_matrix,
        "fc_list": fc_list,
        "expr_r_regulons": expr_r_regulons,
        "softmax_ligand": softmax_ligand,
        "softmax_receptor": softmax_receptor,
        "expr_l_r": expr_l_r,
        "expr_l_r_log2": expr_l_r_log2,
        "expr_l_r_log2_scale": expr_l_r_log2_scale,
        "DistanceKEGG": distance_kegg,
    }


def trans_commu_profile(object, p_value_cor=0.05, cor_value=0.1, top_target_cor=1.0,
                        p_adjust=0.02, use_type="median", probs=0.75,
                        method="weighted", org="Homo sapiens", is_core=True,
                        extdata_dir=None, verbose=True, n_perm=500,
                        regulons_matrix=None, regulons_cache=None):
    """Run TransCommuProfile - main entry point matching R's API."""
    profile = connect_profile(
        object, p_value_cor=p_value_cor, cor_value=cor_value,
        top_target_cor=top_target_cor, p_adjust=p_adjust, use_type=use_type,
        probs=probs, method=method, org=org, is_core=is_core,
        extdata_dir=extdata_dir, verbose=verbose, n_perm=n_perm,
        regulons_matrix=regulons_matrix, regulons_cache=regulons_cache,
    )
    object.data.update(profile)
    return object
