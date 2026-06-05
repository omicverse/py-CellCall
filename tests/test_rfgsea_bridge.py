#!/usr/bin/env python
"""Test using R's fgseaMultilevel via rpy2 as a bridge."""
import os, sys, warnings, gc
os.environ['R_ENCODING'] = 'UTF-8'
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rpy2.robjects as ro
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from cellcall.connect_profile import _load_lr_tf_data, _get_correlated_targets, mylog2foldchange
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

ro.r('library(fgsea)')

extdata_dir = os.path.join('cellcall', 'extdata')
import rdata
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)
triple_relation, target_relation = _load_lr_tf_data('Homo sapiens', True, extdata_dir)
del parsed, data_df, data_cpm; gc.collect()

cell_types = list(obj.meta_data['celltype'].unique())
detect_genes = list(obj.data['withoutlog'].index)
expr_set = obj.data['withoutlog']
gene_to_row = {g: i for i, g in enumerate(expr_set.index)}

expr_mean_data = {}
for ct in cell_types:
    ct_cols = expr_set.columns[obj.meta_data['celltype'] == ct]
    ct_vals = expr_set[ct_cols].values
    q_vals = np.percentile(ct_vals, 75, axis=1)
    mean_vals = ct_vals.mean(axis=1)
    mean_vals[q_vals == 0] = 0
    expr_mean_data[ct] = mean_vals
expr_mean = pd.DataFrame(expr_mean_data, index=expr_set.index)

fc_list = mylog2foldchange(obj.data['withoutlog'], cell_types, method='median', probs=0.75)

tfs_set = list(set(triple_relation['TF_Symbol']))
target_lookup = {}
for _, row in target_relation.iterrows():
    tf = row['TF_Symbol']
    target = row['Target_Symbol']
    if tf not in target_lookup:
        target_lookup[tf] = []
    target_lookup[tf].append(target)

fc_arrays = {}
for ct in cell_types:
    fc_df = fc_list[ct].set_index("gene_id")["log2fc"]
    fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
    fc_genes = [g for g in fc_df.index if g in detect_genes]
    fc_filtered = fc_df.loc[fc_genes]
    noise = np.arange(len(fc_filtered), dtype=np.float64) * 1e-10
    fc_with_noise = fc_filtered.values + noise
    fc_series = pd.Series(fc_with_noise, index=fc_filtered.index)
    fc_sorted = fc_series.sort_values(ascending=False)
    fc_arrays[ct] = (fc_filtered.loc[fc_sorted.index].values, fc_sorted.index.values)

# Build gene sets for each CT
ct_gene_sets = {}
for ct in cell_types:
    gene_sets = {}
    ct_mask = np.array(expr_set.columns == ct)
    ct_expr_vals = expr_set.values[:, ct_mask]
    for tf in tfs_set:
        if tf not in gene_to_row:
            continue
        targets = target_lookup.get(tf, [])
        targets_in = [t for t in targets if t in gene_to_row]
        if not targets_in:
            continue
        tf_expr = ct_expr_vals[gene_to_row[tf]]
        target_indices = [gene_to_row[g] for g in targets_in]
        target_exprs = ct_expr_vals[target_indices]
        correlated = _get_correlated_targets(tf_expr, target_exprs, targets_in, 0.05, 0.1, 1.0)
        if len(correlated) >= 5:
            gene_sets[tf] = correlated
    ct_gene_sets[ct] = gene_sets

# Run R fgseaMultilevel for each CT
regulons_matrix = pd.DataFrame(0.0, index=tfs_set, columns=cell_types)

for ct in cell_types:
    print(f'Processing {ct}...')
    fc_values, fc_gene_names = fc_arrays[ct]
    gene_sets = ct_gene_sets[ct]
    if not gene_sets:
        continue

    # Build R stats vector with names
    stats_str = ",".join(str(v) for v in fc_values)
    names_str = ",".join('"' + g + '"' for g in fc_gene_names)
    ro.r(f'stats_vals <- c({stats_str})')
    ro.r(f'names(stats_vals) <- c({names_str})')

    # Build pathway list in R
    pathway_parts = []
    for tf, genes in gene_sets.items():
        genes_str = ",".join('"' + g + '"' for g in genes)
        pathway_parts.append(f'{tf}=c({genes_str})')
    ro.r(f'pathways_list <- list({",".join(pathway_parts)})')

    # Run fgseaMultilevel
    ro.r('''
    res <- fgseaMultilevel(
        pathways=pathways_list,
        stats=stats_vals,
        sampleSize=101,
        minSize=5,
        maxSize=600,
        nPermSimple=0
    )
    ''')

    # Extract results
    res_df = ro.r('as.data.frame(res)')
    for tf in gene_sets.keys():
        nes_vals = ro.r(f'res$NES[res$pathway=="{tf}"]')
        pval_vals = ro.r(f'res$pval[res$pathway=="{tf}"]')
        padj_vals = ro.r(f'res$padj[res$pathway=="{tf}"]')
        if len(nes_vals) > 0:
            nes = float(nes_vals[0])
            padj = float(padj_vals[0])
            expr_val = expr_mean.loc[tf, ct] if tf in expr_mean.index else 0
            if padj < 0.02 and nes > 0 and expr_val > 0:
                regulons_matrix.loc[tf, ct] = nes

# Evaluate
common_tfs = sorted(set(regulons_matrix.index) & set(r_ref.index))
common_cts = sorted(set(regulons_matrix.columns) & set(r_ref.columns))
py_aligned = regulons_matrix.loc[common_tfs, common_cts]
r_aligned = r_ref.loc[common_tfs, common_cts]
pearson_full, _ = pearsonr(r_aligned.values.flatten(), py_aligned.values.flatten())

r_active = set(); p_active = set()
for i, tf in enumerate(common_tfs):
    for j, ct in enumerate(common_cts):
        if r_aligned.iloc[i, j] > 0: r_active.add((tf, ct))
        if py_aligned.iloc[i, j] > 0: p_active.add((tf, ct))
fp = len(p_active - r_active)
fn = len(r_active - p_active)
n_active = (regulons_matrix > 0).sum().sum()
print(f'\nR fgseaMultilevel bridge: Pearson={pearson_full:.4f} n={n_active} FP={fp} FN={fn}')

for ct in common_cts:
    r_tfs = set(r_aligned.index[r_aligned[ct] > 0])
    p_tfs = set(py_aligned.index[py_aligned[ct] > 0])
    fp_list = sorted(p_tfs - r_tfs)
    fn_list = sorted(r_tfs - p_tfs)
    if fp_list or fn_list:
        print(f'  {ct}: FP={fp_list} FN={fn_list}')
    else:
        print(f'  {ct}: perfect match ({len(r_tfs)} TFs)')
