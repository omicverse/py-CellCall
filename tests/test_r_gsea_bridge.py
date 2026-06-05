#!/usr/bin/env python
"""Bridge test: Python pipeline + R's clusterProfiler::GSEA for p-values."""
import os, sys, warnings, gc
os.environ['R_ENCODING'] = 'UTF-8'
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rpy2.robjects as ro
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from cellcall.connect_profile import (_load_lr_tf_data, _get_correlated_targets,
                                       mylog2foldchange)
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

ro.r('library(clusterProfiler)')

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

my_expr = obj.data["withoutlog"].copy()
my_expr.columns = obj.meta_data["celltype"].values
detect_genes = set(my_expr.index)
all_genes_needed = set(
    triple_relation["Ligand_Symbol"].tolist()
    + triple_relation["Receptor_Symbol"].tolist()
    + triple_relation["TF_Symbol"].tolist()
    + target_relation["TF_Symbol"].tolist()
    + target_relation["Target_Symbol"].tolist()
)
common_genes = sorted(detect_genes & all_genes_needed)
expr_set = my_expr.loc[common_genes]
detect_genes = set(expr_set.index)
cell_types = list(dict.fromkeys(expr_set.columns))

expr_mean = pd.DataFrame(0.0, index=expr_set.index, columns=cell_types)
expr_vals = expr_set.values.astype(np.float64)
for ct in cell_types:
    ct_mask = expr_set.columns == ct
    ct_vals = expr_vals[:, ct_mask]
    quantile_vals = np.percentile(ct_vals, 75, axis=1, interpolation='linear')
    mean_vals = ct_vals.mean(axis=1)
    mean_vals[quantile_vals == 0] = 0
    expr_mean[ct] = mean_vals
expr_mean = expr_mean.loc[(expr_mean != 0).any(axis=1)]

fc_genes = sorted(detect_genes & set(obj.data["withoutlog"].index))
expr_fc = obj.data["withoutlog"].loc[fc_genes].copy()
expr_fc.columns = obj.meta_data["celltype"].values
fc_list = mylog2foldchange(expr_fc, cell_types, method="median", probs=0.75)

tfs_set = list(set(triple_relation['TF_Symbol']))
target_lookup = {}
for _, row in target_relation.iterrows():
    tf, target = row['TF_Symbol'], row['Target_Symbol']
    target_lookup.setdefault(tf, []).append(target)

gene_to_row = {g: i for i, g in enumerate(expr_set.index)}
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

regulons_matrix = pd.DataFrame(0.0, index=tfs_set, columns=cell_types)

for ct in cell_types:
    print(f'Processing {ct}...')
    gene_sets = ct_gene_sets[ct]
    if not gene_sets:
        continue

    fc_df = fc_list[ct].set_index("gene_id")["log2fc"]
    fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
    fc_genes_in = [g for g in fc_df.index if g in detect_genes]
    fc_filtered = fc_df.loc[fc_genes_in]
    fc_sorted = fc_filtered.sort_values(ascending=False)
    gene_names_list = list(fc_sorted.index)
    fc_values_list = list(fc_sorted.values)

    # Set up R geneList
    fc_str = ",".join(str(v) for v in fc_values_list)
    names_str = ",".join('"' + g + '"' for g in gene_names_list)
    ro.r(f'gene_list <- c({fc_str})')
    ro.r(f'names(gene_list) <- c({names_str})')

    # Build TERM2GENE
    term_rows = []
    for tf, genes in gene_sets.items():
        for g in genes:
            term_rows.append((tf, g))
    term_names_list = [r[0] for r in term_rows]
    gene_names_term_list = [r[1] for r in term_rows]

    term_str = ",".join('"' + t + '"' for t in term_names_list)
    gene_term_str = ",".join('"' + g + '"' for g in gene_names_term_list)
    ro.r(f'term_names <- c({term_str})')
    ro.r(f'gene_names_t <- c({gene_term_str})')
    ro.r('term2gene <- data.frame(term=term_names, gene=gene_names_t)')

    # Run clusterProfiler::GSEA
    ro.r('''
    res <- tryCatch({
        GSEA(geneList = gene_list,
             TERM2GENE = term2gene,
             minGSSize = 5,
             exponent = 1,
             maxGSSize = 500,
             pvalueCutoff = 1,
             pAdjustMethod = "BH",
             verbose = FALSE)
    }, error = function(e) NULL)
    ''')

    has_result = ro.r('!is.null(res)')
    if not has_result[0]:
        print(f'  {ct}: GSEA failed')
        continue

    n_rows = int(ro.r('nrow(as.data.frame(res@result))')[0])
    if n_rows == 0:
        print(f'  {ct}: no results')
        continue

    count = 0
    for tf in gene_sets.keys():
        r_code = f'rr <- res@result[res@result$ID=="{tf}", c("NES","p.adjust")]; if(nrow(rr)>0) c(rr$NES[1], rr$p.adjust[1]) else c(NA, NA)'
        tf_result = ro.r(r_code)
        if len(tf_result) >= 2 and not np.isnan(tf_result[0]):
            nes = float(tf_result[0])
            padj = float(tf_result[1])
            expr_val = expr_mean.loc[tf, ct] if tf in expr_mean.index else 0
            if padj < 0.05 and nes > 0 and expr_val > 0:
                regulons_matrix.loc[tf, ct] = nes
                count += 1
    print(f'  {ct}: {count} active TFs')

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
print(f'\nR clusterProfiler bridge: Pearson={pearson_full:.4f} n={n_active} FP={fp} FN={fn}')

for ct in common_cts:
    r_tfs = set(r_aligned.index[r_aligned[ct] > 0])
    p_tfs = set(py_aligned.index[py_aligned[ct] > 0])
    fp_list = sorted(p_tfs - r_tfs)
    fn_list = sorted(r_tfs - p_tfs)
    agree = sorted(r_tfs & p_tfs)
    print(f'  {ct}: R={len(r_tfs)} Py={len(p_tfs)} agree={len(agree)} FP={fp_list} FN={fn_list}')
