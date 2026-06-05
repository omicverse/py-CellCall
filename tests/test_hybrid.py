#!/usr/bin/env python
"""Test hybrid approach: permutation p-value + z-test for borderline TFs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr, norm as normal_dist
from cellcall.connect_profile import connect_profile
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')

# Run pipeline to get detailed results
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.02)
py_rm = result['regulons_matrix']

# Load R reference
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

# Get TF-level details from pipeline
# We need to re-run with detailed output
from cellcall.connect_profile import _load_lr_tf_data, _handle_complex_receptors, _get_correlated_targets, _spearman_one_vs_many
from cellcall.fgsea import _batch_es_fast, run_gsea_per_tf
from cellcall.foldchange import mylog2foldchange

my_expr = obj.data['withoutlog'].copy()
my_expr.columns = obj.meta_data['celltype'].values
triple_relation, target_relation = _load_lr_tf_data('Homo sapiens', True, extdata_dir)
triple_relation = triple_relation.drop(columns=['pathway_ID'])
complex_mask = triple_relation['Receptor_Symbol'].str.contains(',', na=False)
complex_list = triple_relation.loc[complex_mask, 'Receptor_Symbol'].unique().tolist()
complex_subunits = set()
for c in complex_list:
    complex_subunits.update(c.split(','))
all_genes_needed = set(triple_relation['Ligand_Symbol'].tolist() + triple_relation['Receptor_Symbol'].tolist() + triple_relation['TF_Symbol'].tolist() + target_relation['TF_Symbol'].tolist() + target_relation['Target_Symbol'].tolist() + list(complex_subunits))
common_genes = sorted(set(my_expr.index) & all_genes_needed)
expr_set = my_expr.loc[common_genes]
detect_genes = set(expr_set.index)
complex_matrix = _handle_complex_receptors(complex_list, expr_set, detect_genes)
if not complex_matrix.empty:
    expr_set = pd.concat([expr_set, complex_matrix])
expr_set = expr_set.loc[(expr_set != 0).any(axis=1)]
detect_genes = set(expr_set.index)

# Compute FC like pipeline
fc_genes = sorted(detect_genes & set(obj.data['withoutlog'].index))
expr_fc = obj.data['withoutlog'].loc[fc_genes].copy()
expr_fc.columns = obj.meta_data['celltype'].values
cell_types = list(dict.fromkeys(obj.meta_data['celltype'].values))
fc_list = mylog2foldchange(expr_fc, cell_types, method='median', probs=0.75)
del expr_fc

# Compute expr_mean like pipeline
expr_mean = pd.DataFrame(0.0, index=expr_set.index, columns=cell_types)
expr_vals = expr_set.values.astype(np.float64)
for ct in cell_types:
    ct_mask = expr_set.columns == ct
    ct_vals = expr_vals[:, ct_mask]
    quantile_vals = np.percentile(ct_vals, 75, axis=1, interpolation='linear')
    mean_vals = ct_vals.mean(axis=1)
    mean_vals[quantile_vals == 0] = 0
    expr_mean[ct] = mean_vals

target_lookup = {}
for tf, group in target_relation.groupby('TF_Symbol'):
    target_lookup[tf] = group['Target_Symbol'].tolist()

tfs_set = triple_relation['TF_Symbol'].unique().tolist()
tfs_in_data = [t for t in tfs_set if t in detect_genes]
gene_to_row = {g: i for i, g in enumerate(expr_set.index)}

# Build gene sets and compute detailed stats for key TFs
test_tfs = ['EGR1', 'NFKB2', 'PML', 'RXRA', 'E2F4', 'IRF7', 'PBX1', 'SOX9', 'FOXO3', 'STAT2', 'ELK1', 'JUND', 'MYC', 'SP1', 'TP53', 'RB1', 'TCF7L2']

print(f"{'TF':<10} {'CT':<8} {'NES':>6} {'gs':>4} {'perm_p':>8} {'z_p':>10} {'z_flag':>7} {'in_R':>5} {'in_Py':>5}")
print("-" * 75)

for tf in test_tfs:
    if tf not in tfs_in_data:
        continue
    for ct in cell_types:
        # Check if in expr_mean
        if tf not in expr_mean.index or expr_mean.loc[tf, ct] == 0:
            continue

        # Build gene set
        ct_mask = np.array(expr_set.columns == ct)
        ct_expr_vals = expr_set.values.astype(np.float32)[:, ct_mask]
        tf_expr = ct_expr_vals[gene_to_row[tf]]
        targets = target_lookup.get(tf, [])
        targets_in = [g for g in targets if g in detect_genes]
        target_indices = [gene_to_row[g] for g in targets_in if g in gene_to_row]
        target_exprs = ct_expr_vals[target_indices]
        target_names = [g for g in targets_in if g in gene_to_row]

        if not target_names:
            continue

        corrs, pvals = _spearman_one_vs_many(tf_expr, target_exprs)
        mask = (pvals < 0.05) & (corrs > 0.1)
        filtered_names = [target_names[i] for i in range(len(target_names)) if mask[i]]
        filtered_corrs = corrs[mask]

        if len(filtered_names) <= 1:
            continue

        order = np.argsort(-filtered_corrs)
        sorted_names = [filtered_names[i] for i in order]
        n_top = int(np.floor(1.0 * len(sorted_names)))
        if len(sorted_names) > n_top + 1:
            sorted_names = sorted_names[1: n_top + 1]

        gene_set = sorted_names
        gs = len(gene_set)
        if gs < 5:
            continue

        # Compute NES
        fc_df = fc_list[ct].set_index('gene_id')['log2fc']
        fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
        fc_genes_in = [g for g in fc_df.index if g in detect_genes]
        fc_filtered = fc_df.loc[fc_genes_in]
        noise = np.arange(len(fc_filtered), dtype=np.float64) * 1e-10
        fc_series = pd.Series(fc_filtered.values + noise, index=fc_filtered.index)
        fc_sorted = fc_series.sort_values(ascending=False)
        fc_values = fc_filtered.loc[fc_sorted.index].values.astype(np.float64)
        fc_names = fc_sorted.index.values

        gene_to_fc = {g: i for i, g in enumerate(fc_names)}
        hit_mask = np.zeros(len(fc_names), dtype=bool)
        for g in gene_set:
            if g in gene_to_fc:
                hit_mask[gene_to_fc[g]] = True

        n = len(fc_values)
        rng = np.random.RandomState(42)
        n_perm = 500
        rand_mat = rng.random((n_perm, n))
        perm_idx = np.argpartition(rand_mat, gs, axis=1)[:, :gs]
        perm_masks = np.zeros((n_perm, n), dtype=bool)
        rows = np.arange(n_perm)[:, None]
        perm_masks[rows, perm_idx] = True

        es_obs, es_null = _batch_es_fast(fc_values, hit_mask, perm_masks)
        null_abs_mean = np.abs(es_null).mean()
        null_std = np.std(es_null)
        if null_abs_mean < 1e-15:
            continue
        nes = es_obs / null_abs_mean

        perm_p = (np.abs(es_null) >= abs(es_obs)).mean()
        if perm_p == 0:
            perm_p = 1.0 / n_perm

        # Z-test
        if null_std > 1e-15:
            z = abs(es_obs) / null_std
            z_p = 2 * (1 - normal_dist.cdf(z))
        else:
            z_p = 1.0

        # Check R reference
        in_r = r_ref.loc[tf, ct] > 0 if tf in r_ref.index and ct in r_ref.columns else False
        in_py = py_rm.loc[tf, ct] > 0 if tf in py_rm.index and ct in py_rm.columns else False

        z_flag = "YES" if z_p >= 0.05 else "no"
        print(f"{tf:<10} {ct:<8} {nes:6.3f} {gs:4d} {perm_p:8.4f} {z_p:10.6f} {z_flag:>7} {'YES' if in_r else 'no':>5} {'YES' if in_py else 'no':>5}")
