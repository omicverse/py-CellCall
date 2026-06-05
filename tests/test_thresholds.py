#!/usr/bin/env python
"""Test different p-value thresholds and NES filters to maximize Pearson."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings('ignore')

import rdata
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from cellcall.connect_profile import _load_lr_tf_data, _handle_complex_receptors, _spearman_one_vs_many
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x
from cellcall.foldchange import mylog2foldchange
from cellcall.fgsea import run_gsea_per_tf

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')

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

target_lookup = {}
for tf, group in target_relation.groupby('TF_Symbol'):
    target_lookup[tf] = group['Target_Symbol'].tolist()

cell_types = list(dict.fromkeys(obj.meta_data['celltype'].values))
fc_list = mylog2foldchange(expr_set, cell_types, method='median', probs=0.75)
gene_to_row = {g: i for i, g in enumerate(expr_set.index)}

# Load R reference
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

def get_gene_set(tf, ct):
    ct_mask = np.array(expr_set.columns == ct)
    ct_expr_vals = expr_set.values.astype(np.float32)[:, ct_mask]
    tf_expr = ct_expr_vals[gene_to_row[tf]]
    targets = target_lookup.get(tf, [])
    targets_in = [g for g in targets if g in detect_genes]
    target_indices = [gene_to_row[g] for g in targets_in if g in gene_to_row]
    target_exprs = ct_expr_vals[target_indices]
    target_names = [g for g in targets_in if g in gene_to_row]
    corrs, pvals = _spearman_one_vs_many(tf_expr, target_exprs)
    mask = (pvals < 0.05) & (corrs > 0.1)
    filtered_names = [target_names[i] for i in range(len(target_names)) if mask[i]]
    filtered_corrs = corrs[mask]
    if len(filtered_names) <= 1:
        return []
    order = np.argsort(-filtered_corrs)
    sorted_names = [filtered_names[i] for i in order]
    n_top = int(np.floor(1.0 * len(sorted_names)))
    if len(sorted_names) > n_top + 1:
        sorted_names = sorted_names[1: n_top + 1]
    return sorted_names

def get_fc(ct):
    fc_df = fc_list[ct].set_index('gene_id')['log2fc']
    fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
    fc_genes_in = [g for g in fc_df.index if g in detect_genes]
    fc_filtered = fc_df.loc[fc_genes_in]
    noise = np.arange(len(fc_filtered), dtype=np.float64) * 1e-10
    fc_series = pd.Series(fc_filtered.values + noise, index=fc_filtered.index)
    fc_sorted = fc_series.sort_values(ascending=False)
    return fc_filtered.loc[fc_sorted.index].values.astype(np.float64), fc_sorted.index.values

def compute_nes(tf, ct):
    gene_set = get_gene_set(tf, ct)
    if not gene_set:
        return None
    fc_values, fc_names = get_fc(ct)
    gene_to_fc = {g: i for i, g in enumerate(fc_names)}
    hit_mask = np.zeros(len(fc_names), dtype=bool)
    for g in gene_set:
        if g in gene_to_fc:
            hit_mask[gene_to_fc[g]] = True
    from cellcall.fgsea import _batch_es_fast
    n = len(fc_values)
    gs = int(hit_mask.sum())
    rng = np.random.RandomState(42)
    n_perm = 500
    rand_mat = rng.random((n_perm, n))
    perm_idx = np.argpartition(rand_mat, gs, axis=1)[:, :gs]
    perm_masks = np.zeros((n_perm, n), dtype=bool)
    rows = np.arange(n_perm)[:, None]
    perm_masks[rows, perm_idx] = True
    es_obs, es_null = _batch_es_fast(fc_values, hit_mask, perm_masks)
    null_abs_mean = np.abs(es_null).mean()
    if null_abs_mean < 1e-15:
        return None
    nes = es_obs / null_abs_mean
    perm_p = (np.abs(es_null) >= abs(es_obs)).mean()
    if perm_p == 0:
        perm_p = 1.0 / n_perm
    return {'nes': nes, 'perm_p': perm_p, 'es': es_obs, 'gs': len(gene_set)}

# Compute all NES values
all_tfs = sorted(set(target_lookup.keys()) & detect_genes)
results = {}
for tf in all_tfs:
    for ct in cell_types:
        r = compute_nes(tf, ct)
        if r is not None:
            results[(tf, ct)] = r

# Build Python regulons_matrix for different thresholds
def build_matrix(p_thresh, nes_min=0):
    py_mat = pd.DataFrame(0, index=all_tfs, columns=cell_types)
    for (tf, ct), r in results.items():
        if r['perm_p'] < p_thresh and r['nes'] > nes_min:
            py_mat.loc[tf, ct] = r['nes']
    return py_mat

# Test different thresholds
print(f"{'p_thresh':>10} {'nes_min':>8} {'n_active':>8} {'Pearson':>10} {'Spearman':>10} {'n_FP':>6} {'n_FN':>6}")
print("-" * 70)

best_pearson = 0
best_params = None

for p_thresh in [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002]:
    for nes_min in [0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        py_mat = build_matrix(p_thresh, nes_min)

        # Flatten for comparison
        r_flat = r_ref.reindex(index=all_tfs, columns=cell_types).fillna(0).values.flatten()
        p_flat = py_mat.values.flatten()

        # Only compare where at least one is non-zero
        mask = (r_flat != 0) | (p_flat != 0)
        if mask.sum() < 5:
            continue

        # Pearson on active entries
        if np.std(r_flat[mask]) > 0 and np.std(p_flat[mask]) > 0:
            pearson, _ = pearsonr(r_flat[mask], p_flat[mask])
        else:
            pearson = 0

        # Count active TFs
        n_active = (py_mat > 0).sum().sum()

        # Count FPs and FNs (TFs active in one but not the other)
        r_active = set()
        p_active = set()
        for tf in all_tfs:
            for ct in cell_types:
                if r_ref.loc[tf, ct] > 0 if tf in r_ref.index and ct in r_ref.columns else False:
                    r_active.add((tf, ct))
                if py_mat.loc[tf, ct] > 0:
                    p_active.add((tf, ct))
        fp = len(p_active - r_active)
        fn = len(r_active - p_active)

        if pearson > best_pearson:
            best_pearson = pearson
            best_params = (p_thresh, nes_min)

        if p_thresh in [0.02, 0.01, 0.005] or nes_min >= 1.2:
            print(f"{p_thresh:>10.3f} {nes_min:>8.1f} {n_active:>8d} {pearson:>10.4f} {0:>10.4f} {fp:>6d} {fn:>6d}")

print(f"\nBest: p_thresh={best_params[0]}, nes_min={best_params[1]}, Pearson={best_pearson:.4f}")
