#!/usr/bin/env python
"""Test z-test filter as post-processing on pipeline output."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr, norm as normal_dist
from cellcall.connect_profile import (
    _load_lr_tf_data, _handle_complex_receptors, _get_correlated_targets,
    _spearman_one_vs_many
)
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x
from cellcall.foldchange import mylog2foldchange
from cellcall.fgsea import _batch_es_fast

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')

r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

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

# Compute FC like pipeline (on original data, not expr_set)
fc_genes = sorted(detect_genes & set(obj.data['withoutlog'].index))
expr_fc = obj.data['withoutlog'].loc[fc_genes].copy()
expr_fc.columns = obj.meta_data['celltype'].values
cell_types = list(dict.fromkeys(obj.meta_data['celltype'].values))
fc_list = mylog2foldchange(expr_fc, cell_types, method='median', probs=0.75)
del expr_fc

# Compute expr_mean
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

# Compute all TF stats (NES + z_p) for all cell types
print("Computing NES + z_p for all TFs...")
all_stats = {}  # (tf, ct) -> {nes, perm_p, z_p, gs}

for ct_idx, ct in enumerate(cell_types):
    print(f"  {ct} ({ct_idx+1}/{len(cell_types)})...")
    ct_mask = np.array(expr_set.columns == ct)
    ct_expr_vals = expr_set.values.astype(np.float32)[:, ct_mask]

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

    n = len(fc_values)
    rng = np.random.RandomState(42)
    n_perm = 500

    for tf in tfs_in_data:
        if tf not in expr_mean.index or expr_mean.loc[tf, ct] == 0:
            continue
        if tf not in gene_to_row:
            continue

        targets = target_lookup.get(tf, [])
        targets_in = [g for g in targets if g in detect_genes and g in gene_to_row]
        if not targets_in:
            continue

        target_indices = [gene_to_row[g] for g in targets_in]
        target_exprs = ct_expr_vals[target_indices]
        target_names = targets_in

        corrs, pvals = _spearman_one_vs_many(ct_expr_vals[gene_to_row[tf]], target_exprs)
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

        hit_mask = np.zeros(n, dtype=bool)
        for g in gene_set:
            if g in gene_to_fc:
                hit_mask[gene_to_fc[g]] = True

        actual_gs = int(hit_mask.sum())
        if actual_gs < 5:
            continue

        rand_mat = rng.random((n_perm, n))
        perm_idx = np.argpartition(rand_mat, actual_gs, axis=1)[:, :actual_gs]
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

        all_stats[(tf, ct)] = {'nes': nes, 'perm_p': perm_p, 'z_p': z_p, 'gs': actual_gs}

print(f"Total stats: {len(all_stats)}")

# Build matrices for different filter strategies
def build_matrix(strategy, param):
    py_mat = pd.DataFrame(0, index=tfs_set, columns=cell_types)
    for (tf, ct), s in all_stats.items():
        keep = False
        if strategy == 'perm_only':
            keep = s['perm_p'] < param and s['nes'] > 0
        elif strategy == 'ztest':
            keep = s['perm_p'] < 0.02 and s['nes'] > 0 and s['z_p'] < param
        elif strategy == 'perm_or_strong_nes':
            keep = s['nes'] > 0 and (s['perm_p'] < param or s['nes'] > 1.5)
        elif strategy == 'perm_and_z_or_nes':
            # perm_p < threshold AND (z_p < z_thresh OR nes > nes_thresh)
            keep = s['perm_p'] < param and s['nes'] > 0 and (s['z_p'] < 0.06 or s['nes'] > 1.5)
        if keep:
            py_mat.loc[tf, ct] = s['nes']
    return py_mat

print(f"\n{'Strategy':<25} {'Param':>6} {'Pearson':>8} {'n_act':>6} {'n_FP':>5} {'n_FN':>5}")
print("-" * 65)

best_pearson = 0
best_desc = ""

for p_thresh in [0.02, 0.03, 0.05]:
    for strat, param in [
        ('perm_only', p_thresh),
        ('ztest', 0.05), ('ztest', 0.06), ('ztest', 0.08), ('ztest', 0.10),
        ('perm_or_strong_nes', p_thresh),
        ('perm_and_z_or_nes', p_thresh),
    ]:
        if strat == 'ztest' and param != 0.05 and param != 0.06 and param != 0.08 and param != 0.10:
            continue
        py_mat = build_matrix(strat, param)
        r_aligned = r_ref.reindex(index=tfs_set, columns=cell_types).fillna(0)
        r_flat = r_aligned.values.flatten()
        p_flat = py_mat.values.flatten()
        if np.std(r_flat) > 0 and np.std(p_flat) > 0:
            pearson, _ = pearsonr(r_flat, p_flat)
        else:
            pearson = 0
        n_active = (py_mat > 0).sum().sum()
        r_active = set(zip(*np.where(r_aligned.values > 0)))
        p_active = set(zip(*np.where(py_mat.values > 0)))
        fp = len(p_active - r_active)
        fn = len(r_active - p_active)
        desc = f"{strat} p={param}"
        print(f"{desc:<25} {param:>6.3f} {pearson:>8.4f} {n_active:>6d} {fp:>5d} {fn:>5d}")
        if pearson > best_pearson:
            best_pearson = pearson
            best_desc = desc

print(f"\nBest: {best_desc} -> Pearson={best_pearson:.4f}")
