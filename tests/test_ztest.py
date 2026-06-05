#!/usr/bin/env python
"""Test z-test p-value approach for reducing false positives."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings('ignore')

import rdata
import pandas as pd
import numpy as np
from scipy.stats import norm as normal_dist
from cellcall.connect_profile import _load_lr_tf_data, _handle_complex_receptors, _spearman_one_vs_many
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

def compute_stats(tf, ct):
    gene_set = get_gene_set(tf, ct)
    if not gene_set:
        return None
    fc_values, fc_names = get_fc(ct)
    gene_to_fc = {g: i for i, g in enumerate(fc_names)}
    hit_mask = np.zeros(len(fc_names), dtype=bool)
    for g in gene_set:
        if g in gene_to_fc:
            hit_mask[gene_to_fc[g]] = True

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
    null_std = np.std(es_null)
    if null_abs_mean < 1e-15:
        return None
    nes = es_obs / null_abs_mean

    perm_p = (np.abs(es_null) >= abs(es_obs)).mean()
    if perm_p == 0:
        perm_p = 1.0 / n_perm

    if null_std > 1e-15:
        z = abs(es_obs) / null_std
        z_p = 2 * (1 - normal_dist.cdf(z))
        if z_p == 0:
            z_p = 1e-300
    else:
        z_p = 1.0

    return {
        'es': es_obs, 'nes': nes, 'perm_p': perm_p, 'z_p': z_p,
        'null_mean': null_abs_mean, 'null_std': null_std,
        'gs': len(gene_set)
    }

# Test all known FPs and key TFs
test_tfs = [
    # Python FPs
    ('EGR1', 'SPGed'), ('EGR1', 'ST'), ('EGR1', 'SPGing'),
    ('NFKB2', 'ST'), ('PML', 'ST'), ('RXRA', 'ST'),
    ('EP300', 'ST'), ('HEY1', 'SSC'), ('IRF1', 'SPGed'),
    ('NFATC1', 'SPGed'), ('NFATC2', 'SPGed'), ('SMAD2', 'SSC'),
    # Key shared TFs (should remain)
    ('PBX1', 'ST'), ('SOX9', 'ST'), ('FOXO3', 'ST'),
    ('IKBKB', 'ST'), ('STAT2', 'ST'),
    # FN
    ('E2F4', 'SPGed'),
]

print(f"{'TF':<10} {'CT':<8} {'NES':>6} {'gs':>4} {'perm_p':>8} {'z_p':>10} {'z>0.05?':>8}")
print("-" * 60)
for tf, ct in test_tfs:
    s = compute_stats(tf, ct)
    if s is None:
        print(f"{tf:<10} {ct:<8} - no gene set")
        continue
    z_flag = "YES" if s['z_p'] >= 0.05 else "no"
    print(f"{tf:<10} {ct:<8} {s['nes']:6.3f} {s['gs']:4d} {s['perm_p']:8.4f} {s['z_p']:10.6f} {z_flag:>8}")
