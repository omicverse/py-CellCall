#!/usr/bin/env python
"""Diagnostic: check what multilevel does to the FP/FN p-values."""
import sys, os, warnings, gc, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from cellcall.fgsea import (_calc_es_single, _build_multilevel,
                            _get_pvalue_from_levels, _beta_mean_log,
                            _batch_es_fast)
from cellcall.connect_profile import connect_profile, _compute_fc_arrays
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
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)
del parsed, data_df, data_cpm; gc.collect()

# Get FC values and gene sets for SPGed cell type (has both FP and FN)
fc_arrays = _compute_fc_arrays(obj)
fc_values, fc_gene_names = fc_arrays['SPGed']

# Get gene sets (need to reconstruct from the pipeline)
# Manually set up for EGR1 (FP) and E2F4 (FN)
from cellcall.connect_profile import _load_lr_tf_data, _build_gene_sets_per_ct

# Load reference data
triple_relation, target_relation = _load_lr_tf_data('Homo sapiens', True, None)
if "pathway_ID" in triple_relation.columns:
    triple_relation = triple_relation.drop(columns=["pathway_ID"])

# Get the correlation-filtered gene sets
gene_to_idx = {g: i for i, g in enumerate(fc_gene_names)}
n = len(fc_gene_names)
values = fc_values.astype(np.float64)

# For EGR1 (FP) - need to compute its gene set
from cellcall.connect_profile import _get_correlated_targets
import scipy.stats as sp_stats

# Compute TF-target correlations for SPGed
ct = 'SPGed'
ct_mask = obj.meta_data['celltype'] == ct
ct_expr = obj.data['withoutlog'].values[:, ct_mask.values]
tfs_set = list(obj.tfs) if hasattr(obj, 'tfs') else []
if not tfs_set:
    tfs_set = obj.meta_data.columns.tolist()[:10]  # fallback

# Use the actual regulons data to get gene sets
# For now, let's just test with known gene sets
# EGR1 targets (from the R regulons)
# E2F4 targets (from the R regulons)

# Let's manually construct small gene sets and test multilevel
print("=== Manual multilevel test for EGR1 (FP, SPGed) ===")

# Simulate a gene set of size 50 (typical)
rng = np.random.RandomState(42)
k = 50

# Random subset (mimicking a gene set)
random_subset = np.sort(rng.choice(n, k, replace=False))
hit_mask = np.zeros(n, dtype=bool)
hit_mask[random_subset] = True
es_random = _calc_es_single(values, hit_mask)
print(f"Random subset ES: {es_random:.4f}")

# Test: does multilevel change the p-value vs fixed permutations?
n_perm = 500
abs_cumsum = np.cumsum(np.abs(values))

# Fixed permutation p-value
es_null = np.zeros(n_perm)
for i in range(n_perm):
    perm = np.sort(rng.choice(n, k, replace=False))
    pm = np.zeros(n, dtype=bool)
    pm[perm] = True
    es_null[i] = _calc_es_single(values, pm)

fixed_pval = (np.abs(es_null) >= abs(es_random)).mean()
if fixed_pval == 0:
    fixed_pval = 1.0 / n_perm
print(f"Fixed permutation p-value (n={n_perm}): {fixed_pval:.6f}")

# Multilevel p-value
print("Building multilevel (sample_size=31)...")
t0 = time.time()
ml_levels = _build_multilevel(values, k, 31, abs(es_random) * 0.5, rng,
                               abs_cumsum=abs_cumsum, max_levels=10)
t1 = time.time()
print(f"Built {len(ml_levels)} levels in {t1-t0:.1f}s")

for i, lev in enumerate(ml_levels):
    print(f"  Level {i}: bound={lev['bound']:.4f} high={len(lev['high_scores'])} low={len(lev['low_scores'])}")

ml_pval = _get_pvalue_from_levels(abs(es_random), ml_levels, 31)
print(f"Multilevel p-value: {ml_pval:.6f}")

# Compare
print(f"\nDifference: {abs(fixed_pval - ml_pval):.6f}")
print(f"Fixed: {fixed_pval:.6f}, Multilevel: {ml_pval:.6f}")

print("\nDone!")
