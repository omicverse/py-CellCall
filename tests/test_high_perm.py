#!/usr/bin/env python
"""Test if higher n_perm improves Pearson for borderline TFs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import gc

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr
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

r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

def evaluate(py_rm, label):
    common_tfs = sorted(set(py_rm.index) & set(r_ref.index))
    common_cts = sorted(set(py_rm.columns) & set(r_ref.columns))
    py_aligned = py_rm.loc[common_tfs, common_cts]
    r_aligned = r_ref.loc[common_tfs, common_cts]
    r_flat = r_aligned.values.flatten()
    p_flat = py_aligned.values.flatten()
    pearson_full, _ = pearsonr(r_flat, p_flat)
    active_mask = (r_flat != 0) | (p_flat != 0)
    pearson_act, _ = pearsonr(r_flat[active_mask], p_flat[active_mask]) if active_mask.sum() > 1 else (0, 0)
    n_active = (py_rm > 0).sum().sum()
    r_active = set(zip(*np.where(r_aligned.values > 0)))
    p_active = set(zip(*np.where(py_rm.values > 0)))
    fp = len(p_active - r_active)
    fn = len(r_active - p_active)
    print(f"  {label}: full={pearson_full:.4f} active={pearson_act:.4f} n={n_active} FP={fp} FN={fn}")
    return pearson_full

# Baseline: n_perm=500
print("Running n_perm=500 (baseline)...")
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.02)
py_rm_500 = result['regulons_matrix']
evaluate(py_rm_500, "n_perm=500")

# Test n_perm=2000
print("\nRunning n_perm=2000...")
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=2000, p_adjust=0.02)
py_rm_2000 = result['regulons_matrix']
evaluate(py_rm_2000, "n_perm=2000")

# Test n_perm=500 with p_adjust=0.025 (slightly relaxed)
print("\nRunning n_perm=500, p_adjust=0.025...")
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.025)
py_rm_025 = result['regulons_matrix']
evaluate(py_rm_025, "p_adjust=0.025")

# Test n_perm=500 with p_adjust=0.015 (tighter)
print("\nRunning n_perm=500, p_adjust=0.015...")
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.015)
py_rm_015 = result['regulons_matrix']
evaluate(py_rm_015, "p_adjust=0.015")

print("\nDone!")
