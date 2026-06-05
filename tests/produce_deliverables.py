#!/usr/bin/env python
"""Produce all deliverables: CSV outputs + R visualization PNGs."""
import os, sys, warnings, gc, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr
from cellcall.connect_profile import connect_profile
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

OUT_DIR = 'tests/deliverables'
os.makedirs(OUT_DIR, exist_ok=True)

print("=== Loading data ===")
extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')
del parsed, data_df, data_cpm; gc.collect()

print("=== Running connect_profile ===")
t0 = time.time()
result = connect_profile(obj, org='Homo sapiens', verbose=True, n_perm=500)
t_total = time.time() - t0
print(f"Total time: {t_total:.1f}s")

# Export CSVs
print("\n=== Exporting CSVs ===")
for key in ['expr_mean', 'regulons_matrix', 'expr_r_regulons',
            'softmax_ligand', 'softmax_receptor', 'expr_l_r',
            'expr_l_r_log2', 'expr_l_r_log2_scale']:
    if key in result and hasattr(result[key], 'to_csv'):
        path = os.path.join(OUT_DIR, f'py_{key}.csv')
        result[key].to_csv(path)
        print(f"  {key}: {result[key].shape} -> {path}")

# Export FC per cell type
for ct, fc_df in result['fc_list'].items():
    path = os.path.join(OUT_DIR, f'py_fc_{ct}.csv')
    fc_df.to_csv(path, index=False)
    print(f"  fc_{ct}: {fc_df.shape} -> {path}")

# Compare with R references
print("\n=== Parity Check ===")
r_ref_dir = 'tests/reference'
for name, r_file, py_key in [
    ('expr_mean', 'r_expr_mean.csv', 'expr_mean'),
    ('regulons_matrix', 'r_regulons_matrix.csv', 'regulons_matrix'),
    ('expr_l_r_log2_scale', 'r_expr_l_r_log2_scale.csv', 'expr_l_r_log2_scale'),
]:
    r_path = os.path.join(r_ref_dir, r_file)
    if os.path.exists(r_path):
        r_ref = pd.read_csv(r_path, index_col=0)
        py_data = result[py_key]
        common_idx = sorted(set(py_data.index) & set(r_ref.index))
        common_cols = sorted(set(py_data.columns) & set(r_ref.columns))
        if common_idx and common_cols:
            r_vals = r_ref.loc[common_idx, common_cols].values.flatten()
            p_vals = py_data.loc[common_idx, common_cols].values.flatten()
            pearson, _ = pearsonr(r_vals, p_vals)
            status = "PASS" if pearson >= 0.95 else "NEAR" if pearson >= 0.90 else "FAIL"
            print(f"  {name:25s}: Pearson={pearson:.4f} [{status}]")
        else:
            print(f"  {name:25s}: no common indices")
    else:
        print(f"  {name:25s}: R reference not found")

# Timing summary
print(f"\n=== Summary ===")
print(f"Python time: {t_total:.1f}s")
print(f"R reference: 1767.9s")
print(f"Speedup: {1767.9/t_total:.1f}x")
print(f"Outputs in: {OUT_DIR}/")
print("Done!")
