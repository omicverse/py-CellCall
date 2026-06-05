#!/usr/bin/env python
"""Detailed analysis of false positives vs true positives NES values."""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
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
del parsed, data_df, data_cpm; gc.collect()

# Run Python pipeline
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.02)
py_rm = result['regulons_matrix']

# Align
common_tfs = sorted(set(py_rm.index) & set(r_ref.index))
common_cts = sorted(set(py_rm.columns) & set(r_ref.columns))
py_aligned = py_rm.loc[common_tfs, common_cts]
r_aligned = r_ref.loc[common_tfs, common_cts]

# Identify all TF/CT pairs and classify them
print("=== All TF/CT pairs with NES values ===")
print(f"{'TF':>12} {'CT':>8} {'R_NES':>8} {'Py_NES':>8} {'R>0':>5} {'Py>0':>5} {'Status':>8}")

# FPs (Python active but R not)
fp_list = []
fn_list = []
tp_list = []

for i, tf in enumerate(common_tfs):
    for j, ct in enumerate(common_cts):
        r_val = r_aligned.iloc[i, j]
        py_val = py_aligned.iloc[i, j]
        r_active = r_val > 0
        py_active = py_val > 0
        if py_active and not r_active:
            status = 'FP'
            fp_list.append((tf, ct, r_val, py_val))
        elif r_active and not py_active:
            status = 'FN'
            fn_list.append((tf, ct, r_val, py_val))
        elif r_active and py_active:
            status = 'TP'
            tp_list.append((tf, ct, r_val, py_val))
        else:
            continue
        print(f"{tf:>12} {ct:>8} {r_val:>8.4f} {py_val:>8.4f} {str(r_active):>5} {str(py_active):>5} {status:>8}")

print(f"\n=== Summary ===")
print(f"TP: {len(tp_list)}, FP: {len(fp_list)}, FN: {len(fn_list)}")

# Analyze FP NES values
print(f"\n=== FP analysis (Python NES values) ===")
for tf, ct, r_val, py_val in sorted(fp_list, key=lambda x: x[3]):
    print(f"  {tf}/{ct}: Py_NES={py_val:.4f}, R_NES={r_val:.4f}")

# Analyze FN NES values
print(f"\n=== FN analysis (R NES values) ===")
for tf, ct, r_val, py_val in sorted(fn_list, key=lambda x: x[2], reverse=True):
    print(f"  {tf}/{ct}: R_NES={r_val:.4f}, Py_NES={py_val:.4f}")

# Show TP NES range
tp_py = [x[3] for x in tp_list]
tp_r = [x[2] for x in tp_list]
print(f"\n=== TP NES range ===")
print(f"  R_NES: {min(tp_r):.4f} to {max(tp_r):.4f} (mean={np.mean(tp_r):.4f})")
print(f"  Py_NES: {min(tp_py):.4f} to {max(tp_py):.4f} (mean={np.mean(tp_py):.4f})")

del result; gc.collect()
