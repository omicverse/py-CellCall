#!/usr/bin/env python
"""Run the actual pipeline and compare with R reference."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr
from cellcall.connect_profile import ConnectProfile
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

# Run pipeline with p_adjust=0.02
cp = ConnectProfile(obj, org='Homo sapiens', verbose=False)
cp.getCorrelatedGenes(p_value_cor=0.05, cor_value=0.1, top_target_cor=1.0)
cp.getGSEA(p_adjust=0.02, n_perm=500)

# Get regulons_matrix
py_rm = cp.regulons_matrix
print(f"Python regulons_matrix shape: {py_rm.shape}")
print(f"Python active entries: {(py_rm > 0).sum().sum()}")
print(f"Python active TFs: {(py_rm > 0).any(axis=1).sum()}")

# Load R reference
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

# Align
common_tfs = sorted(set(py_rm.index) & set(r_ref.index))
common_cts = sorted(set(py_rm.columns) & set(r_ref.columns))
py_aligned = py_rm.loc[common_tfs, common_cts]
r_aligned = r_ref.loc[common_tfs, common_cts]

# Pearson on active entries
r_flat = r_aligned.values.flatten()
p_flat = py_aligned.values.flatten()
active_mask = (r_flat != 0) | (p_flat != 0)
if active_mask.sum() > 0:
    pearson, _ = pearsonr(r_flat[active_mask], p_flat[active_mask])
    print(f"\nPearson on active entries: {pearson:.4f}")
    print(f"Active entries: {active_mask.sum()}")

# Full matrix Pearson (including zeros)
pearson_full, _ = pearsonr(r_flat, p_flat)
print(f"Pearson on full matrix: {pearson_full:.4f}")

# Show differences
print("\nDifferences:")
for tf in common_tfs:
    for ct in common_cts:
        r_val = r_aligned.loc[tf, ct]
        p_val = py_aligned.loc[tf, ct]
        if (r_val > 0) != (p_val > 0):
            tag = "FP" if p_val > 0 else "FN"
            print(f"  {tag}: {tf}/{ct} py={p_val:.3f} r={r_val:.3f}")
        elif r_val > 0 and p_val > 0:
            diff = abs(p_val - r_val) / max(r_val, 0.01)
            if diff > 0.1:
                print(f"  DIFF: {tf}/{ct} py={p_val:.3f} r={r_val:.3f} ({diff:.1%})")
