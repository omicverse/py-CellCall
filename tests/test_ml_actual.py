#!/usr/bin/env python
"""Test fgseaMultilevel on actual cellcall data."""
import sys, os, warnings, gc, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
del parsed, data_df, data_cpm; gc.collect()

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
    r_active = set()
    p_active = set()
    for i, tf in enumerate(common_tfs):
        for j, ct in enumerate(common_cts):
            if r_aligned.iloc[i, j] > 0:
                r_active.add((tf, ct))
            if py_aligned.iloc[i, j] > 0:
                p_active.add((tf, ct))
    fp = p_active - r_active
    fn = r_active - p_active
    print(f"  {label}: full={pearson_full:.4f} active={pearson_act:.4f} n={n_active} FP={len(fp)} FN={len(fn)}")
    if fp or fn:
        for tf, ct in sorted(fp): print(f"    FP: {tf}/{ct}")
        for tf, ct in sorted(fn): print(f"    FN: {tf}/{ct}")
    return pearson_full

print("Running with multilevel (sample_size=31, n_perm=500)...")
t0 = time.time()
r1 = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500,
                     p_adjust=0.02, use_bh=False, use_multilevel=True, sample_size=31)
print(f"Time: {time.time()-t0:.1f}s")
evaluate(r1['regulons_matrix'], "multilevel s=31")
del r1; gc.collect()

print("\nDone!")
