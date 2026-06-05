#!/usr/bin/env python
"""Sweep key parameter combinations to find optimal settings."""
import sys, os, warnings, gc
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

def quick_eval(py_rm):
    common_tfs = sorted(set(py_rm.index) & set(r_ref.index))
    common_cts = sorted(set(py_rm.columns) & set(r_ref.columns))
    py_aligned = py_rm.loc[common_tfs, common_cts]
    r_aligned = r_ref.loc[common_tfs, common_cts]
    r_flat = r_aligned.values.flatten()
    p_flat = py_aligned.values.flatten()
    pearson_full, _ = pearsonr(r_flat, p_flat)
    r_active = set()
    p_active = set()
    for i, tf in enumerate(common_tfs):
        for j, ct in enumerate(common_cts):
            if r_aligned.iloc[i, j] > 0: r_active.add((tf, ct))
            if py_aligned.iloc[i, j] > 0: p_active.add((tf, ct))
    fp = len(p_active - r_active)
    fn = len(r_active - p_active)
    n_active = (py_rm > 0).sum().sum()
    return pearson_full, n_active, fp, fn

# Test key combinations
configs = [
    ("baseline",       dict(p_adjust=0.02, cor_value=0.1, p_value_cor=0.05, n_perm=500)),
    ("tight cor=0.15", dict(p_adjust=0.02, cor_value=0.15, p_value_cor=0.05, n_perm=500)),
    ("loose p=0.03",   dict(p_adjust=0.03, cor_value=0.1, p_value_cor=0.05, n_perm=500)),
    ("tight p=0.015",  dict(p_adjust=0.015, cor_value=0.1, p_value_cor=0.05, n_perm=500)),
    ("cor=0.12, p=0.025", dict(p_adjust=0.025, cor_value=0.12, p_value_cor=0.05, n_perm=500)),
    ("cor=0.08, p=0.03", dict(p_adjust=0.03, cor_value=0.08, p_value_cor=0.05, n_perm=500)),
]

for label, params in configs:
    result = connect_profile(obj, org='Homo sapiens', verbose=False,
                             use_multilevel=False, **params)
    p, n, fp, fn = quick_eval(result['regulons_matrix'])
    print(f"  {label:25s}: Pearson={p:.4f} n={n} FP={fp} FN={fn}")
    del result; gc.collect()

print("\nDone!")
