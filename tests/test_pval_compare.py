#!/usr/bin/env python
"""Diagnostic: Run pipeline verbosely to see gene sets for FP/FN TFs."""
import sys, os, warnings, gc, time
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

# Monkey-patch run_gsea_per_tf to log TF details
import cellcall.fgsea as fgsea_mod
original_run = fgsea_mod.run_gsea_per_tf

def patched_run(fc_values, fc_names, gene_sets, **kwargs):
    """Log details for FP/FN TFs."""
    from cellcall.fgsea import _calc_es_single
    import numpy as np

    target_tfs = {'EGR1', 'E2F4', 'IRF7', 'NFKB2', 'PML', 'RXRA',
                  'FOXO1', 'HIF1A', 'NFATC3'}

    for tf in target_tfs:
        if tf in gene_sets:
            genes = gene_sets[tf]
            gene_to_idx = {g: i for i, g in enumerate(fc_names)}
            k = len([g for g in genes if g in gene_to_idx])
            print(f"  [PATCH] {tf}: gene_set_size={len(genes)}, in_data={k}")

    result = original_run(fc_values, fc_names, gene_sets, **kwargs)

    for tf in target_tfs:
        if tf in result:
            print(f"  [PATCH] {tf}: NES={result[tf]:.4f} (ACTIVE)")
        else:
            print(f"  [PATCH] {tf}: not active")

    return result

fgsea_mod.run_gsea_per_tf = patched_run

print("Running pipeline with patched GSEA...")
result = connect_profile(obj, org='Homo sapiens', verbose=True, n_perm=500, p_adjust=0.02)
py_rm = result['regulons_matrix']

# Evaluate
common_tfs = sorted(set(py_rm.index) & set(r_ref.index))
common_cts = sorted(set(py_rm.columns) & set(r_ref.columns))
py_aligned = py_rm.loc[common_tfs, common_cts]
r_aligned = r_ref.loc[common_tfs, common_cts]

from scipy.stats import pearsonr
r_flat = r_aligned.values.flatten()
p_flat = py_aligned.values.flatten()
pearson_full, _ = pearsonr(r_flat, p_flat)
print(f"\nPearson: {pearson_full:.4f}")

# Check FP/FN details
for tf in ['EGR1', 'E2F4', 'IRF7', 'NFKB2', 'PML', 'RXRA', 'FOXO1', 'HIF1A', 'NFATC3']:
    if tf in py_rm.index and 'SPGed' in py_rm.columns:
        py_val = py_rm.loc[tf, 'SPGed'] if tf in py_rm.index else 0
        r_val = r_ref.loc[tf, 'SPGed'] if tf in r_ref.index else 0
        print(f"  {tf}/SPGed: Py={py_val:.4f}, R={r_val:.4f}")
    if tf in py_rm.index and 'SSC' in py_rm.columns:
        py_val = py_rm.loc[tf, 'SSC'] if tf in py_rm.index else 0
        r_val = r_ref.loc[tf, 'SSC'] if tf in r_ref.index else 0
        print(f"  {tf}/SSC: Py={py_val:.4f}, R={r_val:.4f}")
    if tf in py_rm.index and 'ST' in py_rm.columns:
        py_val = py_rm.loc[tf, 'ST'] if tf in py_rm.index else 0
        r_val = r_ref.loc[tf, 'ST'] if tf in r_ref.index else 0
        print(f"  {tf}/ST: Py={py_val:.4f}, R={r_val:.4f}")

del result, obj; gc.collect()
print("\nDone!")
