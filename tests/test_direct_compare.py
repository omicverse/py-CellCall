#!/usr/bin/env python
"""Direct comparison: run pipeline once, log gene set sizes and p-values for FP/FN TFs."""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from cellcall.connect_profile import connect_profile
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x
from cellcall.fgsea import _calc_es_single
import cellcall.fgsea as fgsea_mod

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

# Intercept run_gsea_per_tf to log details for FP/FN TFs
target_tfs = {'EGR1', 'E2F4', 'IRF7', 'NFKB2', 'PML', 'RXRA',
              'FOXO1', 'HIF1A', 'NFATC3'}
logged_info = {}

original_run = fgsea_mod.run_gsea_per_tf

def patched_run(fc_values, fc_names, gene_sets, **kwargs):
    """Log gene set sizes and compute ES for target TFs."""
    gene_to_idx = {g: i for i, g in enumerate(fc_names)}
    n = len(fc_names)
    values = fc_values.astype(np.float64)

    for tf in target_tfs:
        if tf in gene_sets:
            gs = gene_sets[tf]
            genes_in = [g for g in gs if g in gene_to_idx]
            k = len(genes_in)
            if k > 0:
                hit_mask = np.zeros(n, dtype=bool)
                for g in genes_in:
                    hit_mask[gene_to_idx[g]] = True
                es = _calc_es_single(values, hit_mask)
                logged_info[tf] = {'gs_size': len(gs), 'in_data': k, 'es': es}
            else:
                logged_info[tf] = {'gs_size': len(gs), 'in_data': 0, 'es': 0}
        else:
            logged_info[tf] = None

    result = original_run(fc_values, fc_names, gene_sets, **kwargs)
    return result

fgsea_mod.run_gsea_per_tf = patched_run

# Run pipeline for each cell type that has FP/FN TFs
for ct in ['SPGed', 'SSC', 'ST']:
    print(f"\n{'='*60}")
    print(f"Cell type: {ct}")
    print(f"{'='*60}")

    result = connect_profile(obj, org='Homo sapiens', verbose=False,
                             n_perm=500, p_adjust=0.02)
    py_rm = result['regulons_matrix']

    for tf in sorted(logged_info.keys()):
        info = logged_info[tf]
        if info is None:
            continue
        py_val = py_rm.loc[tf, ct] if tf in py_rm.index and ct in py_rm.columns else 0
        r_val = r_ref.loc[tf, ct] if tf in r_ref.index and ct in r_ref.columns else 0
        status = 'TP' if (py_val > 0 and r_val > 0) else \
                 'FP' if py_val > 0 else \
                 'FN' if r_val > 0 else 'TN'
        print(f"  {tf}/{ct}: gs={info['gs_size']} in_data={info['in_data']} "
              f"ES={info['es']:.4f} Py_NES={py_val:.4f} R_NES={r_val:.4f} [{status}]")

    del result; gc.collect()

del obj; gc.collect()
print("\nDone!")
