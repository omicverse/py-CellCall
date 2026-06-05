#!/usr/bin/env python
"""Direct comparison of FC values between Python and R."""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr
from cellcall.connect_profile import connect_profile, _load_lr_tf_data, mylog2foldchange
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

# Get Python FC values
cell_types = obj.meta_data['celltype'].unique()
expr_fc = obj.data['withoutlog']
fc_list = mylog2foldchange(expr_fc, list(cell_types), method='median', probs=0.75)

# Get R FC values
import rpy2.robjects as ro
ro.r('library(cellcall)')
ro.r('data("example_Data")')

# Compute R FC for each cell type
for ct in ['SPGed', 'SSC', 'ST']:
    ro.r(f'''
    fc_r <- mylog2foldchange(example_Data, "{ct}", method="median", probs=0.75)
    write.csv(fc_r, "/tmp/r_fc_{ct}.csv")
    ''')
    r_fc = pd.read_csv(f'/tmp/r_fc_{ct}.csv', index_col=0)

    # Python FC
    py_fc = fc_list[ct].set_index("gene_id")["log2fc"]

    # Align
    common_genes = sorted(set(r_fc.index) & set(py_fc.index))
    r_vals = r_fc.loc[common_genes].values.flatten()
    py_vals = py_fc.loc[common_genes].values.flatten()

    pearson_r, _ = pearsonr(r_vals, py_vals)
    spearman_r, _ = spearmanr(r_vals, py_vals)

    # Check ranking agreement
    r_rank = pd.Series(r_vals, index=common_genes).rank(ascending=False)
    py_rank = pd.Series(py_vals, index=common_genes).rank(ascending=False)
    rank_corr, _ = spearmanr(r_rank, py_rank)

    # Check top genes
    r_top20 = set(r_fc.loc[common_genes].nlargest(20).index)
    py_top20 = set(py_fc.loc[common_genes].nlargest(20).index)
    overlap = len(r_top20 & py_top20)

    print(f"\n{ct}:")
    print(f"  FC Pearson: {pearson_r:.6f}")
    print(f"  FC Spearman: {spearman_r:.6f}")
    print(f"  Rank Spearman: {rank_corr:.6f}")
    print(f"  Top-20 overlap: {overlap}/20")
    print(f"  R top-5: {sorted(r_top20)[:5]}")
    print(f"  Py top-5: {sorted(py_top20)[:5]}")

    # Find disagreements in top 50
    r_top50 = set(r_fc.loc[common_genes].nlargest(50).index)
    py_top50 = set(py_fc.loc[common_genes].nlargest(50).index)
    only_r = r_top50 - py_top50
    only_py = py_top50 - r_top50
    if only_r or only_py:
        print(f"  Only in R top-50 ({len(only_r)}): {sorted(only_r)[:5]}")
        print(f"  Only in Py top-50 ({len(only_py)}): {sorted(only_py)[:5]}")

del obj; gc.collect()
print("\nDone!")
