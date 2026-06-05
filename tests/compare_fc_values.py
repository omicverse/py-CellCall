#!/usr/bin/env python
"""Compare R and Python fold change values."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings('ignore')

import rdata
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from cellcall.connect_profile import _load_lr_tf_data, _handle_complex_receptors
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x
from cellcall.foldchange import mylog2foldchange

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')

my_expr = obj.data['withoutlog'].copy()
my_expr.columns = obj.meta_data['celltype'].values
triple_relation, target_relation = _load_lr_tf_data('Homo sapiens', True, extdata_dir)
triple_relation = triple_relation.drop(columns=['pathway_ID'])
complex_mask = triple_relation['Receptor_Symbol'].str.contains(',', na=False)
complex_list = triple_relation.loc[complex_mask, 'Receptor_Symbol'].unique().tolist()
complex_subunits = set()
for c in complex_list:
    complex_subunits.update(c.split(','))
all_genes_needed = set(triple_relation['Ligand_Symbol'].tolist() + triple_relation['Receptor_Symbol'].tolist() + triple_relation['TF_Symbol'].tolist() + target_relation['TF_Symbol'].tolist() + target_relation['Target_Symbol'].tolist() + list(complex_subunits))
common_genes = sorted(set(my_expr.index) & all_genes_needed)
expr_set = my_expr.loc[common_genes]
detect_genes = set(expr_set.index)
complex_matrix = _handle_complex_receptors(complex_list, expr_set, detect_genes)
if not complex_matrix.empty:
    expr_set = pd.concat([expr_set, complex_matrix])
expr_set = expr_set.loc[(expr_set != 0).any(axis=1)]

cell_types = list(dict.fromkeys(obj.meta_data['celltype'].values))
fc_list = mylog2foldchange(expr_set, cell_types, method='median', probs=0.75)

ref_dir = os.path.join('tests', 'reference')

for ct in ['ST', 'SSC', 'SPGing', 'SPGed']:
    py_fc_file = os.path.join(ref_dir, f'fc_{ct}.csv')
    r_fc_file = os.path.join(ref_dir, f'r_fc_{ct}.csv')

    if not os.path.exists(py_fc_file) or not os.path.exists(r_fc_file):
        print(f"Missing FC files for {ct}")
        continue

    py_fc = pd.read_csv(py_fc_file)
    r_fc = pd.read_csv(r_fc_file)

    # Merge on gene_id
    merged = pd.merge(py_fc, r_fc, on='gene_id', how='inner', suffixes=('_py', '_r'))

    # Correlation
    corr, pval = spearmanr(merged['log2fc_py'], merged['log2fc_r'])
    abs_diff = np.abs(merged['log2fc_py'] - merged['log2fc_r'])
    print(f"\n{ct}:")
    print(f"  n_genes: {len(merged)}")
    print(f"  Spearman corr: {corr:.6f}")
    print(f"  Pearson corr: {np.corrcoef(merged['log2fc_py'], merged['log2fc_r'])[0,1]:.6f}")
    print(f"  Mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6f}")
    print(f"  Std of diff: {abs_diff.std():.6f}")

    # Look at largest differences
    merged['abs_diff'] = abs_diff
    top_diff = merged.nlargest(10, 'abs_diff')
    print("  Top 10 differences:")
    for _, row in top_diff.iterrows():
        print(f"    {row['gene_id']:<15} py={row['log2fc_py']:.4f}  r={row['log2fc_r']:.4f}  diff={row['abs_diff']:.4f}")

    # Also check the gene set ordering (rank correlation)
    py_rank = py_fc.set_index('gene_id')['log2fc_py'].to_dict()
    r_rank = r_fc.set_index('gene_id')['log2fc_r'].to_dict()
    common_genes_fc = sorted(set(py_rank.keys()) & set(r_rank.keys()))
    py_vals = [py_rank[g] for g in common_genes_fc]
    r_vals = [r_rank[g] for g in common_genes_fc]
    rank_corr, _ = spearmanr(py_vals, r_vals)
    print(f"  Rank Spearman (all genes): {rank_corr:.6f}")

# Now check specific TF-related FC values
print("\n\n=== TF and target FC comparison (SPGed) ===")
ct = 'SPGed'
py_fc_file = os.path.join(ref_dir, f'fc_{ct}.csv')
r_fc_file = os.path.join(ref_dir, f'r_fc_{ct}.csv')
py_fc = pd.read_csv(py_fc_file).set_index('gene_id')['log2fc']
r_fc = pd.read_csv(r_fc_file).set_index('gene_id')['log2fc']

# Check EGR1 and its targets
from cellcall.fgsea import run_gsea_per_tf
target_lookup = {}
for tf, group in target_relation.groupby('TF_Symbol'):
    target_lookup[tf] = group['Target_Symbol'].tolist()

# EGR1
egr1_targets = target_lookup.get('EGR1', [])
egr1_in = [g for g in egr1_targets if g in detect_genes]
print(f"\nEGR1 targets in SPGed ({len(egr1_in)} genes):")
for g in ['EGR1'] + egr1_in[:15]:
    if g in py_fc.index and g in r_fc.index:
        diff = py_fc[g] - r_fc[g]
        print(f"  {g:<15} py={py_fc[g]:.4f}  r={r_fc[g]:.4f}  diff={diff:.4f}")
