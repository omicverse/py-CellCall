#!/usr/bin/env python
"""Compare intermediate results between R and Python."""
import os
import sys
import pandas as pd
import numpy as np

ref_dir = os.path.join(os.path.dirname(__file__), "reference")
test_tfs = ['PBX1', 'SMAD1', 'PPARGC1A', 'DDIT3', 'RORA', 'SMAD9', 'NFYB',
            'E2F4', 'RUNX2', 'NFATC3', 'FOXO1', 'EGR1']
cell_types = ['ST', 'SSC', 'SPGing', 'SPGed']

print("=== Fold Change Comparison ===")
for ct in cell_types:
    py_fc_file = os.path.join(ref_dir, f"fc_{ct}.csv")
    r_fc_file = os.path.join(ref_dir, f"r_fc_{ct}.csv")
    if not os.path.exists(py_fc_file) or not os.path.exists(r_fc_file):
        print(f"  {ct}: missing files")
        continue

    py_fc = pd.read_csv(py_fc_file, index_col=0)
    r_fc = pd.read_csv(r_fc_file, index_col=0)

    common = sorted(set(py_fc.index) & set(r_fc.index))
    py_vals = py_fc.loc[common].values.flatten()
    r_vals = r_fc.loc[common].values.flatten()
    corr = np.corrcoef(py_vals, r_vals)[0, 1]
    max_diff = np.abs(py_vals - r_vals).max()
    print(f"  {ct}: {len(common)} genes, Pearson={corr:.6f}, max_diff={max_diff:.6f}")

    if max_diff > 0.01:
        diffs = np.abs(py_vals - r_vals)
        top_idx = np.argsort(-diffs)[:5]
        for i in top_idx:
            gene = common[i]
            print(f"    {gene}: R={r_vals[i]:.6f}, Py={py_vals[i]:.6f}")

print("\n=== Gene Set Comparison ===")
for ct in cell_types:
    for tf in test_tfs:
        py_file = os.path.join(ref_dir, f"gs_{tf}_{ct}.txt")
        r_file = os.path.join(ref_dir, f"r_gs_{tf}_{ct}.txt")
        if not os.path.exists(py_file) or not os.path.exists(r_file):
            continue

        with open(py_file) as f:
            py_genes = set(f.read().strip().split('\n'))
        with open(r_file) as f:
            r_genes = set(f.read().strip().split('\n'))

        if not r_genes and not py_genes:
            continue
        overlap = py_genes & r_genes
        py_only = py_genes - r_genes
        r_only = r_genes - py_genes
        jaccard = len(overlap) / max(len(py_genes | r_genes), 1)

        print(f"  {tf}/{ct}: R={len(r_genes)}, Py={len(py_genes)}, "
              f"overlap={len(overlap)}, Jaccard={jaccard:.3f}")
        if py_only and len(py_only) <= 10:
            print(f"    Py-only: {sorted(py_only)}")
        if r_only and len(r_only) <= 10:
            print(f"    R-only: {sorted(r_only)}")
