#!/usr/bin/env python
"""Quick test of fgseaMultilevel implementation."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from cellcall.fgsea import _calc_es, _build_multilevel, _get_pvalue_from_levels

# Create a simple test: 100 genes, 10 in the pathway
rng = np.random.RandomState(42)
n = 100
values = rng.randn(n)

# True pathway: top 10 by values
true_pathway = np.argsort(values)[-10:]

# Compute true ES
hit_mask = np.zeros(n, dtype=bool)
hit_mask[true_pathway] = True
es_true = _calc_es(values, hit_mask)
print(f"True ES: {es_true:.4f}")

# Random pathway (should have lower ES)
random_pathway = np.sort(rng.choice(n, 10, replace=False))
hit_mask_rand = np.zeros(n, dtype=bool)
hit_mask_rand[random_pathway] = True
es_rand = _calc_es(values, hit_mask_rand)
print(f"Random ES: {es_rand:.4f}")

# Build multilevel for k=10
print("Building multilevel (sample_size=21)...")
levels = _build_multilevel(values, 10, 21, abs(es_true) * 0.5, rng,
                           n_simple_perm=200, max_levels=10)
print(f"Built {len(levels)} levels")
for i, lev in enumerate(levels):
    print(f"  Level {i}: bound={lev['bound']:.4f} high={len(lev['high_scores'])} low={len(lev['low_scores'])}")

# Get p-value for true pathway
pval = _get_pvalue_from_levels(abs(es_true), levels, 21)
print(f"\nP-value for true pathway: {pval:.6f}")

# Get p-value for random pathway
pval_rand = _get_pvalue_from_levels(abs(es_rand), levels, 21)
print(f"P-value for random pathway: {pval_rand:.6f}")

print("\nDone!")
