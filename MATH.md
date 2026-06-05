# Mathematical Bounds for py-cellcall

## B1: GSEA NES Matching

**Bound type**: Empirical observation (not provable bound)

**Description**: For shared TFs between R and Python, the NES values are nearly identical:
- ST (23 shared): Pearson=0.9974, NES ratio = 1.010 ± 0.014
- SSC (11 shared): Pearson=0.9614, NES ratio = 1.018 ± 0.028

**Root cause of regulons_matrix gap (0.886)**:
- R uses `fgsea::fgseaMultilevel` for exact p-value computation
- Python uses permutation-based p-values with numpy's RandomState
- R and Python RNG produce different permutation sequences (same seed → different results)
- This leads to different TF selections: ~39/42 shared (93%), but Python has ~11 extra TFs
- The extra TFs have small NES values that reduce global Pearson correlation

**No closed-form bound**: The gap is algorithmic (fgsea exact vs permutation), not numerical.

## B2: Spearman Correlation (Tie Handling)

**Bound type**: (E) Exact identity for unique values; bounded for tied values

**Description**: For genes with unique expression values, `scipy.stats.rankdata(method='average')`
+ Pearson-on-ranks is mathematically identical to Spearman rank correlation.

For tied values (common in sparse scRNA-seq with ~75% zeros):
- Both R and Python use average ranking for ties
- P-value computation: R uses AS89 algorithm (via cor.test), Python uses t-distribution approximation
- For n > 100, the difference is < 0.001 in p-values

## B3: Fold Change Computation

**Bound type**: (E) Exact algebraic identity

**Description**: Python's `np.percentile(values, 75, axis=1)` with `interpolation='linear'`
is mathematically identical to R's `quantile(values, probs=0.75, names=FALSE)` (type 7).
Both use linear interpolation between adjacent order statistics.

Verified: expr_mean Pearson = 1.0000 (exact match).

## B4: CPM Normalization

**Bound type**: (E) Exact algebraic identity

**Description**: CPM = count / column_sum * 1e6. This is a simple scalar division,
identical in R and Python. Verified: column sums match exactly.
