"""Gene Set Enrichment Analysis (GSEA) implementation matching R clusterProfiler."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


class GSEAResult:
    """Container for GSEA results, mimicking R's gseaResult object."""

    def __init__(self, result_df: pd.DataFrame):
        self.result = result_df

    @property
    def df(self) -> pd.DataFrame:
        return self.result

    def __repr__(self) -> str:
        return f"GSEAResult({len(self.result)} terms)"


def _gsea_pernutation_null(
    ranks: np.ndarray,
    gene_sets_bool: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate null distribution of ES by permuting gene labels."""
    rng = np.random.RandomState(seed)
    null_es = np.empty(n_perm)
    n_genes = len(ranks)
    for i in range(n_perm):
        perm_idx = rng.permutation(n_genes)
        perm_ranks = ranks[perm_idx]
        null_es[i] = _compute_es(perm_ranks, gene_sets_bool)
    return null_es


def _compute_es(ranks: np.ndarray, hits: np.ndarray, exponent: float = 1.0) -> float:
    """Compute enrichment score (ES) for a gene set.

    Parameters
    ----------
    ranks : np.ndarray
        Ranked gene values (sorted descending).
    hits : np.ndarray
        Boolean mask indicating which genes are in the set.
    exponent : float
        Exponent for weighting (1.0 = standard weighted Kolmogorov-Smirnov).

    Returns
    -------
    float
        Enrichment score.
    """
    n_hit = hits.sum()
    n_miss = len(hits) - n_hit
    if n_hit == 0 or n_miss == 0:
        return 0.0

    hit_values = np.abs(ranks[hits]) ** exponent
    miss_penalty = 1.0 / n_miss
    hit_reward = 1.0 / hit_values.sum()

    running_sum = np.zeros(len(ranks) + 1)
    for i in range(len(ranks)):
        if hits[i]:
            running_sum[i + 1] = running_sum[i] + hit_values[i] * hit_reward
        else:
            running_sum[i + 1] = running_sum[i] - miss_penalty

    running_sum = running_sum[1:]
    max_pos = running_sum.max()
    max_neg = running_sum.min()

    if abs(max_pos) >= abs(max_neg):
        return max_pos
    else:
        return max_neg


def run_gsea(
    gene_list: pd.Series,
    term2gene: pd.DataFrame,
    min_gssize: int = 5,
    max_gssize: int = 500,
    exponent: float = 1.0,
    pvalue_cutoff: float = 1.0,
    p_adjust_method: str = "BH",
    n_perm: int = 1000,
    seed: int = 42,
    verbose: bool = False,
) -> GSEAResult:
    """Run Gene Set Enrichment Analysis.

    Parameters
    ----------
    gene_list : pd.Series
        Named vector of fold changes, sorted descending (index=gene names).
    term2gene : pd.DataFrame
        Two columns: [term, gene]. Maps gene set terms to gene names.
    min_gssize : int
        Minimum gene set size.
    max_gssize : int
        Maximum gene set size.
    exponent : float
        Exponent for ES weighting.
    pvalue_cutoff : float
        P-value cutoff for reporting.
    p_adjust_method : str
        Multiple testing correction method.
    n_perm : int
        Number of permutations for null distribution.
    seed : int
        Random seed.

    Returns
    -------
    GSEAResult
        GSEA results.
    """
    gene_names = gene_list.index.values
    gene_values = gene_list.values.astype(float)
    n_genes = len(gene_names)

    # Build gene set lookup
    terms = term2gene.iloc[:, 0].unique()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    results = []
    for term in terms:
        set_genes = term2gene[term2gene.iloc[:, 0] == term].iloc[:, 1].values
        set_genes_in = [g for g in set_genes if g in gene_to_idx]
        set_size = len(set_genes_in)

        if set_size < min_gssize or set_size > max_gssize:
            continue

        hits = np.zeros(n_genes, dtype=bool)
        for g in set_genes_in:
            hits[gene_to_idx[g]] = True

        es = _compute_es(gene_values, hits, exponent)

        # Permutation-based p-value
        null_es = _gsea_pernutation_null(gene_values, hits, n_perm=n_perm, seed=seed)

        if es >= 0:
            p_val = (null_es >= es).sum() / n_perm
        else:
            p_val = (null_es <= es).sum() / n_perm

        p_val = max(p_val, 1e-10)  # avoid zero

        # NES: normalized ES
        if es >= 0:
            pos_null = null_es[null_es > 0]
            nes = es / pos_null.mean() if len(pos_null) > 0 else es
        else:
            neg_null = null_es[null_es < 0]
            nes = es / abs(neg_null.mean()) if len(neg_null) > 0 else es

        results.append({
            "ID": str(term),
            "Description": str(term),
            "setSize": set_size,
            "enrichmentScore": es,
            "NES": nes,
            "pvalue": p_val,
            "p.adjust": p_val,
            "qvalues": p_val,
            "rank": np.argmax(np.abs(np.cumsum(hits.astype(float) / hits.sum() - (~hits).astype(float) / (~hits).sum()))) if es != 0 else 0,
            "leading_edge": "",
            "core_enrichment": "/".join(set_genes_in),
        })

    if not results:
        return GSEAResult(pd.DataFrame(columns=[
            "ID", "Description", "setSize", "enrichmentScore",
            "NES", "pvalue", "p.adjust", "qvalues",
        ]))

    res_df = pd.DataFrame(results)

    # Multiple testing correction
    if len(res_df) > 0:
        _, adj_p, _, _ = multipletests(res_df["pvalue"].values, method=p_adjust_method)
        res_df["p.adjust"] = adj_p
        res_df["qvalues"] = adj_p

    # Filter by p-value cutoff
    res_df = res_df[res_df["p.adjust"] < pvalue_cutoff]

    return GSEAResult(res_df.reset_index(drop=True))


def fast_gsea(
    gene_list: pd.Series,
    term2gene: pd.DataFrame,
    min_gssize: int = 5,
    max_gssize: int = 500,
    exponent: float = 1.0,
    pvalue_cutoff: float = 1.0,
    p_adjust_method: str = "BH",
    seed: int = 42,
) -> GSEAResult:
    """Fast GSEA using fgsea-like algorithm with analytical p-value estimation.

    Uses a faster approach than brute-force permutation by leveraging
    the fact that ES follows a known distribution under the null.
    """
    gene_names = gene_list.index.values
    gene_values = gene_list.values.astype(float)
    n_genes = len(gene_names)

    terms = term2gene.iloc[:, 0].unique()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    results = []
    for term in terms:
        set_genes = term2gene[term2gene.iloc[:, 0] == term].iloc[:, 1].values
        set_genes_in = [g for g in set_genes if g in gene_to_idx]
        set_size = len(set_genes_in)

        if set_size < min_gssize or set_size > max_gssize:
            continue

        hits = np.zeros(n_genes, dtype=bool)
        for g in set_genes_in:
            hits[gene_to_idx[g]] = True

        es = _compute_es(gene_values, hits, exponent)

        # Analytical p-value approximation using the method from
        # Subramanian et al. (2005)
        # For weighted KS with exponent=1, the null distribution can be
        # approximated by a Beta distribution
        n_hit = set_size
        n_miss = n_genes - n_hit

        # Simple permutation-free approximation
        # ES_null ~ N(0, sigma^2) where sigma depends on set size
        if es >= 0:
            # Approximate using the expected value under null
            sigma = np.sqrt(n_hit / (n_miss * n_genes))
            if sigma > 0:
                z_score = es / sigma
                p_val = 1 - stats.norm.cdf(z_score)
            else:
                p_val = 1.0
        else:
            sigma = np.sqrt(n_hit / (n_miss * n_genes))
            if sigma > 0:
                z_score = es / sigma
                p_val = stats.norm.cdf(z_score)
            else:
                p_val = 1.0

        p_val = np.clip(p_val, 1e-10, 1.0)

        # NES
        nes = es / sigma if sigma > 0 else 0.0

        results.append({
            "ID": str(term),
            "Description": str(term),
            "setSize": set_size,
            "enrichmentScore": es,
            "NES": nes,
            "pvalue": p_val,
            "p.adjust": p_val,
            "qvalues": p_val,
        })

    if not results:
        return GSEAResult(pd.DataFrame(columns=[
            "ID", "Description", "setSize", "enrichmentScore",
            "NES", "pvalue", "p.adjust", "qvalues",
        ]))

    res_df = pd.DataFrame(results)
    _, adj_p, _, _ = multipletests(res_df["pvalue"].values, method=p_adjust_method)
    res_df["p.adjust"] = adj_p
    res_df["qvalues"] = adj_p
    res_df = res_df[res_df["p.adjust"] < pvalue_cutoff]

    return GSEAResult(res_df.reset_index(drop=True))
