"""GSEA matching R's fgseaMultilevel NES normalization.

NES = ES / mean(|null_ES|) matching R's exact formula (not gseapy's ES/std).
Uses vectorized batch ES computation for speed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import genextreme, norm as _norm_dist


def _batch_es_fast(values, obs_mask, perm_masks):
    """Compute ES for observed + all permutations, vectorized."""
    n = len(values)
    pow_vals = np.power(np.abs(values), 1.0).astype(np.float32)
    n_perm = len(perm_masks)

    masks = np.empty((n_perm + 1, n), dtype=np.float32)
    masks[0] = obs_mask
    masks[1:] = perm_masks

    n_hit = masks.sum(axis=1)
    n_miss = n - n_hit

    valid = (n_hit > 0) & (n_miss > 0)
    es_all = np.zeros(n_perm + 1, dtype=np.float64)

    if not valid.any():
        return 0.0, np.zeros(n_perm)

    vm = masks[valid]
    nh = n_hit[valid]
    nm = n_miss[valid]

    hit_sum = (vm * pow_vals[None, :]).sum(axis=1)
    valid2 = hit_sum > 0
    if not valid2.any():
        return 0.0, np.zeros(n_perm)

    vm2 = vm[valid2]
    nh2 = nh[valid2]
    nm2 = nm[valid2]
    hs2 = hit_sum[valid2]

    cum_hit = np.cumsum(vm2 * pow_vals[None, :], axis=1) / hs2[:, None]
    cum_miss = np.cumsum(1.0 - vm2, axis=1) / nm2[:, None]
    rs = cum_hit - cum_miss

    max_pos = rs.max(axis=1)
    max_neg = rs.min(axis=1)
    es_vals = np.where(np.abs(max_pos) >= np.abs(max_neg), max_pos, max_neg)

    v_idx = np.where(valid)[0]
    v2_idx = np.where(valid2)[0]
    es_all[v_idx[v2_idx]] = es_vals

    return float(es_all[0]), es_all[1:]


def _compute_es_single(values, obs_mask):
    """Compute ES for a single hit mask."""
    n = len(values)
    pow_vals = np.power(np.abs(values), 1.0)
    n_hit = int(obs_mask.sum())
    n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return 0.0

    hit_sum = (obs_mask * pow_vals).sum()
    if hit_sum == 0:
        return 0.0

    cum_hit = np.cumsum(obs_mask * pow_vals) / hit_sum
    cum_miss = np.cumsum(~obs_mask) / n_miss
    rs = cum_hit - cum_miss
    max_pos = rs.max()
    max_neg = rs.min()
    return max_pos if abs(max_pos) >= abs(max_neg) else max_neg


def run_gsea_per_tf(fc_values, fc_names, gene_sets, min_size=5, max_size=600,
                    n_perm=500, p_threshold=0.05, seed=42, z_threshold=None,
                    use_bh=False):
    """Run GSEA per TF matching R's NES normalization.

    NES = ES / mean(|null_ES|), matching R's fgsea formula.
    Uses vectorized argpartition for fast permutation generation.
    If use_bh=True, applies BH FDR correction before filtering (matching R).
    """
    gene_to_idx = {g: i for i, g in enumerate(fc_names)}
    n = len(fc_names)
    rng = np.random.RandomState(seed)
    values = fc_values.astype(np.float64)

    # Pre-compute gene set sizes and build observed masks
    tf_data = {}  # tf -> (gs, obs_mask)
    for tf, genes in gene_sets.items():
        genes_in = [g for g in genes if g in gene_to_idx]
        gs = len(genes_in)
        if gs < min_size or gs > max_size:
            continue
        obs = np.zeros(n, dtype=bool)
        for g in genes_in:
            obs[gene_to_idx[g]] = True
        tf_data[tf] = (gs, obs)

    if not tf_data:
        return {}

    # Pre-generate random index pool for each unique n_hit using argpartition
    unique_sizes = set(gs for gs, _ in tf_data.values())
    perm_pool = {}  # n_hit -> (n_perm, n_hit) indices
    BATCH_SZ = min(100, n_perm)  # batch to avoid large contiguous allocations
    for sz in unique_sizes:
        parts = []
        for _ in range(0, n_perm, BATCH_SZ):
            bs = min(BATCH_SZ, n_perm - len(parts) * BATCH_SZ)
            if bs <= 0:
                break
            rand_mat = rng.random((bs, n))
            parts.append(np.argpartition(rand_mat, sz, axis=1)[:, :sz])
            del rand_mat
        perm_pool[sz] = np.vstack(parts)

    # Phase 1: Compute NES and raw p-values for all TFs
    raw_results = {}  # tf -> (nes, p_val, es_obs, es_null_all)
    for tf, (gs, obs_mask) in tf_data.items():
        idx = perm_pool[gs]  # (n_perm, gs)

        # Compute observed ES once
        es_obs = _compute_es_single(values, obs_mask)

        # Process permutations in batches to avoid large allocations
        es_null_all = np.zeros(n_perm, dtype=np.float64)
        batch_size = min(100, n_perm)
        for b_start in range(0, n_perm, batch_size):
            b_end = min(b_start + batch_size, n_perm)
            bs = b_end - b_start
            perm_masks = np.zeros((bs, n), dtype=bool)
            rows = np.arange(bs)[:, None]
            perm_masks[rows, idx[b_start:b_end]] = True
            _, es_batch = _batch_es_fast(values, obs_mask, perm_masks)
            es_null_all[b_start:b_end] = es_batch
            del perm_masks

        null_abs_mean = np.abs(es_null_all).mean()
        if null_abs_mean < 1e-15:
            continue
        nes = es_obs / null_abs_mean

        p_val = (np.abs(es_null_all) >= abs(es_obs)).mean()
        if p_val == 0:
            p_val = 1.0 / n_perm

        raw_results[tf] = (nes, p_val, es_obs, es_null_all)

    if not raw_results:
        return {}

    # Phase 2: Apply BH correction if requested
    if use_bh and len(raw_results) > 1:
        from statsmodels.stats.multitest import multipletests
        tf_list = list(raw_results.keys())
        pvals = np.array([raw_results[tf][1] for tf in tf_list])
        nes_arr = np.array([raw_results[tf][0] for tf in tf_list])
        # Only correct TFs with positive NES (same as R)
        pos_mask = nes_arr > 0
        if pos_mask.any():
            adj_pvals = np.ones(len(tf_list))
            pos_pvals = pvals[pos_mask]
            if len(pos_pvals) > 0:
                _, adj_pos, _, _ = multipletests(pos_pvals, method='fdr_bh')
                adj_pvals[pos_mask] = adj_pos
            # Filter
            results = {}
            for i, tf in enumerate(tf_list):
                if adj_pvals[i] < p_threshold and nes_arr[i] > 0:
                    # Optional z-test filter
                    if z_threshold is not None:
                        es_obs, es_null = raw_results[tf][2], raw_results[tf][3]
                        null_std = np.std(es_null)
                        if null_std > 1e-15:
                            z = abs(es_obs) / null_std
                            z_p = 2 * (1 - _norm_dist.cdf(z))
                            if z_p >= z_threshold:
                                continue
                    results[tf] = nes_arr[i]
            return results
        return {}

    # No BH correction - use raw p-values
    results = {}
    for tf, (nes, p_val, es_obs, es_null_all) in raw_results.items():
        # Optional z-test filter for borderline cases
        if z_threshold is not None and nes > 0 and p_val < p_threshold:
            null_std = np.std(es_null_all)
            if null_std > 1e-15:
                z = abs(es_obs) / null_std
                z_p = 2 * (1 - _norm_dist.cdf(z))
                if z_p >= z_threshold:
                    continue  # z-test rejects this TF

        if nes > 0 and p_val < p_threshold:
            results[tf] = nes

    return results
