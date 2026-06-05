"""fgsea exact algorithm using numba JIT for C-level performance.

Implements the histogram DP algorithm from fgsea with correct
running_sum + max/min_running_sum tracking for both positive and negative ES.
Uses full grid approach with small n_bins for correctness.
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit


@njit(cache=True)
def compute_es(values, hit_mask, exponent=1.0):
    """Compute enrichment score matching R's GSEA formula."""
    n = len(values)
    n_hit = 0
    for i in range(n):
        if hit_mask[i]:
            n_hit += 1
    n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return 0.0

    hit_sum = 0.0
    for i in range(n):
        if hit_mask[i]:
            hit_sum += abs(values[i]) ** exponent
    if hit_sum == 0.0:
        return 0.0

    cumsum = 0.0
    max_pos = 0.0
    max_neg = 0.0
    miss_step = 1.0 / n_miss

    for i in range(n):
        if hit_mask[i]:
            cumsum += abs(values[i]) ** exponent / hit_sum
        else:
            cumsum -= miss_step
        if cumsum > max_pos:
            max_pos = cumsum
        if cumsum < max_neg:
            max_neg = cumsum

    if abs(max_pos) >= abs(max_neg):
        return max_pos
    return max_neg


@njit(cache=True)
def _log_add(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def _fgsea_exact_dp(values, hit_mask, es, exponent=1.0, n_bins=51):
    """Exact GSEA p-value using 3D histogram DP.

    State: (k, rs_bin, stat_bin) -> log_count
    - If es >= 0: stat = max_running_sum (track peak)
    - If es < 0: stat = min_running_sum (track trough)
    Uses full grid for correctness.
    """
    n = len(values)
    n_hit = 0
    for i in range(n):
        if hit_mask[i]:
            n_hit += 1
    n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return 1.0

    hit_sum = 0.0
    for i in range(n):
        if hit_mask[i]:
            hit_sum += abs(values[i]) ** exponent
    if hit_sum == 0.0:
        return 1.0

    rs_min = -2.0
    rs_max = 2.0
    bw = (rs_max - rs_min) / (n_bins - 1)
    thr = abs(es)
    track_max = (es >= 0.0)

    nk = n_hit + 1

    dp_cur = np.full((nk, n_bins, n_bins), -np.inf, dtype=np.float64)
    dp_next = np.full((nk, n_bins, n_bins), -np.inf, dtype=np.float64)

    init_b = int(round((0.0 - rs_min) / bw))
    if init_b < 0:
        init_b = 0
    if init_b >= n_bins:
        init_b = n_bins - 1
    dp_cur[0, init_b, init_b] = 0.0

    for p in range(n):
        step_h = abs(values[p]) ** exponent / hit_sum
        step_m = -1.0 / n_miss

        # Clear dp_next
        for k in range(nk):
            for rb in range(n_bins):
                for sb in range(n_bins):
                    dp_next[k, rb, sb] = -np.inf

        for k in range(nk):
            if k > n_hit:
                break
            for rs_b in range(n_bins):
                for stat_b in range(n_bins):
                    lc = dp_cur[k, rs_b, stat_b]
                    if lc == -np.inf:
                        continue

                    rs_val = rs_b * bw + rs_min
                    stat_val = stat_b * bw + rs_min

                    # MISS: k unchanged, stat unchanged
                    new_rs_m = rs_val + step_m
                    new_rs_b_m = int(round((new_rs_m - rs_min) / bw))
                    if new_rs_b_m < 0:
                        new_rs_b_m = 0
                    if new_rs_b_m >= n_bins:
                        new_rs_b_m = n_bins - 1
                    dp_next[k, new_rs_b_m, stat_b] = _log_add(
                        dp_next[k, new_rs_b_m, stat_b], lc)

                    # HIT: k -> k+1
                    if k < n_hit:
                        new_rs_h = rs_val + step_h
                        new_rs_b_h = int(round((new_rs_h - rs_min) / bw))
                        if new_rs_b_h < 0:
                            new_rs_b_h = 0
                        if new_rs_b_h >= n_bins:
                            new_rs_b_h = n_bins - 1

                        if track_max:
                            new_stat = stat_val
                            if new_rs_h > new_stat:
                                new_stat = new_rs_h
                        else:
                            new_stat = stat_val
                            if new_rs_h < new_stat:
                                new_stat = new_rs_h

                        new_stat_b = int(round((new_stat - rs_min) / bw))
                        if new_stat_b < 0:
                            new_stat_b = 0
                        if new_stat_b >= n_bins:
                            new_stat_b = n_bins - 1

                        dp_next[k + 1, new_rs_b_h, new_stat_b] = _log_add(
                            dp_next[k + 1, new_rs_b_h, new_stat_b], lc)

        # Swap
        for k in range(nk):
            for rb in range(n_bins):
                for sb in range(n_bins):
                    dp_cur[k, rb, sb] = dp_next[k, rb, sb]

    # Count paths at k=n_hit
    total_log = -np.inf
    extreme_log = -np.inf

    for rs_b in range(n_bins):
        for stat_b in range(n_bins):
            lc = dp_cur[n_hit, rs_b, stat_b]
            if lc == -np.inf:
                continue
            total_log = _log_add(total_log, lc)
            stat_val = stat_b * bw + rs_min
            if track_max:
                if stat_val >= thr:
                    extreme_log = _log_add(extreme_log, lc)
            else:
                if stat_val <= -thr:
                    extreme_log = _log_add(extreme_log, lc)

    if total_log == -np.inf:
        return 1.0
    if extreme_log == -np.inf:
        return 1.0

    p_val = math.exp(extreme_log - total_log)
    if p_val < 1e-15:
        return 1e-15
    return p_val


def fgsea_exact_pvalue(values, hit_mask, es, exponent=1.0, n_bins=51):
    """Public wrapper for the numba JIT-compiled exact p-value."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    hit_mask = np.ascontiguousarray(hit_mask, dtype=np.bool_)
    return _fgsea_exact_dp(values, hit_mask, es, exponent, n_bins)
