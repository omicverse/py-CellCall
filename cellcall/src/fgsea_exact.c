/*
 * fgsea_exact.c - Exact GSEA via histogram DP, matching R's fgsea.
 *
 * Key insight: NES = ES / mean(|null_ES|), not ES / std(null).
 * The 4D DP tracks (position, k_hits, rs_bin, max_rs_bin).
 * After processing all positions, the max_rs distribution at k=n_hit
 * gives us the exact null ES distribution. We extract:
 *   1. p-value = P(|null_ES| >= |observed_ES|)
 *   2. null_mean = mean(|max_rs|) weighted by path counts
 *   3. NES = ES / null_mean
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Compute enrichment score */
double compute_es(const double *values, const int *hit_mask,
                  int n, double exponent) {
    int n_hit = 0, n_miss = 0;
    for (int i = 0; i < n; i++) {
        if (hit_mask[i]) n_hit++;
        else n_miss++;
    }
    if (n_hit == 0 || n_miss == 0) return 0.0;

    double hit_sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (hit_mask[i]) hit_sum += pow(fabs(values[i]), exponent);
    }
    if (hit_sum == 0.0) return 0.0;

    double cumsum = 0.0, max_pos = 0.0, max_neg = 0.0;
    double miss_step = 1.0 / n_miss;
    for (int i = 0; i < n; i++) {
        if (hit_mask[i]) cumsum += pow(fabs(values[i]), exponent) / hit_sum;
        else cumsum -= miss_step;
        if (cumsum > max_pos) max_pos = cumsum;
        if (cumsum < max_neg) max_neg = cumsum;
    }
    return fabs(max_pos) >= fabs(max_neg) ? max_pos : max_neg;
}

/*
 * Exact p-value + NES via histogram DP.
 *
 * Returns:
 *   es_out   - enrichment score
 *   nes_out  - normalized enrichment score (ES / mean(|null|))
 *   pval_out - exact p-value
 */
void fgsea_exact(
    const double *values, const int *hit_mask, int n,
    double exponent, int n_bins,
    double *es_out, double *nes_out, double *pval_out
) {
    int n_hit = 0, n_miss = 0;
    for (int i = 0; i < n; i++) {
        if (hit_mask[i]) n_hit++;
        else n_miss++;
    }
    if (n_hit == 0 || n_miss == 0) {
        *es_out = 0.0; *nes_out = 0.0; *pval_out = 1.0;
        return;
    }

    double hit_sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (hit_mask[i]) hit_sum += pow(fabs(values[i]), exponent);
    }
    if (hit_sum == 0.0) {
        *es_out = 0.0; *nes_out = 0.0; *pval_out = 1.0;
        return;
    }

    double es = compute_es(values, hit_mask, n, exponent);
    *es_out = es;

    double rs_min = -2.0, rs_max = 2.0;
    double bw = (rs_max - rs_min) / (n_bins - 1);
    double thr = fabs(es);

    int grid = n_bins * n_bins;
    int max_k = n_hit + 1;
    int total_size = max_k * grid;

    double *dp_cur = (double *)calloc(total_size, sizeof(double));
    double *dp_next = (double *)calloc(total_size, sizeof(double));
    if (!dp_cur || !dp_next) {
        free(dp_cur); free(dp_next);
        *nes_out = 0.0; *pval_out = 1.0;
        return;
    }

    /* Initialize: k=0, rs=0, max=0 */
    int init_b = (int)round((0.0 - rs_min) / bw);
    if (init_b < 0) init_b = 0;
    if (init_b >= n_bins) init_b = n_bins - 1;
    dp_cur[0 * grid + init_b * n_bins + init_b] = 1.0;

    int active_k_max = 0;

    for (int p = 0; p < n; p++) {
        double step_h = pow(fabs(values[p]), exponent) / hit_sum;
        double step_m = -1.0 / n_miss;

        memset(dp_next, 0, total_size * sizeof(double));
        int next_k_max = 0;

        for (int k = 0; k <= active_k_max && k <= n_hit; k++) {
            for (int rs_b = 0; rs_b < n_bins; rs_b++) {
                for (int max_b = 0; max_b < n_bins; max_b++) {
                    int idx = k * grid + rs_b * n_bins + max_b;
                    double cnt = dp_cur[idx];
                    if (cnt == 0) continue;

                    double rs_val = rs_b * bw + rs_min;
                    double max_val = max_b * bw + rs_min;

                    /* MISS: k stays, max unchanged */
                    double new_rs_m = rs_val + step_m;
                    int new_b_m = (int)round((new_rs_m - rs_min) / bw);
                    if (new_b_m < 0) new_b_m = 0;
                    if (new_b_m >= n_bins) new_b_m = n_bins - 1;
                    int idx_m = k * grid + new_b_m * n_bins + max_b;
                    dp_next[idx_m] += cnt;

                    /* HIT: k+1, max may change */
                    if (k < n_hit) {
                        double new_rs_h = rs_val + step_h;
                        int new_b_h = (int)round((new_rs_h - rs_min) / bw);
                        if (new_b_h < 0) new_b_h = 0;
                        if (new_b_h >= n_bins) new_b_h = n_bins - 1;

                        double new_max = (new_rs_h > max_val) ? new_rs_h : max_val;
                        int new_mb = (int)round((new_max - rs_min) / bw);
                        if (new_mb < 0) new_mb = 0;
                        if (new_mb >= n_bins) new_mb = n_bins - 1;

                        int idx_h = (k + 1) * grid + new_b_h * n_bins + new_mb;
                        dp_next[idx_h] += cnt;
                        if (k + 1 > next_k_max) next_k_max = k + 1;
                    }
                }
            }
        }

        /* Swap */
        double *tmp = dp_cur;
        dp_cur = dp_next;
        dp_next = tmp;
        active_k_max = next_k_max;
    }

    /* Extract null distribution at k=n_hit:
     * - p-value: fraction of paths with max_rs >= |es|
     * - null_mean: mean of |max_rs| across all paths (for NES normalization)
     */
    double extreme = 0.0, total = 0.0;
    double sum_abs_max = 0.0;

    for (int rs_b = 0; rs_b < n_bins; rs_b++) {
        for (int max_b = 0; max_b < n_bins; max_b++) {
            int idx = n_hit * grid + rs_b * n_bins + max_b;
            double cnt = dp_cur[idx];
            if (cnt == 0) continue;
            total += cnt;
            double max_val = max_b * bw + rs_min;
            sum_abs_max += cnt * fabs(max_val);
            if (max_val >= thr) extreme += cnt;
        }
    }

    free(dp_cur);
    free(dp_next);

    if (total == 0) {
        *pval_out = 1.0;
        *nes_out = 0.0;
        return;
    }

    double p_val = extreme / total;
    if (p_val < 1e-15) p_val = 1e-15;
    *pval_out = p_val;

    /* NES = ES / mean(|null_max_rs|) */
    double null_mean = sum_abs_max / total;
    if (null_mean < 1e-15) {
        *nes_out = 0.0;
    } else {
        *nes_out = es / null_mean;
    }
}

/* Batch: compute ES + NES + p-value for multiple gene sets */
void fgsea_batch(
    const double *values, int n,
    const int *hit_masks,       /* (n_sets, n) flattened */
    int n_sets,
    double exponent,
    int n_bins,
    double *es_out,             /* (n_sets,) */
    double *nes_out,            /* (n_sets,) */
    double *pval_out            /* (n_sets,) */
) {
    for (int s = 0; s < n_sets; s++) {
        const int *mask = hit_masks + s * n;
        fgsea_exact(values, mask, n, exponent, n_bins,
                    &es_out[s], &nes_out[s], &pval_out[s]);
    }
}
