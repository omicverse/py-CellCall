"""Python benchmark for cellcall - timing and R correlation comparison."""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata")
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")


def pearson_correlation(a, b):
    """Pearson correlation between two flat arrays."""
    a_flat = np.asarray(a).flatten()
    b_flat = np.asarray(b).flatten()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return 0.0
    r, _ = stats.pearsonr(a_flat[mask], b_flat[mask])
    return r


def spearman_correlation(a, b):
    """Spearman correlation between two flat arrays."""
    a_flat = np.asarray(a).flatten()
    b_flat = np.asarray(b).flatten()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return 0.0
    r, _ = stats.spearmanr(a_flat[mask], b_flat[mask])
    return r


def align_and_flatten(py_df, r_df):
    """Align two DataFrames by index/columns and return flattened arrays."""
    common_idx = py_df.index.intersection(r_df.index)
    common_cols = py_df.columns.intersection(r_df.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return None, None, 0
    py_vals = py_df.loc[common_idx, common_cols].values
    r_vals = r_df.loc[common_idx, common_cols].values
    return py_vals, r_vals, len(common_idx) * len(common_cols)


def main():
    print("=" * 60)
    print("  cellcall Python vs R Benchmark")
    print("=" * 60)

    # Load data
    parquet_path = os.path.join(EXTDATA_DIR, "example_data.parquet")
    if os.path.exists(parquet_path):
        count_matrix = pd.read_parquet(parquet_path)
    else:
        from cellcall.io import load_example_data
        data = load_example_data(EXTDATA_DIR)
        count_matrix = data["in.content"]
        del data

    gc.collect()
    print(f"\nData: {count_matrix.shape[0]} genes x {count_matrix.shape[1]} cells")
    cell_types = set([c.split("_")[1] for c in count_matrix.columns])
    print(f"Cell types: {sorted(cell_types)}")

    # Run Python pipeline
    from cellcall.connect_profile import create_nichcon_object, connect_profile

    print("\n--- Python Pipeline ---")
    gc.collect()
    t0 = time.time()

    obj = create_nichcon_object(
        data=count_matrix, min_feature=0, names_field=2, names_delim="_",
        source="UMI", org="Homo sapiens"
    )
    t_create = time.time() - t0

    del count_matrix
    gc.collect()

    t1 = time.time()
    result = connect_profile(
        obj, p_value_cor=0.05, cor_value=0.1, top_target_cor=1,
        method="weighted", p_adjust=0.05, use_type="median", probs=0.75,
        org="Homo sapiens", is_core=True, verbose=False
    )
    t_pipeline = time.time() - t1
    t_total = time.time() - t0

    print(f"  CreateObject:    {t_create:.2f}s")
    print(f"  ConnectProfile:  {t_pipeline:.2f}s")
    print(f"  Total:           {t_total:.2f}s")
    print(f"  Output shape:    {result['expr_l_r_log2_scale'].shape}")
    print(f"  Non-zero L-R:    {(result['expr_l_r_log2_scale'] > 0).any(axis=1).sum()}")
    print(f"  Non-zero regulons: {(result['regulons_matrix'] > 0).sum().sum()}")

    # Save Python outputs
    for key in ["expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2", "expr_l_r_log2_scale"]:
        result[key].to_csv(os.path.join(REFERENCE_DIR, f"py_{key}.csv"))

    # Compare with R if available
    r_time_path = os.path.join(REFERENCE_DIR, "r_timing.txt")
    if os.path.exists(r_time_path):
        r_time = float(open(r_time_path).read().strip())
        print(f"\n--- R Timing ---")
        print(f"  R TransCommuProfile: {r_time:.2f}s")
        print(f"  Python Total:        {t_total:.2f}s")
        speedup = r_time / t_total if t_total > 0 else 0
        print(f"  Speedup:             {speedup:.1f}x faster")

        print(f"\n--- Correlation: Python vs R ---")
        print(f"{'Output':<30} {'Pearson':>10} {'Spearman':>10} {'Overlap':>10}")
        print("-" * 62)

        all_pass = True
        for name in ["expr_mean", "regulons_matrix", "expr_l_r_log2",
                      "expr_l_r_log2_scale"]:
            py_path = os.path.join(REFERENCE_DIR, f"py_{name}.csv")
            r_path = os.path.join(REFERENCE_DIR, f"r_{name}.csv")
            if not os.path.exists(r_path):
                continue

            py_df = pd.read_csv(py_path, index_col=0)
            r_df = pd.read_csv(r_path, index_col=0)

            py_vals, r_vals, n = align_and_flatten(py_df, r_df)
            if n == 0:
                print(f"{name:<30} {'N/A':>10} {'N/A':>10} {0:>10}")
                continue

            pr = pearson_correlation(py_vals, r_vals)
            sr = spearman_correlation(py_vals, r_vals)

            status = "PASS" if pr >= 0.97 else "FAIL"
            if pr < 0.97:
                all_pass = False

            print(f"{name:<30} {pr:>10.4f} {sr:>10.4f} {n:>10}  {status}")

        print("-" * 62)
        print(f"Gate (Pearson >= 0.97): {'ALL PASS' if all_pass else 'SOME FAILED'}")

    else:
        print(f"\n--- R Reference Not Found ---")
        print("To generate R reference, run:")
        print("  Rscript tests/r_benchmark.R")
        print(f"\nPython outputs saved to {REFERENCE_DIR}/")

        # Self-consistency check: run twice
        print(f"\n--- Self-Consistency Check ---")
        gc.collect()
        obj2 = create_nichcon_object(
            data=pd.read_parquet(parquet_path), min_feature=0,
            names_field=2, names_delim="_", source="UMI", org="Homo sapiens"
        )
        r2 = connect_profile(
            obj2, p_value_cor=0.05, cor_value=0.1, top_target_cor=1,
            method="weighted", p_adjust=0.05, use_type="median", probs=0.75,
            org="Homo sapiens", is_core=True, verbose=False
        )
        for key in ["expr_mean", "regulons_matrix", "expr_l_r_log2_scale"]:
            diff = np.abs(result[key].values - r2[key].values).max()
            print(f"  {key}: max diff = {diff:.2e} {'OK' if diff < 1e-10 else 'WARN'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
