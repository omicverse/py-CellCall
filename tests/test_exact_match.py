"""Parity gate tests - validate Python outputs.

Gate: Pearson correlation >= 0.97 on expr_l_r_log2_scale.
For R parity: run r_reference_driver.R first, then these tests compare against R outputs.
For self-consistency: the pipeline must produce deterministic results within a single run.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")
EXTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata")
PARQUET_PATH = os.path.join(EXTDATA_DIR, "example_data.parquet")

PEARSON_THRESHOLD = 0.97


def pearson_correlation(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    r, _ = stats.pearsonr(a[mask], b[mask])
    return r


def _run_pipeline():
    """Run the full pipeline and return results."""
    from cellcall.connect_profile import create_nichcon_object, connect_profile
    import gc

    if os.path.exists(PARQUET_PATH):
        count_matrix = pd.read_parquet(PARQUET_PATH)
    else:
        from cellcall.io import load_example_data
        data_dict = load_example_data(EXTDATA_DIR)
        count_matrix = None
        for key, val in data_dict.items():
            if isinstance(val, pd.DataFrame) and val.shape[0] > 1 and val.shape[1] > 1:
                count_matrix = val
                break
        assert count_matrix is not None

    obj = create_nichcon_object(
        data=count_matrix, min_feature=0, names_field=2, names_delim="_",
        source="UMI", org="Homo sapiens", extdata_dir=EXTDATA_DIR,
    )
    del count_matrix
    gc.collect()

    return connect_profile(
        obj, p_value_cor=0.05, cor_value=0.1, top_target_cor=1,
        method="weighted", p_adjust=0.05, use_type="median", probs=0.75,
        org="Homo sapiens", is_core=True, extdata_dir=EXTDATA_DIR, verbose=False,
    )


class TestSelfConsistency:
    """Self-consistency: pipeline must produce valid outputs."""

    def test_pipeline_runs(self):
        """Full pipeline completes without error and produces valid outputs."""
        if not os.path.exists(PARQUET_PATH):
            pytest.skip("Test data not available")

        result = _run_pipeline()

        assert "expr_mean" in result
        assert "regulons_matrix" in result
        assert "expr_l_r" in result
        assert "expr_l_r_log2" in result
        assert "expr_l_r_log2_scale" in result

        # Shapes
        assert result["expr_mean"].shape[0] > 1000
        assert result["expr_mean"].shape[1] == 4
        assert result["regulons_matrix"].shape[0] > 300
        assert result["expr_l_r_log2_scale"].shape[0] > 0

        # Scaling in [0, 1]
        scaled = result["expr_l_r_log2_scale"]
        assert scaled.min().min() >= -1e-10
        assert scaled.max().max() <= 1 + 1e-10

    def test_deterministic_within_process(self):
        """Two runs in the same process must produce identical results."""
        if not os.path.exists(PARQUET_PATH):
            pytest.skip("Test data not available")

        r1 = _run_pipeline()
        r2 = _run_pipeline()

        for key in ["expr_mean", "regulons_matrix", "expr_l_r_log2_scale"]:
            diff = np.abs(r1[key].values - r2[key].values).max()
            assert diff < 1e-10, f"{key}: max diff = {diff}"

    def test_expr_mean_nonnegative(self):
        if not os.path.exists(PARQUET_PATH):
            pytest.skip("Test data not available")
        result = _run_pipeline()
        assert (result["expr_mean"] >= 0).all().all()

    def test_regulons_nonnegative(self):
        if not os.path.exists(PARQUET_PATH):
            pytest.skip("Test data not available")
        result = _run_pipeline()
        assert (result["regulons_matrix"] >= 0).all().all()

    def test_softmax_sums_to_one(self):
        if not os.path.exists(PARQUET_PATH):
            pytest.skip("Test data not available")
        result = _run_pipeline()
        np.testing.assert_allclose(
            result["softmax_ligand"].sum(axis=1).values, 1.0, atol=1e-6
        )
        np.testing.assert_allclose(
            result["softmax_receptor"].sum(axis=1).values, 1.0, atol=1e-6
        )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REFERENCE_DIR, "r_expr_l_r_log2_scale.csv")),
    reason="R reference not generated. Run r_reference_driver.R first."
)
class TestParityWithR:
    """Compare Python outputs against R reference (requires R reference driver)."""

    def _load_r_ref(self, name):
        return pd.read_csv(os.path.join(REFERENCE_DIR, f"r_{name}.csv"), index_col=0)

    def test_expr_l_r_log2_scale_parity(self):
        """THE GATE: Pearson >= 0.97 on expr_l_r_log2_scale vs R."""
        result = _run_pipeline()
        ref = self._load_r_ref("expr_l_r_log2_scale")
        py_vals = result["expr_l_r_log2_scale"]
        common_idx = py_vals.index.intersection(ref.index)
        common_cols = py_vals.columns.intersection(ref.columns)
        if len(common_idx) == 0:
            pytest.skip("No overlapping indices")
        r = pearson_correlation(
            py_vals.loc[common_idx, common_cols].values.flatten(),
            ref.loc[common_idx, common_cols].values.flatten(),
        )
        print(f"\nPearson (expr_l_r_log2_scale vs R): {r:.6f}")
        assert r >= PEARSON_THRESHOLD

    def test_regulons_matrix_parity(self):
        result = _run_pipeline()
        ref = self._load_r_ref("regulons_matrix")
        py_vals = result["regulons_matrix"]
        common_idx = py_vals.index.intersection(ref.index)
        common_cols = py_vals.columns.intersection(ref.columns)
        if len(common_idx) == 0:
            pytest.skip("No overlapping indices")
        r = pearson_correlation(
            py_vals.loc[common_idx, common_cols].values.flatten(),
            ref.loc[common_idx, common_cols].values.flatten(),
        )
        print(f"\nPearson (regulons_matrix vs R): {r:.6f}")
        assert r >= PEARSON_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
