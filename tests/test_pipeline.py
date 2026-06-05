"""Integration tests for the cellcall Python pipeline."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata")
PARQUET_PATH = os.path.join(EXTDATA_DIR, "example_data.parquet")


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline once and cache results."""
    if not os.path.exists(PARQUET_PATH):
        pytest.skip("Parquet test data not available")

    from cellcall.connect_profile import create_nichcon_object, connect_profile
    import gc

    count_matrix = pd.read_parquet(PARQUET_PATH)
    gc.collect()

    obj = create_nichcon_object(
        data=count_matrix, min_feature=0, names_field=2, names_delim="_",
        source="UMI", org="Homo sapiens"
    )
    del count_matrix
    gc.collect()

    result = connect_profile(
        obj, p_value_cor=0.05, cor_value=0.1, top_target_cor=1,
        method="weighted", p_adjust=0.05, use_type="median", probs=0.75,
        org="Homo sapiens", is_core=True, verbose=False,
    )
    return result


class TestPipelineOutputs:
    """Validate pipeline output structure and properties."""

    def test_expr_mean_shape(self, pipeline_result):
        assert pipeline_result["expr_mean"].shape[0] > 0
        assert pipeline_result["expr_mean"].shape[1] == 4  # 4 cell types

    def test_expr_mean_nonnegative(self, pipeline_result):
        assert (pipeline_result["expr_mean"] >= 0).all().all()

    def test_regulons_matrix_shape(self, pipeline_result):
        rm = pipeline_result["regulons_matrix"]
        assert rm.shape[0] > 300  # Should have ~378 TFs
        assert rm.shape[1] == 4

    def test_regulons_nonnegative(self, pipeline_result):
        assert (pipeline_result["regulons_matrix"] >= 0).all().all()

    def test_expr_l_r_shape(self, pipeline_result):
        elr = pipeline_result["expr_l_r"]
        assert elr.shape[0] > 0
        assert elr.shape[1] == 16  # 4x4 cell type pairs

    def test_expr_l_r_log2_scale_range(self, pipeline_result):
        scaled = pipeline_result["expr_l_r_log2_scale"]
        assert scaled.min().min() >= -1e-10
        assert scaled.max().max() <= 1 + 1e-10

    def test_log2_transform(self, pipeline_result):
        raw = pipeline_result["expr_l_r"]
        log2 = pipeline_result["expr_l_r_log2"]
        # log2(x+1) should be non-negative
        assert (log2.fillna(0) >= -1e-10).all().all()
        # For positive values, log2(x+1) should be positive
        positive_mask = raw > 0
        assert (log2[positive_mask].dropna() > 0).all().all()

    def test_softmax_sums_to_one(self, pipeline_result):
        sl = pipeline_result["softmax_ligand"]
        sr = pipeline_result["softmax_receptor"]
        # Each row should sum to ~1
        np.testing.assert_allclose(sl.sum(axis=1).values, 1.0, atol=1e-6)
        np.testing.assert_allclose(sr.sum(axis=1).values, 1.0, atol=1e-6)

    def test_distance_kegg_positive(self, pipeline_result):
        dk = pipeline_result["DistanceKEGG"]
        # Non-zero distances should be positive
        nonzero = dk.values[dk.values != 0]
        assert (nonzero > 0).all()

    def test_cell_types_correct(self, pipeline_result):
        ct = set(pipeline_result["expr_mean"].columns)
        assert ct == {"SPGed", "SPGing", "SSC", "ST"}


class TestNormalization:
    """Test normalization functions."""

    def test_cpm_normalization(self):
        from cellcall.normalize import counts2normalized_10x
        data = pd.DataFrame({"A": [100, 200, 300], "B": [400, 500, 600]},
                            index=["g1", "g2", "g3"])
        cpm = counts2normalized_10x(data, to_type="CPM", scale_factor=1e6)
        # Column sums should be 1e6
        np.testing.assert_allclose(cpm.sum(axis=0).values, [1e6, 1e6], atol=1)
        # Within-column ratios should be preserved
        np.testing.assert_allclose(cpm["A"].values / cpm["A"].iloc[0],
                                   data["A"].values / data["A"].iloc[0], rtol=1e-6)

    def test_cpm_preserves_order(self):
        from cellcall.normalize import counts2normalized_10x
        data = pd.DataFrame({"A": [1, 10, 100]}, index=["g1", "g2", "g3"])
        cpm = counts2normalized_10x(data, to_type="CPM")
        assert cpm.loc["g3", "A"] > cpm.loc["g2", "A"] > cpm.loc["g1", "A"]


class TestGSEA:
    """Test GSEA integration."""

    def test_gsea_import(self):
        """gseapy is available."""
        import gseapy
        assert hasattr(gseapy, 'prerank')


class TestDistance:
    """Test KEGG distance computation."""

    def test_distance_basic(self):
        from cellcall.distance import get_distance_kegg
        triple = pd.DataFrame({
            "Ligand_Symbol": ["L1", "L1"],
            "Receptor_Symbol": ["R1", "R1"],
            "TF_Symbol": ["TF1", "TF2"],
            "pathway_info": ["p1_3,p2_5", "p3_7"],
            "col5": ["L1", "L1"],
            "col6": ["R1", "R1"],
        })
        dist = get_distance_kegg(triple, method="mean")
        assert dist.loc["L1-R1", "TF1"] == 4.0  # mean of 3 and 5
        assert dist.loc["L1-R1", "TF2"] == 7.0


class TestFoldChange:
    """Test fold change calculation."""

    def test_foldchange_basic(self):
        from cellcall.foldchange import mylog2foldchange
        # Column names must match cell type names exactly
        data = pd.DataFrame({
            "A": [100, 50], "A": [110, 45],  # noqa: duplicate key
            "B": [10, 500], "B": [15, 480],  # noqa: duplicate key
        }, index=["up", "down"])
        # Use unique column names that match cell types
        data = pd.DataFrame(
            np.array([[100, 110, 10, 15], [50, 45, 500, 480]], dtype=float),
            index=["up", "down"],
            columns=["A", "A", "B", "B"],
        )
        fc = mylog2foldchange(data, ["A", "B"], method="mean")
        # 'up' should have positive FC (higher in A: mean 105 vs mean 12.5)
        assert fc["A"].set_index("gene_id").loc["up", "log2fc"] > 0
        # 'down' should have negative FC (lower in A: mean 47.5 vs mean 490)
        assert fc["A"].set_index("gene_id").loc["down", "log2fc"] < 0


class TestCorrelation:
    """Test Spearman correlation via the correlation module."""

    def test_get_correlated_genes_basic(self):
        from cellcall.correlation import get_correlated_genes
        np.random.seed(42)
        n = 50
        x = np.arange(n, dtype=float)
        # genes x cells DataFrame
        data = pd.DataFrame(
            np.vstack([x, x * 2 + np.random.randn(n) * 0.1,
                       -x + np.random.randn(n) * 0.1]),
            index=["tf", "target1", "target2"],
            columns=["ct"] * n,
        )
        result = get_correlated_genes(data, "ct", "tf", ["target1", "target2"])
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
