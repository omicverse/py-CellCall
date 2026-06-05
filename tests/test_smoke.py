"""Smoke tests - basic import and instantiation checks."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import():
    """cellcall module imports without error."""
    import cellcall
    assert hasattr(cellcall, "__version__")


def test_cellinter_class():
    """CellInter can be instantiated."""
    from cellcall.object import CellInter
    obj = CellInter()
    assert obj.project == "Microenvironment"
    assert obj.org == "Homo sapiens"
    assert isinstance(obj.data, dict)
    assert isinstance(obj.meta_data, pd.DataFrame)


def test_counts2normalized_10x():
    """CPM normalization produces correct values."""
    from cellcall.normalize import counts2normalized_10x
    data = pd.DataFrame(
        {"A": [10, 20, 30], "B": [40, 50, 60]},
        index=["g1", "g2", "g3"],
    )
    cpm = counts2normalized_10x(data, to_type="CPM", scale_factor=1e6)
    # Column sums should be 1e6
    assert abs(cpm.sum(axis=0).iloc[0] - 1e6) < 1
    assert abs(cpm.sum(axis=0).iloc[1] - 1e6) < 1


def test_get_correlated_genes():
    """Correlation filtering returns list."""
    from cellcall.correlation import get_correlated_genes
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.rand(10, 20),
        index=[f"gene{i}" for i in range(10)],
        columns=["ctA"] * 10 + ["ctB"] * 10,
    )
    result = get_correlated_genes(
        data, cell_type="ctA", tf="gene0",
        target_list=[f"gene{i}" for i in range(1, 10)],
    )
    assert isinstance(result, list)


def test_get_distance_kegg():
    """Distance computation returns DataFrame."""
    from cellcall.distance import get_distance_kegg
    triple = pd.DataFrame({
        "col1": ["a", "b"],
        "col2": ["c", "d"],
        "Ligand_Symbol": ["L1", "L1"],
        "Receptor_Symbol": ["R1", "R1"],
        "Ligand_Symbol2": ["L1", "L1"],
        "Receptor_Symbol2": ["R1", "R1"],
        "TF_Symbol": ["TF1", "TF2"],
        "pathway_info": ["p1_3", "p2_5"],
    })
    dist = get_distance_kegg(triple)
    assert isinstance(dist, pd.DataFrame)
    assert dist.shape[0] > 0


def test_mylog2foldchange():
    """Fold change calculation works."""
    from cellcall.foldchange import mylog2foldchange
    data = pd.DataFrame(
        np.random.rand(5, 10) + 0.1,
        index=[f"gene{i}" for i in range(5)],
        columns=["A"] * 5 + ["B"] * 5,
    )
    result = mylog2foldchange(data, ["A", "B"])
    assert "A" in result
    assert "B" in result
    assert "gene_id" in result["A"].columns
    assert "log2fc" in result["A"].columns


def test_trans2triple_score():
    """Triple score transformation works."""
    from cellcall.triple_score import trans2triple_score
    df = pd.DataFrame({
        "Ligand": ["L1", "L1"],
        "Receptor": ["R1", "R1"],
        "TF": ["TF1", "TF2"],
        "weight1": [0.5, 0.3],
        "weight2": [0.1, 0.2],
    })
    result = trans2triple_score(df)
    assert "value" in result.columns
    assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
