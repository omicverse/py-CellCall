"""Shared fixtures for cellcall tests."""

import os
import sys
import pytest
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXTDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata")
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")


@pytest.fixture
def extdata_dir():
    """Path to extdata directory."""
    return EXTDATA_DIR


@pytest.fixture
def reference_dir():
    """Path to reference outputs directory."""
    return REFERENCE_DIR


@pytest.fixture
def example_data(extdata_dir):
    """Load example_Data.Rdata fixture."""
    from cellcall.io import load_example_data
    return load_example_data(extdata_dir)


@pytest.fixture
def triple_relation(extdata_dir):
    """Load ligand-receptor-TF triple relation data."""
    path = os.path.join(extdata_dir, "new_ligand_receptor_TFs.txt")
    return pd.read_csv(path, sep="\t", header=0)


@pytest.fixture
def target_relation(extdata_dir):
    """Load TF-target relation data."""
    path = os.path.join(extdata_dir, "tf_target.txt")
    return pd.read_csv(path, sep="\t", header=0)
