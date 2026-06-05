"""CellInter class - the central data structure for cell-cell communication analysis."""

from __future__ import annotations
import pandas as pd
import numpy as np


class CellInter:
    """Cell-cell communication analysis object.

    Stores normalized expression, mean expression, regulon scores,
    ligand-receptor scores, and downstream pathway results.
    """

    def __init__(
        self,
        data: dict | None = None,
        meta_data: pd.DataFrame | None = None,
        reductions: dict | None = None,
        project: str = "Microenvironment",
        org: str = "Homo sapiens",
        version: str = "1.0",
    ):
        self.data: dict = data if data is not None else {}
        self.meta_data: pd.DataFrame = meta_data if meta_data is not None else pd.DataFrame()
        self.reductions: dict = reductions if reductions is not None else {}
        self.project = project
        self.org = org
        self.version = version

    def __repr__(self) -> str:
        n_cells = len(self.meta_data)
        n_genes = self.data.get("withoutlog", pd.DataFrame()).shape[0] if "withoutlog" in self.data else 0
        return f"CellInter(project={self.project!r}, org={self.org!r}, cells={n_cells}, genes={n_genes})"
