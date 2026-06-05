# Dependency Reuse Audit: py-cellcall

## R Dependencies -> Python Equivalents

| R Package | Python Equivalent | Reuse Strategy |
|---|---|---|
| stats | scipy, numpy | Native Python |
| dplyr | pandas | Native Python |
| stringr | pandas str methods | Native Python |
| psych | scipy.stats.spearmanr | Native Python |
| clusterProfiler | Custom _fast_gsea_es | Native implementation |
| reshape2 | pandas.melt/pivot | Native Python |
| tidyr | pandas | Native Python |
| pheatmap | seaborn/matplotlib | Visualization only |
| ggplot2 | matplotlib/seaborn | Visualization only |
| circlize | (not ported) | Visualization only |
| ComplexHeatmap | (not ported) | Visualization only |
| networkD3 | (not ported) | Visualization only |
| ggalluvial | (not ported) | Visualization only |
| ggrepel | (not ported) | Visualization only |
| ggridges | (not ported) | Visualization only |
| RColorBrewer | matplotlib colormaps | Visualization only |
| jsonlite | json (stdlib) | Native Python |
| grid/gridBase | matplotlib | Visualization only |
| enrichplot | (not ported) | Visualization only |
| DOSE | (not ported) | Visualization only |

## Existing py- Mirrors

No existing py-cellcall or equivalent was found on GitHub.

## Decision

All core computation uses native Python (numpy, pandas, scipy). No R bridge (rpy2) needed.
