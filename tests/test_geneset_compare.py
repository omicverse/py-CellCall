#!/usr/bin/env python
"""Compare gene sets between Python and R for FP/FN TFs."""
import sys, os, warnings, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import rdata, pandas as pd, numpy as np
from cellcall.connect_profile import connect_profile, _load_lr_tf_data
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')
r_ref = pd.read_csv('tests/reference/r_regulons_matrix.csv', index_col=0)

# Monkey-patch to capture gene sets
import cellcall.fgsea as fgsea_mod
captured_gene_sets = {}
original_run = fgsea_mod.run_gsea_per_tf

def capture_run(fc_values, fc_names, gene_sets, **kwargs):
    captured_gene_sets.clear()
    captured_gene_sets.update(gene_sets)
    return original_run(fc_values, fc_names, gene_sets, **kwargs)

fgsea_mod.run_gsea_per_tf = capture_run

result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500, p_adjust=0.02)

# Now compare Python gene sets vs R for SPGed (has both FP EGR1 and FN E2F4)
print("=== Gene sets captured ===")
for tf in ['EGR1', 'E2F4', 'IRF7', 'NFKB2', 'PML', 'RXRA', 'FOXO1', 'HIF1A', 'NFATC3']:
    if tf in captured_gene_sets:
        gs = captured_gene_sets[tf]
        print(f"  {tf}: {len(gs)} genes")
    else:
        print(f"  {tf}: NOT in gene sets")

# Now let's run R's cellcall to get R's gene sets for comparison
import rpy2.robjects as ro
ro.r('library(cellcall)')

# Run R cellcall
ro.r('''
library(cellcall)
data("example_Data")
meta <- data.frame(
    sampleID = colnames(example_Data),
    celltype = sapply(strsplit(colnames(example_Data), "_"), function(x) x[2])
)
rownames(meta) <- meta$sampleID
r_obj <- create_cellchat(data=example_Data, meta=meta, group.by="celltype")
r_obj <- subsetData(r_obj)
r_obj <- detectGenes(r_obj)
r_obj <- computeOverExpressedGenes(r_obj)
r_obj <- computeLR(r_obj, min.cells = 10)
r_obj <- computeTF(r_obj)
''')

# Get R's regulons for EGR1 and E2F4
ro.r('''
r_regulons <- r_obj@results$cellchat_signaling$regulons
# Get EGR1 regulons
egr1_r <- r_regulons[["EGR1"]]
e2f4_r <- r_regulons[["E2F4"]]
cat("R EGR1 regulons:\\n")
print(egr1_r)
cat("\\nR E2F4 regulons:\\n")
print(e2f4_r)
''')

# Compare
print("\n=== Python gene set sizes ===")
for tf in ['EGR1', 'E2F4']:
    if tf in captured_gene_sets:
        gs = set(captured_gene_sets[tf])
        print(f"  {tf}: {len(gs)} genes: {sorted(gs)[:10]}...")

del result, obj; gc.collect()
print("\nDone!")
