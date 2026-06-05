"""R GSEA backend via subprocess - calls R's GSEA for exact p-values."""

from __future__ import annotations
import os
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd


def _find_rscript():
    import glob
    for pattern in ['C:/Program Files/R/R-*/bin/Rscript.exe',
                    '/usr/bin/Rscript', '/usr/local/bin/Rscript']:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    return None


def compute_regulons_via_r(
    fc_list, cell_types, gene_sets_per_ct,
    p_adjust=0.05, verbose=True,
):
    """Compute regulon NES scores using R's clusterProfiler::GSEA via subprocess."""
    r_home = _find_rscript()
    if r_home is None:
        raise RuntimeError("R not found")

    work_dir = tempfile.mkdtemp(prefix="cellcall_gsea_")
    # Use forward slashes for R compatibility
    work_dir_r = work_dir.replace("\\", "/")

    # Save FC data per cell type
    all_tfs = set()
    for ct in cell_types:
        fc_list[ct].to_csv(os.path.join(work_dir, f"fc_{ct}.csv"), index=False)
        if ct in gene_sets_per_ct:
            all_tfs.update(gene_sets_per_ct[ct].keys())

    # Save gene sets
    for ct in cell_types:
        if ct not in gene_sets_per_ct:
            continue
        ct_dir = os.path.join(work_dir, f"genesets_{ct}")
        os.makedirs(ct_dir, exist_ok=True)
        for tf, genes in gene_sets_per_ct[ct].items():
            with open(os.path.join(ct_dir, f"{tf}.txt"), "w") as f:
                f.write("\n".join(genes))

    ct_vec = ",".join([f'"{ct}"' for ct in cell_types])

    # Build R script - use \\\\n for literal \\n in R, forward slashes for paths
    r_lines = [
        'suppressPackageStartupMessages(library(clusterProfiler))',
        f'work_dir <- "{work_dir_r}"',
        f'p_adjust <- {p_adjust}',
        f'cell_types <- c({ct_vec})',
        '',
        'output <- c()',
        'for (ct in cell_types) {',
        '    fc_path <- file.path(work_dir, paste0("fc_", ct, ".csv"))',
        '    if (!file.exists(fc_path)) next',
        '    fc_df <- read.csv(fc_path)',
        '    geneList <- fc_df$log2fc',
        '    names(geneList) <- fc_df$gene_id',
        '    geneList <- sort(geneList, decreasing=TRUE)',
        '',
        '    gs_dir <- file.path(work_dir, paste0("genesets_", ct))',
        '    if (!dir.exists(gs_dir)) next',
        '    gs_files <- list.files(gs_dir, pattern="*.txt")',
        '    n_active <- 0',
        '',
        '    for (gs_file in gs_files) {',
        '        tf <- sub("\\\\.txt$", "", gs_file)',
        '        genes <- readLines(file.path(gs_dir, gs_file))',
        '        if (length(genes) < 5) next',
        '',
        '        term_gene <- data.frame(term.name=rep(1, length(genes)), gene=genes)',
        '        tryCatch({',
        '            y <- GSEA(geneList=geneList, TERM2GENE=term_gene,',
        '                       minGSSize=5, exponent=1, maxGSSize=500,',
        '                       pvalueCutoff=1, pAdjustMethod="BH", verbose=FALSE)',
        '            if (length(y@result$NES) > 0 && length(y@result$p.adjust) > 0) {',
        '                if (y@result$p.adjust < p_adjust && y@result$NES > 0) {',
        '                    output <- c(output, paste0(ct, "\\t", tf, "\\t", y@result$NES))',
        '                    n_active <- n_active + 1',
        '                }',
        '            }',
        '        }, error=function(e) {})',
        '    }',
        '    cat(ct, ":", n_active, "\\n")',
        '}',
        '',
        'if (length(output) > 0) { writeLines(output, file.path(work_dir, "results.tsv")) }',
        'cat("Done\\n")',
    ]

    r_script = "\n".join(r_lines)
    r_path = os.path.join(work_dir, "run_gsea.R")
    with open(r_path, "w") as f:
        f.write(r_script)

    if verbose:
        print("    Running R GSEA...")

    result = subprocess.run(
        [r_home, r_path],
        capture_output=True, text=True, timeout=1800,
        cwd=work_dir,
    )

    if verbose and result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"    R: {line}")

    if result.returncode != 0 and verbose:
        print(f"    R error: {result.stderr[-200:]}")

    # Parse results
    regulons_matrix = pd.DataFrame(0.0, index=list(all_tfs), columns=cell_types)
    results_path = os.path.join(work_dir, "results.tsv")
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    ct, tf, nes = parts[0], parts[1], float(parts[2])
                    if tf in regulons_matrix.index:
                        regulons_matrix.loc[tf, ct] = nes

    shutil.rmtree(work_dir, ignore_errors=True)
    return regulons_matrix
