#!/usr/bin/env Rscript
# Generate ALL R visualizations using official cellcall functions

cat("=== R Visualization (Official cellcall functions) ===\n\n")

# Paths
base_dir <- normalizePath(file.path("..", "cellcall-master"))
r_dir <- file.path(base_dir, "R")
extdata_dir <- file.path(base_dir, "inst", "extdata")
output_dir <- normalizePath(file.path(getwd(), "outputs", "r_plots"), mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Source all R functions
for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) cat("  skip:", basename(f), "\n"))
}

# Load libraries
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(psych))
suppressPackageStartupMessages(library(stats))
suppressPackageStartupMessages(library(clusterProfiler))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(pheatmap))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(RColorBrewer))
suppressPackageStartupMessages(library(circlize))
suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(ggrepel))
suppressPackageStartupMessages(library(ggridges))
suppressPackageStartupMessages(library(ggalluvial))
suppressPackageStartupMessages(library(networkD3))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(DOSE))
suppressPackageStartupMessages(library(gridBase))
suppressPackageStartupMessages(library(grid))
cat("Libraries loaded.\n")

# Monkey-patch system.file
system.file <- function(..., package = NULL) {
  args <- list(...)
  if (!is.null(package) && package == "cellcall" && length(args) >= 2) {
    return(file.path(extdata_dir, args[[2]]))
  }
  base::system.file(...)
}

# ========== Load data & run pipeline ==========
cat("Loading data...\n")
load(file.path(extdata_dir, "example_Data.Rdata"))
data_obj <- NULL
for (obj_name in ls()) {
  obj <- get(obj_name)
  if (is.data.frame(obj) && ncol(obj) > 1 && nrow(obj) > 1) {
    data_obj <- as.data.frame(obj)
    break
  }
}

cat("Creating CellInter object...\n")
mt <- CreateNichConObject(
  data = data_obj, min.feature = 0, names.field = 2,
  names.delim = "_", source = "UMI", Org = "Homo sapiens"
)

cat("Running TransCommuProfile...\n")
mt <- TransCommuProfile(
  mt, pValueCor = 0.05, CorValue = 0.1, topTargetCor = 1,
  p.adjust = 0.05, use.type = "median", probs = 0.75,
  method = "weighted", Org = "Homo sapiens", IS_core = TRUE
)
cat("Pipeline done.\n\n")

# ========== 1. viewPheatmap ==========
cat("1. viewPheatmap...\n")
tryCatch({
  png(file.path(output_dir, "01_viewPheatmap.png"), width = 1200, height = 900, res = 150)
  viewPheatmap(mt, slot = "expr_l_r_log2_scale", show_rownames = TRUE,
               show_colnames = TRUE, cluster_rows = TRUE, cluster_cols = FALSE,
               fontsize = 10, main = "L-R Communication Score")
  dev.off()
  cat("  Saved 01_viewPheatmap.png\n")
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 2. getHyperPathway + plotBubble ==========
cat("2. getHyperPathway + plotBubble...\n")
tryCatch({
  cell_types <- unique(mt@meta.data$celltype)
  pathway_list <- list()
  cc_vec <- c()
  for (i in 1:length(cell_types)) {
    for (j in 1:length(cell_types)) {
      cc <- paste(cell_types[i], cell_types[j], sep = "-")
      tryCatch({
        hp <- getHyperPathway(mt@data$expr_l_r_log2_scale, mt,
                              cella_cellb = cc, IS_core = TRUE, Org = "Homo sapiens")
        pathway_list[[length(pathway_list) + 1]] <- hp
        cc_vec <- c(cc_vec, cc)
      }, error = function(e) NULL)
    }
  }
  if (length(pathway_list) > 0) {
    bubble_data <- getForBubble(pathway_list, cc_vec)
    write.csv(bubble_data, file.path(output_dir, "02_bubble_data.csv"), row.names = FALSE)

    png(file.path(output_dir, "02_plotBubble.png"), width = 1500, height = 1000, res = 150)
    p <- plotBubble(bubble_data)
    print(p)
    dev.off()
    cat("  Saved 02_plotBubble.png\n")
  }
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 3. LR2TF + sankey_graph ==========
cat("3. LR2TF + sankey_graph...\n")
tryCatch({
  cell_types <- unique(mt@meta.data$celltype)
  for (i in 1:min(2, length(cell_types))) {
    for (j in 1:min(2, length(cell_types))) {
      if (i != j) {
        sender <- cell_types[i]
        receiver <- cell_types[j]
        tryCatch({
          mt_tmp <- LR2TF(mt, sender_cell = sender, recevier_cell = receiver,
                          slot = "expr_l_r_log2_scale", org = "Homo sapiens")
          sankey_data <- mt_tmp@reductions$sankey
          if (!is.null(sankey_data) && nrow(sankey_data) > 0) {
            triple_score <- trans2tripleScore(sankey_data)

            # Save sankey data
            write.csv(sankey_data,
                      file.path(output_dir, paste0("03_sankey_data_", sender, "_", receiver, ".csv")),
                      row.names = FALSE)

            # Plot sankey
            png(file.path(output_dir, paste0("03_sankey_", sender, "_", receiver, ".png")),
                width = 1200, height = 700, res = 150)
            p <- sankey_graph(triple_score, axes = 1:3, isGrandSon = TRUE, font.size = 4)
            print(p)
            dev.off()
            cat(paste0("  Saved 03_sankey_", sender, "_", receiver, ".png\n"))
          }
        }, error = function(e) cat(paste0("  ", sender, "->", receiver, ": ", conditionMessage(e), "\n")))
      }
    }
  }
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 4. ViewInterCircos ==========
cat("4. ViewInterCircos...\n")
tryCatch({
  # gridBase must be loaded for gridOMI
  require(gridBase)
  require(grid)
  png(file.path(output_dir, "04_ViewInterCircos.png"), width = 1200, height = 1200, res = 150)
  plot.new()
  ViewInterCircos(mt, slot = "expr_l_r_log2_scale", font = 2,
                  DIY = TRUE, linkcolor.from.sender = TRUE,
                  gap.degree = 10, trackhight1 = 0.05, trackhight2 = 0.032)
  dev.off()
  cat("  Saved 04_ViewInterCircos.png\n")
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 5. getGSEAplot ==========
cat("5. getGSEAplot...\n")
tryCatch({
  gsea_list <- mt@data$gsea.list
  fc_list <- mt@data$fc.list
  if (!is.null(gsea_list) && !is.null(fc_list)) {
    for (ct in names(gsea_list)) {
      gsea_obj <- gsea_list[[ct]]
      if (length(gsea_obj) > 0 && is(gsea_obj, "gseaResult")) {
        top_tfs <- head(gsea_obj@result$ID, 3)
        if (length(top_tfs) > 0) {
          png(file.path(output_dir, paste0("05_getGSEAplot_", ct, ".png")),
              width = 1500, height = 500 * length(top_tfs), res = 150)
          p <- getGSEAplot(gsea_list, myCelltype = ct, fc.list = fc_list,
                           geneSetID = top_tfs)
          print(p)
          dev.off()
          cat(paste0("  Saved 05_getGSEAplot_", ct, ".png\n"))
        }
      }
    }
  }
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 6. ridgeplot.DIY ==========
cat("6. ridgeplot.DIY...\n")
tryCatch({
  gsea_list <- mt@data$gsea.list
  if (!is.null(gsea_list)) {
    for (ct in names(gsea_list)) {
      gsea_obj <- gsea_list[[ct]]
      if (length(gsea_obj) > 0 && is(gsea_obj, "gseaResult") && nrow(gsea_obj@result) > 0) {
        tryCatch({
          png(file.path(output_dir, paste0("06_ridgeplot_", ct, ".png")),
              width = 1200, height = 900, res = 150)
          ridgeplot.DIY(gsea_obj, showCategory = min(10, nrow(gsea_obj@result)))
          dev.off()
          cat(paste0("  Saved 06_ridgeplot_", ct, ".png\n"))
        }, error = function(e) {
          dev.off()
          cat(paste0("  ridgeplot ", ct, " skipped:", conditionMessage(e), "\n"))
        })
      }
    }
  }
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 7. LRT.Dimplot (3D sankey) ==========
cat("7. LRT.Dimplot...\n")
tryCatch({
  cell_types <- unique(mt@meta.data$celltype)
  for (i in 1:min(2, length(cell_types))) {
    for (j in 1:min(2, length(cell_types))) {
      if (i != j) {
        sender <- cell_types[i]
        receiver <- cell_types[j]
        tryCatch({
          mt_tmp <- LR2TF(mt, sender_cell = sender, recevier_cell = receiver,
                          slot = "expr_l_r_log2_scale", org = "Homo sapiens")
          sankey_data <- mt_tmp@reductions$sankey
          if (!is.null(sankey_data) && nrow(sankey_data) > 0) {
            html_path <- file.path(output_dir, paste0("07_LRT_Dimplot_", sender, "_", receiver, ".html"))
            LRT.Dimplot(mt_tmp, fontSize = 12, nodeWidth = 30, width = 1200)
            cat(paste0("  Saved 07_LRT_Dimplot_", sender, "_", receiver, ".html\n"))
          }
        }, error = function(e) cat(paste0("  LRT.Dimplot ", sender, "->", receiver, ": ", conditionMessage(e), "\n")))
      }
    }
  }
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

# ========== 7. Save all data matrices ==========
cat("7. Saving data matrices...\n")
tryCatch({
  for (slot_name in c("expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2",
                       "expr_l_r_log2_scale", "softmax_ligand", "softmax_receptor",
                       "expr_r_regulons", "DistanceKEGG")) {
    if (slot_name %in% names(mt@data)) {
      write.csv(mt@data[[slot_name]], file.path(output_dir, paste0("data_", slot_name, ".csv")))
    }
  }
  write.csv(mt@meta.data, file.path(output_dir, "data_meta_data.csv"), row.names = FALSE)
  cat("  Saved all data matrices\n")
}, error = function(e) cat("  Error:", conditionMessage(e), "\n"))

cat("\n=== R Visualization Complete ===\n")
cat("Output:", output_dir, "\n")
cat("Files:\n")
for (f in list.files(output_dir)) {
  cat("  ", f, "\n")
}
