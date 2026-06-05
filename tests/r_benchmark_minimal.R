#!/usr/bin/env Rscript
# R benchmark - sources cellcall directly and patches file paths

cat("=== cellcall R Benchmark ===\n\n")

# Paths
base_dir <- normalizePath(file.path("..", "cellcall-master"))
r_dir <- file.path(base_dir, "R")
extdata_dir <- file.path(base_dir, "inst", "extdata")

cat("R source:", r_dir, "\n")
cat("extdata:", extdata_dir, "\n")

# Source all R files
for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) cat("  skip:", basename(f), "\n"))
}

# Load libraries
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(psych))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(stats))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(clusterProfiler))
suppressPackageStartupMessages(library(DOSE))
cat("Libraries loaded.\n")

# Monkey-patch system.file to find extdata
# R code calls: system.file("extdata", "filename.txt", package="cellcall")
# We need to return: extdata_dir/filename.txt
original_system.file <- system.file
system.file <- function(..., package = NULL) {
  args <- list(...)
  if (!is.null(package) && package == "cellcall" && length(args) >= 2) {
    # args[1] is "extdata", args[2] is the filename
    return(file.path(extdata_dir, args[[2]]))
  }
  original_system.file(...)
}

# Load example data
cat("Loading example data...\n")
load(file.path(extdata_dir, "example_Data.Rdata"))

data_obj <- NULL
for (obj_name in ls()) {
  obj <- get(obj_name)
  if (is.data.frame(obj) || is.matrix(obj)) {
    if (ncol(obj) > 1 && nrow(obj) > 1) {
      data_obj <- as.data.frame(obj)
      cat(sprintf("Count matrix: %d genes x %d cells\n", nrow(data_obj), ncol(data_obj)))
      break
    }
  }
}

# CreateNichConObject
cat("\n--- CreateNichConObject ---\n")
gc(verbose = FALSE)
t0 <- proc.time()

mt <- CreateNichConObject(
  data = data_obj,
  min.feature = 0,
  names.field = 2,
  names.delim = "_",
  source = "UMI",
  scale.factor = 10^6,
  Org = "Homo sapiens"
)

t_create <- (proc.time() - t0)["elapsed"]
cat(sprintf("CreateObject: %.2fs\n", t_create))
cat(sprintf("Cell types: %s\n", paste(unique(mt@meta.data$celltype), collapse = ", ")))

# TransCommuProfile
cat("\n--- TransCommuProfile ---\n")
gc(verbose = FALSE)
t1 <- proc.time()

mt <- TransCommuProfile(
  mt,
  pValueCor = 0.05,
  CorValue = 0.1,
  topTargetCor = 1,
  p.adjust = 0.05,
  use.type = "median",
  probs = 0.75,
  method = "weighted",
  Org = "Homo sapiens",
  IS_core = TRUE
)

t_pipeline <- (proc.time() - t1)["elapsed"]
t_total <- t_create + t_pipeline
cat(sprintf("ConnectProfile: %.2fs\n", t_pipeline))
cat(sprintf("Total R time: %.2fs\n", t_total))

# Save outputs
cat("\n--- Saving R outputs ---\n")
output_dir <- file.path(base_dir, "py-cellcall", "tests", "reference")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write.csv(mt@data$expr_mean, file.path(output_dir, "r_expr_mean.csv"))
write.csv(mt@data$regulons_matrix, file.path(output_dir, "r_regulons_matrix.csv"))
write.csv(mt@data$expr_l_r, file.path(output_dir, "r_expr_l_r.csv"))
write.csv(mt@data$expr_l_r_log2, file.path(output_dir, "r_expr_l_r_log2.csv"))
write.csv(mt@data$expr_l_r_log2_scale, file.path(output_dir, "r_expr_l_r_log2_scale.csv"))
write.csv(mt@data$expr_r_regulons, file.path(output_dir, "r_expr_r_regulons.csv"))
write.csv(mt@data$softmax_ligand, file.path(output_dir, "r_softmax_ligand.csv"))
write.csv(mt@data$softmax_receptor, file.path(output_dir, "r_softmax_receptor.csv"))
write.csv(mt@data$DistanceKEGG, file.path(output_dir, "r_DistanceKEGG.csv"))
writeLines(as.character(t_total), file.path(output_dir, "r_timing.txt"))

cat(sprintf("  expr_mean: %d x %d\n", nrow(mt@data$expr_mean), ncol(mt@data$expr_mean)))
cat(sprintf("  regulons_matrix: %d x %d, nonzero: %d\n",
            nrow(mt@data$regulons_matrix), ncol(mt@data$regulons_matrix),
            sum(mt@data$regulons_matrix != 0)))
cat(sprintf("  expr_l_r_log2_scale: %d x %d\n",
            nrow(mt@data$expr_l_r_log2_scale), ncol(mt@data$expr_l_r_log2_scale)))

cat("\n=== R Benchmark Complete ===\n")
