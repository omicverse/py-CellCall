#!/usr/bin/env Rscript
# R benchmark for cellcall - outputs timing and reference data for comparison

cat("=== cellcall R Benchmark ===\n\n")

# Timing helper
time_it <- function(expr) {
  gc(verbose = FALSE)
  t0 <- proc.time()
  result <- eval(expr)
  t1 <- proc.time()
  elapsed <- (t1 - t0)["elapsed"]
  list(result = result, time = elapsed)
}

# Load cellcall
cat("Loading cellcall...\n")
suppressPackageStartupMessages(library(cellcall))

# Load example data
cat("Loading example data...\n")
extdata_dir <- system.file("extdata", package = "cellcall")
load(file.path(extdata_dir, "example_Data.Rdata"))

# Find the count matrix
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

# === Step 0: CreateNichConObject ===
cat("\n--- Step 0: CreateNichConObject ---\n")
t0 <- CreateNichConObject(
  data = data_obj,
  min.feature = 0,
  names.field = 2,
  names.delim = "_",
  source = "UMI",
  scale.factor = 10^6,
  Org = "Homo sapiens"
)
cat(sprintf("Cell types: %s\n", paste(unique(t0@meta.data$celltype), collapse = ", ")))

# === Step 1: TransCommuProfile ===
cat("\n--- Step 1: TransCommuProfile (main pipeline) ---\n")
gc(verbose = FALSE)
t_start <- proc.time()

mt <- TransCommuProfile(
  t0,
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

t_end <- proc.time()
r_time <- (t_end - t_start)["elapsed"]
cat(sprintf("\nR TransCommuProfile time: %.2f seconds\n", r_time))

# === Save outputs ===
cat("\n--- Saving R reference outputs ---\n")
output_dir <- file.path(getwd(), "tests", "reference")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# expr_mean
write.csv(mt@data$expr_mean, file.path(output_dir, "r_expr_mean.csv"))
cat(sprintf("expr_mean: %d x %d\n", nrow(mt@data$expr_mean), ncol(mt@data$expr_mean)))

# regulons_matrix
write.csv(mt@data$regulons_matrix, file.path(output_dir, "r_regulons_matrix.csv"))
cat(sprintf("regulons_matrix: %d x %d, nonzero: %d\n",
            nrow(mt@data$regulons_matrix), ncol(mt@data$regulons_matrix),
            sum(mt@data$regulons_matrix != 0)))

# expr_l_r_log2_scale
write.csv(mt@data$expr_l_r_log2_scale, file.path(output_dir, "r_expr_l_r_log2_scale.csv"))
cat(sprintf("expr_l_r_log2_scale: %d x %d\n",
            nrow(mt@data$expr_l_r_log2_scale), ncol(mt@data$expr_l_r_log2_scale)))

# expr_l_r
write.csv(mt@data$expr_l_r, file.path(output_dir, "r_expr_l_r.csv"))
write.csv(mt@data$expr_l_r_log2, file.path(output_dir, "r_expr_l_r_log2.csv"))

# expr_r_regulons
write.csv(mt@data$expr_r_regulons, file.path(output_dir, "r_expr_r_regulons.csv"))

# softmax
write.csv(mt@data$softmax_ligand, file.path(output_dir, "r_softmax_ligand.csv"))
write.csv(mt@data$softmax_receptor, file.path(output_dir, "r_softmax_receptor.csv"))

# DistanceKEGG
write.csv(mt@data$DistanceKEGG, file.path(output_dir, "r_DistanceKEGG.csv"))

# Save timing
writeLines(as.character(r_time), file.path(output_dir, "r_timing.txt"))

cat(sprintf("\nR total time: %.2f seconds\n", r_time))
cat("=== R Benchmark Complete ===\n")
