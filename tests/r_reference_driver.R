#!/usr/bin/env Rscript
# R reference driver for cellcall
# Produces reference outputs from example_Data.Rdata for parity validation

library(cellcall)

cat("Loading example data...\n")
extdata_dir <- system.file("extdata", package = "cellcall")
load(file.path(extdata_dir, "example_Data.Rdata"))

cat("Objects in workspace:\n")
print(ls())

# The example_Data.Rdata should contain a count matrix
# Inspect what we have
for (obj_name in ls()) {
  obj <- get(obj_name)
  cat(sprintf("  %s: class=%s, dim=%s\n", obj_name, class(obj),
              paste(dim(obj), collapse = "x")))
}

# Assuming the data is a count matrix named 'data' or similar
# We need to adapt based on what's actually in the Rdata file
# Let's try to create the CellInter object

# Get the count matrix
data_obj <- NULL
for (obj_name in ls()) {
  obj <- get(obj_name)
  if (is.data.frame(obj) || is.matrix(obj)) {
    if (ncol(obj) > 1 && nrow(obj) > 1) {
      data_obj <- as.data.frame(obj)
      cat(sprintf("Using '%s' as count matrix (%d genes x %d cells)\n",
                  obj_name, nrow(data_obj), ncol(data_obj)))
      break
    }
  }
}

if (is.null(data_obj)) {
  stop("No suitable count matrix found in example_Data.Rdata")
}

# Create CellInter object
cat("\n=== Step 0: CreateNichConObject ===\n")
mt <- CreateNichConObject(
  data = data_obj,
  min.feature = 0,
  names.field = 2,
  names.delim = "_",
  source = "UMI",
  scale.factor = 10^6,
  Org = "Homo sapiens"
)

cat(sprintf("CellInter created: %d cells, %d genes\n",
            ncol(mt@data$withoutlog), nrow(mt@data$withoutlog)))
cat(sprintf("Cell types: %s\n", paste(unique(mt@meta.data$celltype), collapse = ", ")))

# Run TransCommuProfile
cat("\n=== Step 1: TransCommuProfile ===\n")
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

# Save reference outputs
cat("\n=== Saving reference outputs ===\n")
output_dir <- file.path(dirname(extdata_dir), "..", "tests", "reference")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save expr_mean
write.csv(mt@data$expr_mean, file.path(output_dir, "expr_mean.csv"))
cat("Saved expr_mean\n")

# Save regulons_matrix
write.csv(mt@data$regulons_matrix, file.path(output_dir, "regulons_matrix.csv"))
cat("Saved regulons_matrix\n")

# Save expr_l_r
write.csv(mt@data$expr_l_r, file.path(output_dir, "expr_l_r.csv"))
cat("Saved expr_l_r\n")

# Save expr_l_r_log2
write.csv(mt@data$expr_l_r_log2, file.path(output_dir, "expr_l_r_log2.csv"))
cat("Saved expr_l_r_log2\n")

# Save expr_l_r_log2_scale (THE KEY OUTPUT for parity)
write.csv(mt@data$expr_l_r_log2_scale, file.path(output_dir, "expr_l_r_log2_scale.csv"))
cat("Saved expr_l_r_log2_scale\n")

# Save expr_r_regulons
write.csv(mt@data$expr_r_regulons, file.path(output_dir, "expr_r_regulons.csv"))
cat("Saved expr_r_regulons\n")

# Save softmax_ligand
write.csv(mt@data$softmax_ligand, file.path(output_dir, "softmax_ligand.csv"))
cat("Saved softmax_ligand\n")

# Save softmax_receptor
write.csv(mt@data$softmax_receptor, file.path(output_dir, "softmax_receptor.csv"))
cat("Saved softmax_receptor\n")

# Save DistanceKEGG
write.csv(mt@data$DistanceKEGG, file.path(output_dir, "DistanceKEGG.csv"))
cat("Saved DistanceKEGG\n")

# Save fold change lists
fc_dir <- file.path(output_dir, "fc_list")
dir.create(fc_dir, recursive = TRUE, showWarnings = FALSE)
for (ct_name in names(mt@data$fc.list)) {
  write.csv(mt@data$fc.list[[ct_name]], file.path(fc_dir, paste0(ct_name, ".csv")), row.names = FALSE)
}
cat("Saved fc_list\n")

# Save meta.data
write.csv(mt@meta.data, file.path(output_dir, "meta_data.csv"), row.names = FALSE)
cat("Saved meta_data\n")

cat("\n=== Reference generation complete ===\n")
cat("Output directory: ", output_dir, "\n")
