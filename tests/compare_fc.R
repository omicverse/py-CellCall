#!/usr/bin/env Rscript
# Compare fold change values between R and Python

r_dir <- normalizePath(file.path("..", "cellcall-master", "R"))
extdata_dir <- normalizePath(file.path("..", "cellcall-master", "inst", "extdata"))

for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) NULL)
}

suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))

original_system.file <- system.file
system.file <- function(..., package = NULL) {
  args <- list(...)
  if (!is.null(package) && package == "cellcall" && length(args) >= 2) {
    return(file.path(extdata_dir, args[[2]]))
  }
  original_system.file(...)
}

load(file.path(extdata_dir, "example_Data.Rdata"))
data_obj <- NULL
for (obj_name in ls()) {
  obj <- get(obj_name)
  if (is.data.frame(obj) && ncol(obj) > 1 && nrow(obj) > 1) {
    data_obj <- as.data.frame(obj)
    break
  }
}

mt <- CreateNichConObject(data = data_obj, min.feature = 0, names.field = 2,
                          names.delim = "_", source = "UMI", Org = "Homo sapiens")

f.tmp <- file.path(extdata_dir, "new_ligand_receptor_TFs.txt")
triple_relation <- read.table(f.tmp, header = TRUE, quote = "", sep = "\t", stringsAsFactors = FALSE)
triple_relation$pathway_ID <- NULL

f.tmp <- file.path(extdata_dir, "tf_target.txt")
target_relation <- read.table(f.tmp, header = TRUE, quote = "", sep = "\t", stringsAsFactors = FALSE)

my_expr <- mt@data$withoutlog
colnames(my_expr) <- as.character(mt@meta.data$celltype)

complex_tmp <- triple_relation$Receptor_Symbol[grep(",", triple_relation$Receptor_Symbol)] %>% unique()
tmp_complex_symbol <- complex_tmp %>% str_split(",") %>% unlist() %>% unique()
all.gene.needed <- unique(as.character(c(triple_relation$Ligand_Symbol, triple_relation$Receptor_Symbol,
                                          triple_relation$TF_Symbol, target_relation$TF_Symbol,
                                          target_relation$Target_Symbol, tmp_complex_symbol)))
detect_gene <- rownames(my_expr)
expr_set <- my_expr[intersect(detect_gene, all.gene.needed),]
detect_gene <- rownames(expr_set)
cell_type <- unique(colnames(expr_set))

if (length(complex_tmp) > 0) {
  complex_matrix <- matrix(ncol = length(colnames(expr_set)))
  complex_matrix <- as.data.frame(complex_matrix)
  colnames(complex_matrix) <- colnames(expr_set)
  myrownames <- c()
  for (i in 1:length(complex_tmp)) {
    i_tmp <- strsplit(complex_tmp[i], ',')
    if (sum(i_tmp[[1]] %in% detect_gene) == length(i_tmp[[1]])) {
      tmp_df <- expr_set[i_tmp[[1]],]
      tmp_mean <- colMeans(tmp_df)
      tmp_index <- unique(unlist(apply(tmp_df, 1, function(x) {which(x == 0)})))
      tmp_mean[tmp_index] <- 0
      complex_matrix <- rbind(complex_matrix, tmp_mean)
      myrownames <- c(myrownames, complex_tmp[i])
    }
  }
  complex_matrix <- complex_matrix[-1,]
  if (nrow(complex_matrix) > 0) {
    rownames(complex_matrix) <- myrownames
    expr_set <- rbind(expr_set, complex_matrix)
  }
}
expr_set <- expr_set[apply(expr_set, 1, function(x) {sum(x != 0)}) > 0,]

output_dir <- file.path(getwd(), "tests", "reference")

# Compute fold changes for each cell type using R's mylog2foldchange
# This is the function used in data2ConnectProfile.R
# R computes: log2(quantile(ct_expr, probs=0.75) / quantile(all_expr, probs=0.75))

for (ct in cell_type) {
  cat("Computing FC for:", ct, "\n")

  ct_cells <- which(colnames(expr_set) == ct)
  other_cells <- which(colnames(expr_set) != ct)

  ct_expr <- expr_set[, ct_cells, drop = FALSE]
  other_expr <- expr_set[, other_cells, drop = FALSE]

  fc <- numeric(nrow(expr_set))
  names(fc) <- rownames(expr_set)

  for (i in 1:nrow(expr_set)) {
    ct_val <- as.numeric(ct_expr[i, ])
    other_val <- as.numeric(other_expr[i, ])

    # R uses quantile(..., probs=0.75) for the comparison
    # In R, quantile with default type=7
    q75_ct <- quantile(ct_val, probs = 0.75, type = 7)
    q75_other <- quantile(other_val, probs = 0.75, type = 7)

    # R's actual formula: log2((q75_ct + 1) / (q75_other + 1))
    fc[i] <- log2((q75_ct + 1) / (q75_other + 1))
    if (is.infinite(fc[i])) fc[i] <- 0
    if (is.na(fc[i])) fc[i] <- 0
  }

  fc_df <- data.frame(gene_id = names(fc), log2fc = fc, stringsAsFactors = FALSE)
  # Sort by log2fc descending (matching Python's sort)
  fc_df <- fc_df[order(fc_df$log2fc, decreasing = TRUE), ]
  write.csv(fc_df, file.path(output_dir, paste0("r_fc_", ct, ".csv")), row.names = FALSE)
  cat(sprintf("  Saved %d genes\n", nrow(fc_df)))
}

cat("\nDone!\n")
