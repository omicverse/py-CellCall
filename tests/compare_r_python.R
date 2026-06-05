#!/usr/bin/env Rscript
# Compare gene sets and GSEA results between R and Python

r_dir <- normalizePath(file.path("..", "cellcall-master", "R"))
extdata_dir <- normalizePath(file.path("..", "cellcall-master", "inst", "extdata"))

for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) NULL)
}

suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(psych))
suppressPackageStartupMessages(library(magrittr))

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
detect_gene <- rownames(expr_set)

output_dir <- file.path(getwd(), "tests", "reference")

# Test TFs: problematic ones
test_tfs <- c("EGR1", "NFKB2", "PML", "RXRA", "E2F4", "PBX1", "SOX9", "FOXO3")

for (ct in cell_type) {
  cat(ct, ":\n")

  ct_expr_tmp <- expr_set[, which(colnames(expr_set) == ct), drop = FALSE]
  n_cells <- ncol(ct_expr_tmp)
  cat(sprintf("  n_cells=%d\n", n_cells))

  for (tf in test_tfs) {
    if (!(tf %in% rownames(ct_expr_tmp))) next
    targets <- target_relation[which(target_relation$TF_Symbol == tf), 2]
    targets <- targets[targets %in% detect_gene]
    if (length(targets) <= 0) next

    # Correlation
    p_tmp <- psych::corr.test(t(ct_expr_tmp[tf,]), t(ct_expr_tmp[targets,]),
                               adjust = "none", use = "pairwise", method = "spearman")
    tmp_pValue <- p_tmp$p
    tmp_corr <- p_tmp$r
    tmp_pValue[is.na(tmp_pValue)] <- 1
    tmp_corr[is.na(tmp_corr)] <- 0

    # Gene set construction
    corr_gene_tmp <- as.data.frame(t(tmp_corr[1, tmp_pValue < 0.05, drop = FALSE]))
    if (nrow(corr_gene_tmp) <= 1) {
      cat(sprintf("  %s: 0 genes (no significant correlations)\n", tf))
      next
    }
    corr_gene_tmp <- corr_gene_tmp[corr_gene_tmp > 0.1, 1, drop = FALSE]
    corr_target_tmp <- rownames(corr_gene_tmp[order(corr_gene_tmp[,1], decreasing = TRUE), 1, drop = FALSE])
    topTarget <- floor(1 * length(corr_target_tmp))
    if (length(corr_target_tmp) > (topTarget + 1)) {
      corr_target_tmp <- corr_target_tmp[2:(topTarget + 1)]
    }

    gene_set <- sort(corr_target_tmp)
    writeLines(gene_set, file.path(output_dir, paste0("r_gs_", tf, "_", ct, ".txt")))
    cat(sprintf("  %s: %d genes\n", tf, length(gene_set)))

    # Also save correlation and p-value stats for comparison
    # Save top 10 correlated genes with their corr and p-values
    top_genes <- head(corr_target_tmp, 10)
    top_corr <- tmp_corr[1, top_genes]
    top_pval <- tmp_pValue[1, top_genes]
    cat(sprintf("    top 5: %s\n", paste(sapply(1:min(5, length(top_genes)), function(i)
      sprintf("%s(c=%.3f,p=%.4f)", top_genes[i], top_corr[i], top_pval[i])), collapse=", ")))
  }
}

cat("\nDone!\n")
