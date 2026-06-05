#!/usr/bin/env Rscript
# Compute ONLY regulons_matrix using R's GSEA, save for Python to use

cat("=== R regulons_matrix computation ===\n\n")

r_dir <- normalizePath(file.path("..", "cellcall-master", "R"))
extdata_dir <- normalizePath(file.path("..", "cellcall-master", "inst", "extdata"))

for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) NULL)
}

suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(psych))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(stats))
suppressPackageStartupMessages(library(clusterProfiler))

original_system.file <- system.file
system.file <- function(..., package = NULL) {
  args <- list(...)
  if (!is.null(package) && package == "cellcall" && length(args) >= 2) {
    return(file.path(extdata_dir, args[[2]]))
  }
  original_system.file(...)
}

# Load data
load(file.path(extdata_dir, "example_Data.Rdata"))
data_obj <- NULL
for (obj_name in ls()) {
  obj <- get(obj_name)
  if (is.data.frame(obj) && ncol(obj) > 1 && nrow(obj) > 1) {
    data_obj <- as.data.frame(obj)
    break
  }
}

cat("Creating object...\n")
mt <- CreateNichConObject(data = data_obj, min.feature = 0, names.field = 2,
                          names.delim = "_", source = "UMI", Org = "Homo sapiens")

# Load LR-TF data
f.tmp <- file.path(extdata_dir, "new_ligand_receptor_TFs.txt")
triple_relation <- read.table(f.tmp, header = TRUE, quote = "", sep = "\t", stringsAsFactors = FALSE)
triple_relation$pathway_ID <- NULL

f.tmp <- file.path(extdata_dir, "tf_target.txt")
target_relation <- read.table(f.tmp, header = TRUE, quote = "", sep = "\t", stringsAsFactors = FALSE)

# Prepare expression
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

# Handle complex receptors
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

# Fold changes
expr.fc <- mt@data$withoutlog[detect_gene,]
colnames(expr.fc) <- colnames(expr_set)

cat("Computing regulons...\n")
tfs_set <- unique(triple_relation$TF_Symbol)
regulons_matrix <- matrix(data = 0, nrow = length(tfs_set), ncol = length(cell_type))
regulons_matrix <- as.data.frame(regulons_matrix)
rownames(regulons_matrix) <- tfs_set
colnames(regulons_matrix) <- cell_type

for (ct in cell_type) {
  cat(ct, ": ")
  # FC for this cell type
  cell_fc_labels <- colnames(expr.fc)
  cell_fc_labels[cell_fc_labels == ct] <- 0
  cell_fc_labels[cell_fc_labels != '0'] <- 1
  sampleIdsCase <- which(cell_fc_labels == 0)
  probeFC <- as.numeric(apply(expr.fc, 1, function(x) {
    a.tmp <- quantile(as.numeric(x[sampleIdsCase]), probs = 0.75, names = FALSE) + 1
    b.tmp <- quantile(as.numeric(x[-sampleIdsCase]), probs = 0.75, names = FALSE) + 1
    log(a.tmp / b.tmp, base = 2)
  }))
  probeFC[is.infinite(probeFC)] <- 0
  probeFC[is.na(probeFC)] <- 0
  fc_df <- data.frame(gene_id = rownames(expr.fc), log2fc = probeFC)

  # Sort FC descending
  geneList.sort <- dplyr::arrange(fc_df, desc(log2fc))
  rownames(geneList.sort) <- geneList.sort[,1]
  gene.name <- geneList.sort[,1]
  geneList.sort$log2fc <- as.numeric(geneList.sort$log2fc)
  geneList <- geneList.sort[,-1]
  names(geneList) <- gene.name

  # Score each TF
  n_active <- 0
  for (tf in tfs_set) {
    if (!(tf %in% detect_gene)) next
    targets <- target_relation[which(target_relation$TF_Symbol == tf), 2]
    targets <- targets[targets %in% detect_gene]
    if (length(targets) <= 0) next

    # Correlation
    ct_expr_tmp <- expr_set[, which(colnames(expr_set) == ct), drop = FALSE]
    if (!(tf %in% rownames(ct_expr_tmp))) next

    p_tmp <- psych::corr.test(t(ct_expr_tmp[tf,]), t(ct_expr_tmp[targets,]),
                               adjust = "none", use = "pairwise", method = "spearman")
    tmp_pValue <- p_tmp$p
    tmp_corr <- p_tmp$r
    tmp_pValue[is.na(tmp_pValue)] <- 1
    tmp_corr[is.na(tmp_corr)] <- 0

    corr_gene_tmp <- as.data.frame(t(tmp_corr[1, tmp_pValue < 0.05, drop = FALSE]))
    if (nrow(corr_gene_tmp) <= 1) next
    corr_gene_tmp <- corr_gene_tmp[corr_gene_tmp > 0.1, 1, drop = FALSE]
    corr_target_tmp <- rownames(corr_gene_tmp[order(corr_gene_tmp[,1], decreasing = TRUE), 1, drop = FALSE])

    topTarget <- floor(1 * length(corr_target_tmp))
    if (length(corr_target_tmp) > (topTarget + 1)) {
      corr_target_tmp <- corr_target_tmp[2:(topTarget + 1)]
    }

    common_targets_tmp <- intersect(corr_target_tmp, targets)
    if (length(common_targets_tmp) < 5) next

    term_gene_list.tmp <- data.frame(term.name = rep(1, length(common_targets_tmp)),
                                      gene = common_targets_tmp)

    tryCatch({
      nes.tmp <- GSEA(geneList = geneList, TERM2GENE = term_gene_list.tmp,
                       minGSSize = 5, exponent = 1, maxGSSize = 500,
                       pvalueCutoff = 1, pAdjustMethod = "BH", verbose = FALSE)
      if (length(nes.tmp@result$NES) > 0 & length(nes.tmp@result$p.adjust) > 0) {
        if (nes.tmp@result$p.adjust < 0.05 & nes.tmp@result$NES > 0) {
          regulons_matrix[tf, ct] <- nes.tmp@result$NES
          n_active <- n_active + 1
        }
      }
    }, error = function(e) {})
  }
  cat(n_active, "\n")
}

# Save
output_dir <- file.path(getwd(), "tests", "reference")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(regulons_matrix, file.path(output_dir, "r_regulons_matrix.csv"))
cat("\nSaved r_regulons_matrix.csv\n")
cat("Nonzero:", sum(regulons_matrix != 0), "\n")

# Also save timing
writeLines(as.character(0), file.path(output_dir, "r_timing.txt"))
cat("Done.\n")
