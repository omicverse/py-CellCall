#!/usr/bin/env Rscript
# Direct ES computation for EGR1/SPGed to compare with Python

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
detect_gene <- rownames(expr_set)

fc_data <- as.data.frame(my_expr)[detect_gene,]
fc_data <- data.frame(lapply(fc_data, function(x) { x[is.nan(x)] <- 0; x[is.na(x)] <- 0; x }))
fc.list <- mylog2foldChange.diy(inData = fc_data, cell.type = cell_type, method="median", probs=0.75)

# Test EGR1/SPGed specifically
tf <- "EGR1"
ct <- "SPGed"
cat(sprintf("Testing %s/%s\n", tf, ct))

ct_fc <- fc.list[[ct]]
targets <- target_relation[which(target_relation$TF_Symbol == tf), 2]
targets <- targets[targets %in% detect_gene]
cat(sprintf("  Total targets: %d\n", length(targets)))

corGene_tmp <- getCorrelatedGene(data = expr_set, cell_type = ct, tf = tf,
                                  target_list = targets, pValue = 0.05, corValue = 0.1, topGene = 1.0)
common_targets_tmp <- intersect(corGene_tmp, targets)
cat(sprintf("  Correlated targets: %d\n", length(common_targets_tmp)))

if (length(common_targets_tmp) >= 5) {
  # Run GSEA with pvalueCutoff=1
  term_gene <- data.frame(term.name = rep(1, length(common_targets_tmp)), gene = common_targets_tmp)
  geneList.sort <- dplyr::arrange(ct_fc, desc(ct_fc$log2fc))
  rownames(geneList.sort) <- geneList.sort[,1]
  gene.name <- geneList.sort[,1]
  geneList.sort$log2fc <- as.numeric(geneList.sort$log2fc)
  geneList <- geneList.sort[,-1]
  names(geneList) <- gene.name

  cat(sprintf("  Gene list length: %d\n", length(geneList)))
  cat(sprintf("  Gene set genes in list: %d\n", sum(common_targets_tmp %in% names(geneList))))

  # Try GSEA with suppressWarnings
  y <- suppressWarnings(tryCatch({
    GSEA(geneList = geneList, TERM2GENE = term_gene, minGSSize = 5, exponent = 1,
         maxGSSize = 500, pvalueCutoff = 1, pAdjustMethod = "BH", verbose = FALSE)
  }, error = function(e) { cat(sprintf("  GSEA error: %s\n", e$message)); NULL }))

  if (!is.null(y) && nrow(y@result) > 0) {
    r <- y@result[1,]
    cat(sprintf("  R result: NES=%.4f, pvalue=%.6f, p.adjust=%.6f, setSize=%d\n",
                r$NES, r$pvalue, r$p.adjust, r$setSize))
  } else {
    cat("  GSEA returned NULL or empty result\n")

    # Manual ES computation
    cat("  Computing ES manually...\n")
    gene_list_sorted <- geneList.sort[order(geneList.sort$log2fc, decreasing = TRUE),]
    gene_names_sorted <- rownames(gene_list_sorted)
    fc_values <- gene_list_sorted$log2fc

    # Check if gene set genes are in the ranked list
    gs_in_list <- common_targets_tmp[common_targets_tmp %in% gene_names_sorted]
    cat(sprintf("  Gene set genes in ranked list: %d/%d\n", length(gs_in_list), length(common_targets_tmp)))

    if (length(gs_in_list) >= 5) {
      # Compute ES
      n <- length(gene_names_sorted)
      hit_mask <- gene_names_sorted %in% gs_in_list
      n_hit <- sum(hit_mask)
      n_miss <- n - n_hit

      pow_vals <- abs(fc_values)^1
      hit_sum <- sum(hit_mask * pow_vals)

      cum_hit <- cumsum(hit_mask * pow_vals) / hit_sum
      cum_miss <- cumsum(!hit_mask) / n_miss
      rs <- cum_hit - cum_miss

      max_pos <- max(rs)
      max_neg <- min(rs)
      es <- if (abs(max_pos) >= abs(max_neg)) max_pos else max_neg

      cat(sprintf("  Manual ES: %.6f\n", es))
      cat(sprintf("  max_pos: %.6f, max_neg: %.6f\n", max_pos, max_neg))
    }
  }
}

# Also test PBX1/ST as a TP
tf <- "PBX1"
ct <- "ST"
cat(sprintf("\nTesting %s/%s\n", tf, ct))

ct_fc <- fc.list[[ct]]
targets <- target_relation[which(target_relation$TF_Symbol == tf), 2]
targets <- targets[targets %in% detect_gene]
cat(sprintf("  Total targets: %d\n", length(targets)))

corGene_tmp <- getCorrelatedGene(data = expr_set, cell_type = ct, tf = tf,
                                  target_list = targets, pValue = 0.05, corValue = 0.1, topGene = 1.0)
common_targets_tmp <- intersect(corGene_tmp, targets)
cat(sprintf("  Correlated targets: %d\n", length(common_targets_tmp)))

if (length(common_targets_tmp) >= 5) {
  term_gene <- data.frame(term.name = rep(1, length(common_targets_tmp)), gene = common_targets_tmp)
  geneList.sort <- dplyr::arrange(ct_fc, desc(ct_fc$log2fc))
  rownames(geneList.sort) <- geneList.sort[,1]
  gene.name <- geneList.sort[,1]
  geneList.sort$log2fc <- as.numeric(geneList.sort$log2fc)
  geneList <- geneList.sort[,-1]
  names(geneList) <- gene.name

  gene_list_sorted <- geneList.sort[order(geneList.sort$log2fc, decreasing = TRUE),]
  gene_names_sorted <- rownames(gene_list_sorted)
  fc_values <- gene_list_sorted$log2fc

  gs_in_list <- common_targets_tmp[common_targets_tmp %in% gene_names_sorted]
  cat(sprintf("  Gene set genes in ranked list: %d/%d\n", length(gs_in_list), length(common_targets_tmp)))

  if (length(gs_in_list) >= 5) {
    n <- length(gene_names_sorted)
    hit_mask <- gene_names_sorted %in% gs_in_list
    n_hit <- sum(hit_mask)
    n_miss <- n - n_hit

    pow_vals <- abs(fc_values)^1
    hit_sum <- sum(hit_mask * pow_vals)

    cum_hit <- cumsum(hit_mask * pow_vals) / hit_sum
    cum_miss <- cumsum(!hit_mask) / n_miss
    rs <- cum_hit - cum_miss

    max_pos <- max(rs)
    max_neg <- min(rs)
    es <- if (abs(max_pos) >= abs(max_neg)) max_pos else max_neg

    cat(sprintf("  Manual ES: %.6f\n", es))
    cat(sprintf("  max_pos: %.6f, max_neg: %.6f\n", max_pos, max_neg))
  }
}

cat("\nDone!\n")
