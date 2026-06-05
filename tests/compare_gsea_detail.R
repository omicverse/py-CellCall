#!/usr/bin/env Rscript
# Get detailed GSEA results for specific TFs

r_dir <- normalizePath(file.path("..", "cellcall-master", "R"))
extdata_dir <- normalizePath(file.path("..", "cellcall-master", "inst", "extdata"))

for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  tryCatch(source(f), error = function(e) NULL)
}

suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(clusterProfiler))

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

# Compute FC like R
fc_data <- as.data.frame(my_expr)[detect_gene,]
fc_data <- data.frame(lapply(fc_data, function(x) { x[is.nan(x)] <- 0; x[is.na(x)] <- 0; x }))
fc.list <- mylog2foldChange.diy(inData = fc_data, cell.type = cell_type, method="median", probs=0.75)

# Test TFs: FPs and key TPs
test_tfs <- c("EGR1", "NFKB2", "PML", "RXRA", "E2F4", "IRF7", "PBX1", "SOX9", "FOXO3", "STAT2")

for (ct in cell_type) {
  cat(ct, ":\n")
  ct_fc <- fc.list[[ct]]

  for (tf in test_tfs) {
    if (!(tf %in% detect_gene)) next
    targets <- target_relation[which(target_relation$TF_Symbol == tf), 2]
    targets <- targets[targets %in% detect_gene]
    if (length(targets) <= 0) next

    # Get correlated targets
    corGene_tmp <- getCorrelatedGene(data = expr_set, cell_type = ct, tf = tf,
                                      target_list = targets, pValue = 0.05, corValue = 0.1, topGene = 1.0)
    common_targets_tmp <- intersect(corGene_tmp, targets)
    if (length(common_targets_tmp) < 5) next

    # Run GSEA with pvalueCutoff=1 to get all results
    term_gene <- data.frame(term.name = rep(1, length(common_targets_tmp)), gene = common_targets_tmp)
    geneList.sort <- dplyr::arrange(ct_fc, desc(ct_fc$log2fc))
    rownames(geneList.sort) <- geneList.sort[,1]
    gene.name <- geneList.sort[,1]
    geneList.sort$log2fc <- as.numeric(geneList.sort$log2fc)
    geneList <- geneList.sort[,-1]
    names(geneList) <- gene.name

    y <- tryCatch({
      GSEA(geneList = geneList, TERM2GENE = term_gene, minGSSize = 5, exponent = 1,
           maxGSSize = 500, pvalueCutoff = 1, pAdjustMethod = "BH", verbose = FALSE)
    }, error = function(e) NULL)

    if (!is.null(y) && nrow(y@result) > 0) {
      r <- y@result[1,]
      cat(sprintf("  %s: NES=%.4f, pvalue=%.6f, p.adjust=%.6f, qvalue=%.6f, gs_size=%d\n",
                  tf, r$NES, r$pvalue, r$p.adjust, r$qvalue, r$setSize))
    }
  }
}

cat("\nDone!\n")
