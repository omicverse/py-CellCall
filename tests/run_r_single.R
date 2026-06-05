suppressPackageStartupMessages({
  library(clusterProfiler)
  library(cellcall)
})

e <- new.env()
load("cellcall/extdata/example_Data.Rdata", envir=e)
example_Data <- get("in.content", envir=e)
rm(e)

obj <- CreateNichConObject(data=example_Data, project="test", source="CPM", Org="Homo sapiens")

# Run with p.adjust=0.05 (R default)
result <- ConnectProfile(obj,
                         pValueCor=0.05, CorValue=0.1, topTargetCor=1,
                         p.adjust=0.05, use.type="median", probs=0.75,
                         Org="Homo sapiens", IS_core=TRUE)

write.csv(result$regulons_matrix, "tests/reference/r_regulons_matrix_padj005.csv")

cat("padj=0.05:\n")
cat("Active:", sum(result$regulons_matrix > 0), "\n")
for (ct in colnames(result$regulons_matrix)) {
  active <- rownames(result$regulons_matrix)[result$regulons_matrix[,ct] > 0]
  vals <- result$regulons_matrix[active, ct]
  cat(ct, ":", length(active), "TFs\n")
  for (i in seq_along(active)) {
    cat(sprintf("  %s: NES=%.3f\n", active[i], vals[i]))
  }
}

# Also run with p.adjust=0.02
result2 <- ConnectProfile(obj,
                          pValueCor=0.05, CorValue=0.1, topTargetCor=1,
                          p.adjust=0.02, use.type="median", probs=0.75,
                          Org="Homo sapiens", IS_core=TRUE)

write.csv(result2$regulons_matrix, "tests/reference/r_regulons_matrix_padj002.csv")

cat("\npadj=0.02:\n")
cat("Active:", sum(result2$regulons_matrix > 0), "\n")
for (ct in colnames(result2$regulons_matrix)) {
  active <- rownames(result2$regulons_matrix)[result2$regulons_matrix[,ct] > 0]
  cat(ct, ":", length(active), "TFs\n")
}

cat("\nDone!\n")
