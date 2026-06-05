library(clusterProfiler)
library(cellcall)

# Load data
e <- new.env()
load("cellcall/extdata/example_Data.Rdata", envir=e)
example_Data <- get("in.content", envir=e)
rm(e)

# Create object
obj <- CreateNichConObject(data=example_Data, project="test", source="CPM", Org="Homo sapiens")

# Run ConnectProfile with BOTH p.adjust thresholds
for (padj_thresh in c(0.02, 0.05)) {
  cat("\n=== p.adjust threshold:", padj_thresh, "===\n")
  result <- ConnectProfile(obj,
                           pValueCor=0.05,
                           CorValue=0.1,
                           topTargetCor=1,
                           p.adjust=padj_thresh,
                           use.type="median",
                           probs=0.75,
                           Org="Homo sapiens",
                           IS_core=TRUE)

  fname <- paste0("tests/reference/r_regulons_matrix_padj", padj_thresh, ".csv")
  write.csv(result$regulons_matrix, fname)

  cat("Shape:", dim(result$regulons_matrix), "\n")
  cat("Active:", sum(result$regulons_matrix > 0), "\n")
  for (ct in colnames(result$regulons_matrix)) {
    active <- rownames(result$regulons_matrix)[result$regulons_matrix[,ct] > 0]
    cat(ct, ":", length(active), "TFs\n")
  }
}
cat("\nDone!\n")
