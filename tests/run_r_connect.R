library(cellcall)

# Load data from file
e <- new.env()
load("cellcall/extdata/example_Data.Rdata", envir=e)
example_Data <- get("in.content", envir=e)
rm(e)

cat("Raw data:", class(example_Data), "dims:", dim(example_Data), "\n")

# Create CellInter object
obj <- CreateNichConObject(data=example_Data,
                           project="test",
                           source="CPM",
                           Org="Homo sapiens")

cat("Object created:", class(obj), "\n")
cat("Cell types:", unique(obj@meta.data$celltype), "\n")

# Run ConnectProfile with default parameters
result <- ConnectProfile(obj,
                         pValueCor=0.05,
                         CorValue=0.1,
                         topTargetCor=1,
                         p.adjust=0.05,
                         use.type="median",
                         probs=0.75,
                         Org="Homo sapiens",
                         IS_core=TRUE)

# Save regulons_matrix
write.csv(result$regulons_matrix, "tests/reference/r_regulons_matrix_default.csv")
cat("\nDone! regulons_matrix saved.\n")
cat("Shape:", dim(result$regulons_matrix), "\n")
cat("Active entries:", sum(result$regulons_matrix > 0), "\n\n")
cat("Per cell type:\n")
for (ct in colnames(result$regulons_matrix)) {
  active <- rownames(result$regulons_matrix)[result$regulons_matrix[,ct] > 0]
  cat(ct, ":", length(active), "TFs -", paste(active, collapse=", "), "\n")
}
