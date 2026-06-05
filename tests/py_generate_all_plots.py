"""Generate ALL Python cellcall visualizations (matching R outputs)."""

import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "py_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pipeline_result():
    """Run the Python pipeline and return results."""
    from cellcall.connect_profile import create_nichcon_object, connect_profile

    parquet_path = os.path.join(os.path.dirname(__file__), "..", "cellcall", "extdata", "example_data.parquet")
    cm = pd.read_parquet(parquet_path)
    obj = create_nichcon_object(data=cm, min_feature=0, names_field=2, names_delim="_",
                                source="UMI", org="Homo sapiens")
    del cm
    gc.collect()

    result = connect_profile(obj, verbose=False, n_perm=500, p_adjust=0.02)
    return result, obj


def plot_heatmap(data, title, filename, figsize=(12, 8)):
    """Generate heatmap matching R's viewPheatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    # Cluster rows
    from scipy.cluster.hierarchy import linkage, leaves_list
    if len(data) > 1:
        Z = linkage(data.values, method="average")
        order = leaves_list(Z)
        data = data.iloc[order]

    sns.heatmap(data, ax=ax, cmap="YlOrRd", xticklabels=True,
                yticklabels=True if len(data) <= 50 else False,
                linewidths=0.1, linecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Cell-Cell Pair")
    ax.set_ylabel("Ligand-Receptor")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_bubble(data, title, filename, figsize=(14, 10)):
    """Generate bubble plot matching R's plotBubble."""
    if data.empty:
        print(f"  Skipped {filename} (empty data)")
        return

    df = data.copy()
    df["logPvalue"] = -np.log10(df["pvalue"].clip(lower=1e-10))
    df.loc[df["logPvalue"] >= 2, "logPvalue"] = 2
    df.loc[(df["logPvalue"] >= 1.3) & (df["logPvalue"] < 2), "logPvalue"] = 1.3
    df.loc[df["logPvalue"] < 1.3, "logPvalue"] = 1

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        df["cc"], df["pathway"],
        c=df["NES"], s=df["logPvalue"] * 80,
        cmap="RdYlBu_r", edgecolors="black", linewidth=0.5,
        alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="NES", shrink=0.6)
    ax.set_xlabel("Cell-Cell Pair", fontsize=10)
    ax.set_ylabel("Pathway", fontsize=10)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_sankey(sankey_data, sender, receiver, filename):
    """Generate sankey-style plot for L-R-TF relationships."""
    if sankey_data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a simple chord/tile diagram
    ligands = sankey_data["Ligand"].unique()
    receptors = sankey_data["Receptor"].unique()
    tfs = sankey_data["TF"].unique()

    y_positions = {}
    y = 0
    for l in ligands:
        y_positions[f"L:{l}"] = y
        y += 1
    for r in receptors:
        y_positions[f"R:{r}"] = y
        y += 1
    for tf in tfs:
        y_positions[f"TF:{tf}"] = y
        y += 1

    # Draw connections
    for _, row in sankey_data.iterrows():
        src = f"L:{row['Ligand']}"
        mid = f"R:{row['Receptor']}"
        dst = f"TF:{row['TF']}"
        w = float(row["weight1"]) * float(row["weight2"])
        if w > 0:
            alpha = min(1.0, w * 5)
            ax.plot([0, 1, 2], [y_positions[src], y_positions[mid], y_positions[dst]],
                    alpha=alpha, linewidth=max(0.5, w * 10), color="steelblue")

    # Labels
    for key, y_pos in y_positions.items():
        ax.text(-0.1, y_pos, key, fontsize=6, ha="right")
        ax.text(2.1, y_pos, "", fontsize=6, ha="left")

    ax.set_xlim(-0.5, 2.5)
    ax.set_title(f"L-R-TF: {sender} -> {receiver}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_communication_network(result, filename, figsize=(10, 10)):
    """Generate communication network plot."""
    lr_scale = result["expr_l_r_log2_scale"]
    if lr_scale.empty:
        return

    # Extract cell-cell pair scores (mean across L-R pairs)
    cc_scores = lr_scale.mean(axis=0)
    cell_types = sorted(set([c.split("-")[0] for c in cc_scores.index]))

    # Build matrix
    n = len(cell_types)
    matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
    for cc, score in cc_scores.items():
        parts = cc.split("-")
        if len(parts) == 2 and parts[0] in cell_types and parts[1] in cell_types:
            matrix.loc[parts[0], parts[1]] = score

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, ax=ax, cmap="Reds", annot=True, fmt=".3f",
                linewidths=1, linecolor="white", square=True)
    ax.set_title("Cell-Cell Communication Score", fontsize=14)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_regulon_heatmap(regulons, filename, figsize=(10, 12)):
    """Generate regulon activity heatmap."""
    # Filter to active TFs
    active = regulons[(regulons > 0).any(axis=1)]
    if active.empty:
        print(f"  Skipped {filename} (no active TFs)")
        return

    # Sort by total activity
    active = active.loc[active.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(active, ax=ax, cmap="YlOrRd", yticklabels=True,
                linewidths=0.1, linecolor="white",
                xticklabels=True)
    ax.set_title("TF Regulon Activity (NES)", fontsize=14)
    ax.set_ylabel("Transcription Factor")
    ax.set_xlabel("Cell Type")
    plt.yticks(fontsize=5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_gsea_enrichment(fc_values, gene_names, hit_genes, tf_name, filename):
    """Generate GSEA enrichment plot for a specific TF."""
    n = len(fc_values)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    hit_mask = np.zeros(n, dtype=bool)
    for g in hit_genes:
        if g in gene_to_idx:
            hit_mask[gene_to_idx[g]] = True

    n_hit = hit_mask.sum()
    n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return

    abs_vals = np.abs(fc_values)
    hit_sum = abs_vals[hit_mask].sum()
    if hit_sum == 0:
        return

    # Running sum
    phit = np.cumsum(hit_mask.astype(float) * abs_vals) / hit_sum
    prehit = np.cumsum((~hit_mask).astype(float)) / n_miss
    running_sum = phit - prehit

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), height_ratios=[2, 1, 1],
                                         sharex=True)

    # Running enrichment score
    ax1.plot(running_sum, color="green", linewidth=1)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_ylabel("Enrichment Score")
    ax1.set_title(f"GSEA: {tf_name}", fontsize=12)

    # Hit positions
    hit_positions = np.where(hit_mask)[0]
    for pos in hit_positions:
        ax2.axvline(x=pos, color="black", linewidth=0.3)
    ax2.set_ylabel("Hits")
    ax2.set_yticks([])

    # Ranked list (fold change)
    ax3.fill_between(range(n), fc_values, 0, where=fc_values > 0, color="red", alpha=0.5)
    ax3.fill_between(range(n), fc_values, 0, where=fc_values < 0, color="blue", alpha=0.5)
    ax3.set_ylabel("Fold Change")
    ax3.set_xlabel("Rank")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def save_data_matrices(result):
    """Save all data matrices as CSV."""
    for key in ["expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2",
                "expr_l_r_log2_scale", "softmax_ligand", "softmax_receptor",
                "expr_r_regulons", "DistanceKEGG"]:
        if key in result:
            filepath = os.path.join(OUTPUT_DIR, f"data_{key}.csv")
            result[key].to_csv(filepath)
    print("  Saved all data matrices")


def main():
    print("=" * 60)
    print("  Python Visualization Generation")
    print("=" * 60)
    print()

    # Run pipeline
    print("Running pipeline...")
    result, obj = load_pipeline_result()
    print(f"Pipeline done. regulons={((result['regulons_matrix']) > 0).sum().sum()}, "
          f"lr_pairs={result['expr_l_r_log2_scale'].shape[0]}")
    print()

    # 1. Heatmap
    print("1. Heatmap")
    lr_scale = result["expr_l_r_log2_scale"]
    plot_heatmap(lr_scale, "L-R Communication Score (log2 scale)", "01_heatmap.png")

    # 2. Regulon heatmap
    print("2. Regulon heatmap")
    plot_regulon_heatmap(result["regulons_matrix"], "02_regulon_heatmap.png")

    # 3. Communication network
    print("3. Communication network")
    plot_communication_network(result, "03_communication_network.png")

    # 4. Sankey (top L-R pairs)
    print("4. Sankey plots")
    lr_scale_nonzero = lr_scale[lr_scale.max(axis=1) > 0]
    if not lr_scale_nonzero.empty:
        top_lr = lr_scale_nonzero.max(axis=1).sort_values(ascending=False).head(3).index
        for lr_key in top_lr:
            parts = lr_key.split("-", 1)
            if len(parts) == 2:
                sender, receiver = parts[0], parts[1]
                # Create simplified sankey data
                sankey_data = pd.DataFrame({
                    "Ligand": [lr_key.split("-")[0]],
                    "Receptor": [lr_key.split("-")[1].split("-")[0] if "-" in lr_key.split("-")[1] else lr_key.split("-")[1]],
                    "TF": ["N/A"],
                    "weight1": [1.0],
                    "weight2": [1.0],
                })
                plot_sankey(sankey_data, sender, receiver,
                           f"04_sankey_{sender}_{receiver}.png")

    # 5. GSEA enrichment plots for top TFs
    print("5. GSEA enrichment plots")
    regulons = result["regulons_matrix"]
    active_tfs = regulons[(regulons > 0).any(axis=1)]
    if not active_tfs.empty:
        top_tfs = active_tfs.max(axis=1).sort_values(ascending=False).head(3).index
        for tf in top_tfs:
            # This is a simplified version - full GSEA plot requires the ranked gene list
            print(f"  (GSEA plot for {tf} requires rerunning GSEA)")

    # 6. Save data matrices
    print("6. Saving data matrices")
    save_data_matrices(result)

    print()
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
