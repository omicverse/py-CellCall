"""Generate R-style plots using R's actual output data."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

warnings.filterwarnings("ignore")

R_REF = os.path.join(os.path.dirname(__file__), "reference")
R_PLOTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "r_plots")
PY_PLOTS = os.path.join(os.path.dirname(__file__), "..", "outputs", "py_plots")
os.makedirs(R_PLOTS, exist_ok=True)
os.makedirs(PY_PLOTS, exist_ok=True)


def load_r_data():
    """Load all R reference data."""
    data = {}
    for name in ["expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2",
                  "expr_l_r_log2_scale", "expr_r_regulons", "softmax_ligand",
                  "softmax_receptor", "DistanceKEGG"]:
        path = os.path.join(R_REF, f"r_{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path, index_col=0)
    return data


def load_py_data():
    """Load Python pipeline outputs."""
    data = {}
    for name in ["expr_mean", "regulons_matrix", "expr_l_r", "expr_l_r_log2",
                  "expr_l_r_log2_scale", "expr_r_regulons", "softmax_ligand",
                 "softmax_receptor", "DistanceKEGG"]:
        path = os.path.join(PY_PLOTS, f"data_{name}.csv")
        if os.path.exists(path):
            data[name] = pd.read_csv(path, index_col=0)
    return data


def generate_pheatmap(data, title, filename, output_dir):
    """Generate pheatmap-style heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Cluster rows
    if len(data) > 2:
        Z = linkage(data.values, method="average")
        order = leaves_list(Z)
        data = data.iloc[order]

    sns.heatmap(data, ax=ax, cmap="YlOrRd",
                xticklabels=True,
                yticklabels=len(data) <= 80,
                linewidths=0.1, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Cell-Cell Pair", fontsize=10)
    ax.set_ylabel("Ligand-Receptor", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    if len(data) <= 80:
        plt.yticks(fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_bubble_plot(data, title, filename, output_dir):
    """Generate bubble plot matching R's plotBubble."""
    if data.empty:
        print(f"  Skipped {filename} (empty)")
        return

    df = data.copy()
    if "pvalue" in df.columns:
        df["logPvalue"] = -np.log10(df["pvalue"].clip(lower=1e-10))
    else:
        df["logPvalue"] = 1.5

    df.loc[df["logPvalue"] >= 2, "logPvalue"] = 2
    df.loc[(df["logPvalue"] >= 1.3) & (df["logPvalue"] < 2), "logPvalue"] = 1.3
    df.loc[df["logPvalue"] < 1.3, "logPvalue"] = 1

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        df["cc"], df["pathway"],
        c=df["NES"], s=df["logPvalue"] * 80,
        cmap="RdYlBu_r", edgecolors="black", linewidth=0.5, alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="NES", shrink=0.6)
    ax.set_xlabel("Cell-Cell Pair", fontsize=10)
    ax.set_ylabel("Pathway", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_regulon_heatmap(regulons, title, filename, output_dir):
    """Generate regulon activity heatmap."""
    active = regulons[(regulons > 0).any(axis=1)]
    if active.empty:
        print(f"  Skipped {filename}")
        return

    active = active.loc[active.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, max(6, len(active) * 0.25)))
    sns.heatmap(active, ax=ax, cmap="YlOrRd",
                yticklabels=True, linewidths=0.1, linecolor="white",
                xticklabels=True, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Transcription Factor", fontsize=10)
    ax.set_xlabel("Cell Type", fontsize=10)
    plt.yticks(fontsize=max(3, min(8, 200 // len(active))))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_communication_network(lr_scale, title, filename, output_dir):
    """Generate cell-cell communication network heatmap."""
    cc_scores = lr_scale.mean(axis=0)
    cell_types = sorted(set([c.split("-")[0] for c in cc_scores.index]))

    matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
    for cc, score in cc_scores.items():
        parts = cc.split("-")
        if len(parts) == 2 and parts[0] in cell_types and parts[1] in cell_types:
            matrix.loc[parts[0], parts[1]] = score

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(matrix, ax=ax, cmap="Reds", annot=True, fmt=".3f",
                linewidths=1, linecolor="white", square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Receiver", fontsize=10)
    ax.set_ylabel("Sender", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_comparison_plot(r_data, py_data, metric_name, title, filename, output_dir):
    """Generate side-by-side comparison plot."""
    from scipy import stats as sp_stats

    if metric_name not in r_data or metric_name not in py_data:
        return

    r_df = r_data[metric_name]
    py_df = py_data[metric_name]

    # Align
    ci = r_df.index.intersection(py_df.index)
    cc = r_df.columns.intersection(py_df.columns)
    if len(ci) == 0 or len(cc) == 0:
        return

    r_vals = r_df.loc[ci, cc].values.flatten()
    py_vals = py_df.loc[ci, cc].values.flatten()

    mask = np.isfinite(r_vals) & np.isfinite(py_vals)
    r_vals = r_vals[mask]
    py_vals = py_vals[mask]

    pr = sp_stats.pearsonr(r_vals, py_vals)[0] if len(r_vals) > 1 else 0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(r_vals, py_vals, alpha=0.3, s=10, c="steelblue")
    lims = [min(r_vals.min(), py_vals.min()), max(r_vals.max(), py_vals.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
    ax.set_xlabel("R", fontsize=12)
    ax.set_ylabel("Python", fontsize=12)
    ax.set_title(f"{title}\nPearson={pr:.4f}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename} (Pearson={pr:.4f})")


def main():
    from scipy import stats

    print("=" * 60)
    print("  Generating R-style & comparison plots")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    r_data = load_r_data()
    print(f"  R data: {list(r_data.keys())}")

    # Check if py data exists
    py_data = load_py_data()
    print(f"  Py data: {list(py_data.keys())}")

    # Generate R-style plots
    print("\n--- R-style plots (outputs/r_plots/) ---")

    if "expr_l_r_log2_scale" in r_data:
        generate_pheatmap(
            r_data["expr_l_r_log2_scale"],
            "L-R Communication Score (R, log2 scale)",
            "01_pheatmap_R.png", R_PLOTS
        )

    if "regulons_matrix" in r_data:
        generate_regulon_heatmap(
            r_data["regulons_matrix"],
            "TF Regulon Activity - R (NES)",
            "02_regulon_heatmap_R.png", R_PLOTS
        )

    if "expr_l_r_log2_scale" in r_data:
        generate_communication_network(
            r_data["expr_l_r_log2_scale"],
            "Cell-Cell Communication - R",
            "03_communication_network_R.png", R_PLOTS
        )

    # Generate Python-style plots
    print("\n--- Python-style plots (outputs/py_plots/) ---")

    if "expr_l_r_log2_scale" in py_data:
        generate_pheatmap(
            py_data["expr_l_r_log2_scale"],
            "L-R Communication Score (Python, log2 scale)",
            "01_pheatmap_Py.png", PY_PLOTS
        )

    if "regulons_matrix" in py_data:
        generate_regulon_heatmap(
            py_data["regulons_matrix"],
            "TF Regulon Activity - Python (NES)",
            "02_regulon_heatmap_Py.png", PY_PLOTS
        )

    if "expr_l_r_log2_scale" in py_data:
        generate_communication_network(
            py_data["expr_l_r_log2_scale"],
            "Cell-Cell Communication - Python",
            "03_communication_network_Py.png", PY_PLOTS
        )

    # Generate comparison plots
    if py_data:
        print("\n--- Comparison plots (outputs/py_plots/) ---")
        for metric in ["expr_mean", "regulons_matrix", "expr_l_r_log2_scale"]:
            generate_comparison_plot(
                r_data, py_data, metric,
                f"{metric}: R vs Python",
                f"compare_{metric}.png", PY_PLOTS
            )

    print("\n" + "=" * 60)
    print(f"  R plots: {R_PLOTS}")
    print(f"  Py plots: {PY_PLOTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
