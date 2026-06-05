#!/usr/bin/env python
"""Generate publication-quality plots matching R cellcall visualizations.

Produces:
1. Sankey diagram (Ligand → Receptor → TF) using plotly
2. GSEA enrichment plot (4-panel: running score + hits + gradient + ranked metric)
3. Dot plot (L-R communication bubble plot)
4. Heatmap (regulons matrix)
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = 'tests/deliverables/plots'
os.makedirs(OUT_DIR, exist_ok=True)

# Global Set1 palette (9 colors)
SET1 = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
        '#FFFF33', '#A65628', '#F781BF', '#999999']
# Set1-based continuous colormap for heatmaps/gradients
set1_cmap = mcolors.LinearSegmentedColormap.from_list(
    'set1_grad', ['#377EB8', 'white', '#E41A1C'], N=256)

# Load pre-computed data
print("Loading data...")
em = pd.read_csv('tests/deliverables/py_expr_mean.csv', index_col=0)
rm = pd.read_csv('tests/deliverables/py_regulons_matrix.csv', index_col=0)
lr = pd.read_csv('tests/deliverables/py_expr_l_r_log2_scale.csv', index_col=0)

# Load pipeline for GSEA data and triple_relation
import rdata
from cellcall.connect_profile import connect_profile, _load_lr_tf_data
from cellcall.object import CellInter
from cellcall.normalize import counts2normalized_10x

extdata_dir = os.path.join('cellcall', 'extdata')
parsed = rdata.read_rda(os.path.join(extdata_dir, 'example_Data.Rdata'))
data_key = [k for k in parsed.keys() if isinstance(parsed[k], pd.DataFrame)][0]
data_df = parsed[data_key]
sample_ids = data_df.columns.tolist()
cell_types_raw = [s.split('_')[1] for s in sample_ids]
meta = pd.DataFrame({'sampleID': sample_ids, 'celltype': cell_types_raw})
data_cpm = counts2normalized_10x(data_df, to_type='CPM', scale_factor=1e6)
obj = CellInter(data={'count': data_df, 'withoutlog': data_cpm}, meta_data=meta, project='test', org='Homo sapiens')
triple_relation, target_relation = _load_lr_tf_data('Homo sapiens', True, extdata_dir)
del parsed, data_df, data_cpm

cell_types = list(em.columns)
print(f"Cell types: {cell_types}")

# Run pipeline for GSEA data
print("Running pipeline for GSEA data...")
result = connect_profile(obj, org='Homo sapiens', verbose=False, n_perm=500)


# ============================================================
# 1. SANKEY DIAGRAM: Ligand → Receptor → TF (plotly)
# ============================================================
print("Plot 1: Sankey diagram...")


def make_sankey_plotly(result, triple_relation, cell_type, top_ligands=6, top_tfs=6):
    """Create a clean Sankey: Ligand → Receptor → TF.

    Reduces clutter by:
    - Keeping only top ligands by total flow
    - Merging rare receptors into 'Other'
    - Keeping only top TFs by NES
    - Ordering all nodes by flow volume (biggest on top)
    - Fixed 3-column x,y positions to prevent crossing
    """
    lr_data = result['expr_l_r_log2_scale']
    rm_data = result['regulons_matrix']

    ct_cols = [c for c in lr_data.columns if cell_type in c]
    if not ct_cols:
        return None
    lr_active = lr_data[ct_cols].max(axis=1)
    lr_active = lr_active[lr_active > 0].sort_values(ascending=False)

    # Collect all flows
    flows = []
    for lr_pair in lr_active.index:
        lig, rec = lr_pair.split('-')
        tfs = triple_relation[
            (triple_relation['Ligand_Symbol'] == lig) &
            (triple_relation['Receptor_Symbol'] == rec)
        ]['TF_Symbol'].unique()
        for tf in tfs:
            if tf in rm_data.index and rm_data.loc[tf, cell_type] > 0:
                flows.append({'lig': lig, 'rec': rec, 'tf': tf,
                              'value': lr_active[lr_pair]})

    if not flows:
        return None

    df = pd.DataFrame(flows)

    # --- Keep only top ligands by total flow ---
    lig_flow = df.groupby('lig')['value'].sum().sort_values(ascending=False)
    keep_lig = set(lig_flow.head(top_ligands).index)
    df.loc[~df['lig'].isin(keep_lig), 'lig'] = 'Other'

    # --- Merge rare receptors (connected to <2 ligands or low total flow) ---
    rec_flow = df.groupby('rec')['value'].sum().sort_values(ascending=False)
    keep_rec = set(rec_flow.head(12).index)
    df.loc[~df['rec'].isin(keep_rec), 'rec'] = 'Other'

    # --- Keep only top TFs by NES ---
    active_tfs = rm_data.index[rm_data[cell_type] > 0].tolist()
    top_tf_set = set(sorted(active_tfs, key=lambda t: rm_data.loc[t, cell_type], reverse=True)[:top_tfs])
    df.loc[~df['tf'].isin(top_tf_set), 'tf'] = 'Other'

    # Aggregate
    df = df.groupby(['lig', 'rec', 'tf']).agg({'value': 'sum'}).reset_index()
    # Remove zero
    df = df[df['value'] > 0]

    # --- Order nodes by total flow volume (biggest first) ---
    lig_order = df.groupby('lig')['value'].sum().sort_values(ascending=False).index.tolist()
    rec_order = df.groupby('rec')['value'].sum().sort_values(ascending=False).index.tolist()
    tf_order = df.groupby('tf')['value'].sum().sort_values(ascending=False).index.tolist()

    # Build node list
    node_labels = lig_order + rec_order + tf_order
    n_lig, n_rec, n_tf = len(lig_order), len(rec_order), len(tf_order)
    n_nodes = n_lig + n_rec + n_tf

    # Index maps
    lig_idx = {l: i for i, l in enumerate(lig_order)}
    rec_idx = {r: i + n_lig for i, r in enumerate(rec_order)}
    tf_idx = {t: i + n_lig + n_rec for i, t in enumerate(tf_order)}

    # Fixed x,y positions — 3 strict columns
    x_positions = [0.01] * n_lig + [0.5] * n_rec + [0.99] * n_tf
    # y: evenly spaced within each column, ordered top-to-bottom by flow
    def spaced(n, top=0.95, bot=0.05):
        if n == 1:
            return [(top + bot) / 2]
        step = (top - bot) / (n - 1)
        return [top - i * step for i in range(n)]

    y_positions = spaced(n_lig) + spaced(n_rec) + spaced(n_tf)

    # Assign colors from Set1: each ligand distinct, Other=gray
    lig_to_color = {}
    ci = 0
    for l in lig_order:
        if l == 'Other':
            lig_to_color[l] = '#999999'
        else:
            lig_to_color[l] = SET1[ci % len(SET1)]
            ci += 1

    node_colors = (
        [lig_to_color.get(l, '#999999') for l in lig_order] +
        ['#C8C8C8'] * n_rec +
        ['#A8C8E8'] * n_tf
    )

    # Build links
    sources, targets, values, link_colors = [], [], [], []
    for _, row in df.iterrows():
        s1 = lig_idx[row['lig']]
        t1 = rec_idx[row['rec']]
        s2 = t1
        t2 = tf_idx[row['tf']]
        val = row['value']
        base = lig_to_color.get(row['lig'], '#999999')

        sources.append(s1); targets.append(t1); values.append(val)
        link_colors.append(mcolors.to_rgba(base, alpha=0.5))

        sources.append(s2); targets.append(t2); values.append(val)
        link_colors.append(mcolors.to_rgba(base, alpha=0.35))

    link_colors_str = [
        f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.2f})"
        for c in link_colors
    ]
    node_colors_str = [
        f"rgba({int(mcolors.to_rgb(c)[0]*255)},{int(mcolors.to_rgb(c)[1]*255)},{int(mcolors.to_rgb(c)[2]*255)},0.9)"
        for c in node_colors
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0),
            label=node_labels,
            color=node_colors_str,
            x=x_positions,
            y=y_positions,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors_str,
        )
    )])

    fig.update_layout(
        font=dict(size=9, family='Arial'),
        width=750,
        height=max(450, max(n_lig, n_rec, n_tf) * 28),
        margin=dict(l=15, r=15, t=15, b=15),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    return fig


# Generate Sankey for each cell type
for ct in cell_types:
    fig = make_sankey_plotly(result, triple_relation, ct, top_ligands=6, top_tfs=6)
    if fig is not None:
        fig.write_image(os.path.join(OUT_DIR, f'sankey_{ct}.png'), scale=2)
        print(f"  Saved: sankey_{ct}.png")



# ============================================================
# 2. GSEA ENRICHMENT PLOT (matching R's getGSEAplot 4-panel)
# ============================================================
print("Plot 2: GSEA enrichment plot...")


def compute_gsea_running_es(fc_values, gene_names, hit_mask):
    """Compute running enrichment score curve, matching R's enrichplot::gsInfo."""
    abs_vals = np.abs(fc_values)
    hit_sum = (hit_mask * abs_vals).sum()
    if hit_sum == 0:
        return None, None

    cum_hit = np.cumsum(hit_mask * abs_vals) / hit_sum
    cum_miss = np.cumsum(~hit_mask) / (~hit_mask).sum()
    running_es = cum_hit - cum_miss

    es = running_es[np.argmax(np.abs(running_es))]
    return running_es, es


def plot_gsea_enrichment(fc_list, target_relation, expr_set, cell_type, tf_name,
                         selected_genes=None, ax_es=None):
    """Plot GSEA enrichment matching R's getGSEAplot 4-panel layout.

    Panels (top to bottom, rel_heights = 1.5 : 0.5 : 0.2 : 1.5):
    1. Running enrichment score curve (geom_line)
    2. Hit positions (geom_linerange)
    3. FC gradient bar (geom_rect, 7-color blue→white→red)
    4. Ranked list metric (geom_segment lines, color #80b1d3)
    """
    fc_df = fc_list[cell_type].set_index('gene_id')['log2fc']
    fc_df = fc_df.replace([np.inf, -np.inf], 0).fillna(0)
    detect_genes = set(expr_set.index)
    fc_genes_in = [g for g in fc_df.index if g in detect_genes]
    fc_filtered = fc_df.loc[fc_genes_in]
    fc_sorted = fc_filtered.sort_values(ascending=False)
    fc_values = fc_sorted.values
    gene_names = fc_sorted.index

    # Get target genes for this TF
    targets = target_relation[target_relation['TF_Symbol'] == tf_name]['Target_Symbol'].tolist()
    targets_in = [t for t in targets if t in set(gene_names)]

    if len(targets_in) < 5:
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    hit_mask = np.zeros(len(gene_names), dtype=bool)
    for g in targets_in:
        if g in gene_to_idx:
            hit_mask[gene_to_idx[g]] = True

    running_es, es = compute_gsea_running_es(fc_values, gene_names, hit_mask)
    if running_es is None:
        return None

    x = np.arange(len(running_es))

    # Create 4-panel figure matching R's rel_heights = c(1.5, 0.5, 0.2, 1.5)
    if ax_es is None:
        fig, axes = plt.subplots(4, 1, figsize=(10, 7),
                                 gridspec_kw={'height_ratios': [1.5, 0.5, 0.2, 1.5],
                                              'hspace': 0.0})
        ax1, ax2, ax3, ax4 = axes
    else:
        # Single panel mode for combined plots
        ax1 = ax_es
        axes = [ax1]

    # Panel 1: Running enrichment score (geom_line)
    ax1.plot(x, running_es, color=SET1[1], linewidth=1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.2)
    ax1.set_ylabel('Enrichment\nScore', fontsize=12, fontweight='bold')
    ax1.set_title('')
    ax1.set_xlim(0, len(running_es))
    ax1.tick_params(axis='x', labelbottom=False, length=0)
    ax1.tick_params(axis='y', labelsize=12)
    for spine in ['top', 'right', 'bottom']:
        ax1.spines[spine].set_visible(False)
    ax1.margins(y=0.1)

    if ax_es is not None:
        return ax1

    # Panel 2: Hit positions (geom_linerange) — colored vertical lines
    hit_pos = np.where(hit_mask)[0]
    for h in hit_pos:
        ax2.vlines(x=h, ymin=0, ymax=1, color=SET1[1], linewidth=0.5)
    ax2.set_xlim(0, len(running_es))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', labelbottom=False, length=0)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax2.spines[spine].set_visible(False)

    # Panel 3: FC gradient bar — Set1 blue→white→red
    grad_colors = [SET1[1], '#6BAED6', '#A8D4F0', 'white', '#F0A8A8', '#E87070', SET1[0]]
    fc_abs_max = max(abs(fc_values.min()), abs(fc_values.max()))
    cmap = mcolors.LinearSegmentedColormap.from_list('r_gsea', grad_colors, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-fc_abs_max, vcenter=0, vmax=fc_abs_max)
    # Draw as rect per position (matching R's geom_rect)
    bar_colors = cmap(norm(fc_values))
    ax3.bar(x, np.ones(len(x)), width=1, color=bar_colors, edgecolor='none', align='edge')
    ax3.set_xlim(0, len(running_es))
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.tick_params(axis='x', labelbottom=False, length=0)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax3.spines[spine].set_visible(False)

    # Panel 4: Ranked list metric (geom_segment)
    ax4.vlines(x, fc_values, 0, color=SET1[1], linewidth=0.8)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.2)
    ax4.set_ylabel('Ranked list\nmetric', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Rank in ordered dataset', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, len(running_es))
    ax4.tick_params(axis='both', labelsize=12)
    for spine in ['top', 'right']:
        ax4.spines[spine].set_visible(False)
    # Enable bottom axis ticks (R does this for last panel)
    ax4.spines['bottom'].set_visible(True)
    ax4.tick_params(axis='x', labelbottom=True, length=3)

    # Optional: label selected genes with ggrepel-style
    if selected_genes:
        for g in selected_genes:
            if g in gene_to_idx:
                idx = gene_to_idx[g]
                ax4.annotate(g, xy=(idx, fc_values[idx]), fontsize=6,
                           ha='center', va='bottom' if fc_values[idx] > 0 else 'top',
                           color='black', fontweight='bold')

    return axes


# Generate GSEA plots for top TFs per cell type
for ct in cell_types:
    active = rm.index[rm[ct] > 0].tolist()
    if not active:
        continue
    # Pick the TF with highest NES
    best_tf = max(active, key=lambda t: rm.loc[t, ct])
    axes = plot_gsea_enrichment(result['fc_list'], target_relation,
                                obj.data['withoutlog'], ct, best_tf)
    if axes is not None:
        plt.savefig(os.path.join(OUT_DIR, f'gsea_{ct}_{best_tf}.png'), dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close()
        print(f"  Saved: gsea_{ct}_{best_tf}.png")


# ============================================================
# 3. DOT PLOT: L-R Communication (matching R's plotBubble style)
# ============================================================
print("Plot 3: Dot plot (L-R communication)...")

lr_data = result['expr_l_r_log2_scale']
cc_pairs = list(lr_data.columns)

# Filter to active pairs
lr_active_mask = (lr_data > 0).any(axis=1)
lr_active = lr_data[lr_active_mask]

if len(lr_active) > 0:
    # Select top LR pairs by max score
    top_lr = lr_active.max(axis=1).sort_values(ascending=False).head(25).index
    lr_plot = lr_active.loc[top_lr]

    # Melt for scatter plot
    rows = []
    for lr_name in lr_plot.index:
        for cc_pair in lr_plot.columns:
            val = lr_plot.loc[lr_name, cc_pair]
            if val > 0:
                rows.append({'LR': lr_name, 'CellPair': cc_pair, 'Score': val})
    plot_df = pd.DataFrame(rows)

    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(13, max(7, len(top_lr) * 0.35)))

        # Size: communication score
        size_min, size_max = 20, 200
        scores = plot_df['Score']
        sizes = size_min + (scores - scores.min()) / (scores.max() - scores.min() + 1e-10) * (size_max - size_min)

        # Color: communication score
        scatter = ax.scatter(
            x=plot_df['CellPair'],
            y=plot_df['LR'],
            s=sizes,
            c=plot_df['Score'],
            cmap=set1_cmap,
            alpha=0.85,
            edgecolors='gray',
            linewidth=0.3,
        )

        ax.set_xticks(range(len(cc_pairs)))
        ax.set_xticklabels(cc_pairs, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(top_lr)))
        ax.set_yticklabels(top_lr, fontsize=7)
        ax.set_title('L-R Communication Scores', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cell Type Pairs', fontsize=9)
        ax.set_ylabel('Ligand-Receptor', fontsize=9)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Communication Score', fontsize=8)

        # Size legend
        for s_val in [plot_df['Score'].min(), plot_df['Score'].median(), plot_df['Score'].max()]:
            ax.scatter([], [], s=size_min + (s_val - scores.min()) / (scores.max() - scores.min() + 1e-10) * (size_max - size_min),
                      c='gray', alpha=0.5, label=f'{s_val:.2f}')
        ax.legend(title='Score', loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'lr_dotplot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: lr_dotplot.png")


# ============================================================
# 4. HEATMAP: Regulons Matrix (matching R's pheatmap style)
# ============================================================
print("Plot 4: Regulons heatmap...")

# Active TFs sorted by total NES
active_tfs = rm.index[(rm > 0).any(axis=1)]
rm_active = rm.loc[active_tfs]
rm_active = rm_active.loc[rm_active.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(7, max(8, len(active_tfs) * 0.28)))

# Set1 gradient colormap
cmap = set1_cmap

# Draw heatmap
sns.heatmap(rm_active, cmap=cmap, annot=True, fmt='.2f', linewidths=0.5,
            linecolor='lightgray', ax=ax,
            cbar_kws={'label': 'NES', 'shrink': 0.4, 'aspect': 20},
            annot_kws={'size': 7})

# Style: bold labels, proper title
ax.set_title('Regulon NES Scores (Active TFs)', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Cell Type', fontsize=10, fontweight='bold')
ax.set_ylabel('Transcription Factor', fontsize=10, fontweight='bold')
ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'regulons_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: regulons_heatmap.png ({len(active_tfs)} TFs)")



print(f"\nAll plots saved to: {OUT_DIR}/")
print("Done!")
