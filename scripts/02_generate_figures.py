import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
})
sns.set_style("whitegrid")

PALETTE = {
    "C216T": "#7f7f7f",
    "C228T": "#c0392b",
    "CC242-243TT": "#e67e22",
    "C250T": "#8e44ad",
    "-57A>C": "#2980b9",
}

# Load unified V4 datasets
df_t2 = pd.read_csv(f"{OUT_DIR}/table2_ag1_vs_ag2_multimodal.csv")
df_360 = pd.read_csv(f"{OUT_DIR}/tert_ism_360_unified_v1_v2_avi_mpra.csv")
df_hm = pd.read_csv(f"{OUT_DIR}/tf_heatmap_comparison_data.csv")
df_deconv = pd.read_csv(f"{OUT_DIR}/ag2_thyroid_celltypes_deconvolution.csv")
df_cpg = pd.read_csv(f"{OUT_DIR}/ag2_follicular_wgbs_profiles_5variants.csv")
with open(f"{OUT_DIR}/summary_metrics.json") as f:
    metrics = json.load(f)

# =========================================================================
# FIGURE 1: Multimodal Profiles in AlphaGenome (Bulk Thyroid) & AlphaGenome 2 (Thyroid Follicular Cells)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=300, gridspec_kw={"width_ratios": [1.3, 1.0]})

ag1_cols = ["AG1 Bulk Thyroid RNA", "AG1 CAGE", "AG1 ATAC", "AG1 DNase", "AG1 H3K4me3", "AG1 H3K27ac", "AG1 PRO-cap*"]
ag1_labels = ["TERT RNA", "CAGE", "ATAC", "DNase", "H3K4me3", "H3K27ac", "PRO-cap*"]
m1 = df_t2.melt(id_vars=["Variant"], value_vars=ag1_cols, var_name="Modality", value_name="Score")
m1["Modality"] = m1["Modality"].map(dict(zip(ag1_cols, ag1_labels)))

sns.barplot(data=m1, x="Modality", y="Score", hue="Variant", palette=PALETTE, ax=ax1, edgecolor="black", linewidth=0.5)
ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax1.set_title("A. AlphaGenome: Bulk Thyroid Gland Modalities", fontweight="bold", pad=10)
ax1.set_ylabel("Predicted Effect (log2 fold-change)", fontweight="bold")
ax1.set_xlabel("")
ax1.legend(title="Variant", frameon=True, facecolor="white", loc="upper left")

ag2_cols = ["AG2 Follicular scRNA", "AG2 Follicular scATAC", "AG2 Follicular H3K4me3", "AG2 Follicular WGBS (THOR)"]
ag2_labels = ["scRNA-seq\n(TERT)", "scATAC-seq", "H3K4me3", "WGBS CpG\n(THOR)"]
m2 = df_t2.melt(id_vars=["Variant"], value_vars=ag2_cols, var_name="Modality", value_name="Score")
m2["Modality"] = m2["Modality"].map(dict(zip(ag2_cols, ag2_labels)))

sns.barplot(data=m2, x="Modality", y="Score", hue="Variant", palette=PALETTE, ax=ax2, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax2.set_title("B. AlphaGenome 2: Primary Thyroid Follicular Cells", fontweight="bold", pad=10)
ax2.set_ylabel("Predicted Effect Score", fontweight="bold")
ax2.set_xlabel("")
ax2.get_legend().remove()

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Figure1_v4_Multimodal_AG1_AG2.png", dpi=300)
fig.savefig(f"{FIG_DIR}/Figure1_v4_Multimodal_AG1_AG2.pdf")
plt.close(fig)
print("Generated Figure 1.")

# =========================================================================
# FIGURE 2: 360-SNV Follicular Saturation Mutagenesis + Direct MPRA Validation
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300, gridspec_kw={"width_ratios": [1.35, 1.0]})

p95_v2 = metrics["v2_p95"]
ax1.scatter(df_360["pos"], df_360["v2_foll_sc_rna"], color="#95a5a6", alpha=0.65, s=28, label="All SNVs (n = 360)", zorder=2)
ax1.axhline(p95_v2, color="#c0392b", linestyle="--", linewidth=1.1, label=f"95th Percentile ({p95_v2:+.3f})", zorder=1)
ax1.axhline(0, color="#555555", linestyle=":", linewidth=0.8, zorder=1)

annots = [
    (1295135, "A", "C250T (c.-146C>T)\nPathogenic", "#8e44ad", "*", 150, (-108, -10)),
    (1295113, "A", "C228T (c.-124C>T)\nOncogenic / Tier I", "#c0392b", "*", 150, (-52, 14)),
    (1295113, "T", "c.-124C>A (Tier I)", "#d35400", "o", 70, (12, 14)),
    (1295128, "A", "c.-139C>T (VUS)", "#e67e22", "^", 70, (12, 18)),
    (1295101, "A", "C216T (c.-112C>T)\nPTC VUS", "#2980b9", "s", 70, (-48, -32)),
    (1295154, "C", "c.-165C>G\nLikely Benign", "#27ae60", "D", 65, (-28, -34)),
    (1295163, "G", "c.-174G>C\nLikely Benign", "#27ae60", "D", 65, (10, -28)),
    (1295091, "T", "c.-102C>A\nGC-box loss", "#34495e", "v", 65, (12, -8)),
]
for pos, alt, lbl, col, mk, sz, off in annots:
    row = df_360[(df_360["pos"] == pos) & (df_360["alt"] == alt)]
    if not row.empty:
        yv = float(row.iloc[0]["v2_foll_sc_rna"])
        ax1.scatter(pos, yv, color=col, marker=mk, s=sz, edgecolor="black", linewidth=0.7, zorder=5)
        ax1.annotate(lbl, (pos, yv), xytext=off, textcoords="offset points",
                     fontsize=7.8, fontweight="bold", color=col,
                     arrowprops=dict(arrowstyle="->", color=col, lw=0.75, shrinkA=3, shrinkB=4))

ax1.set_xlim(1295066, 1295195)
ax1.set_ylim(-0.28, 0.76)
ax1.set_xlabel("GRCh38 Coordinate on Chromosome 5 (bp)", fontweight="bold")
ax1.set_ylabel("AlphaGenome 2 Follicular TERT scRNA Score", fontweight="bold")
ax1.set_title("A. Core-Promoter Saturation Mutagenesis (360 SNVs)", fontweight="bold", pad=10)
ax1.legend(frameon=True, facecolor="white", loc="upper left")

# Panel B: Direct correlation vs Kircher et al. (2019) GBM SF7996 MPRA
ets_mask = df_360["creates_ets_4mer"]
ax2.scatter(df_360.loc[~ets_mask, "mpra_GBM_SF7996"], df_360.loc[~ets_mask, "v2_foll_sc_rna"],
            color="#95a5a6", alpha=0.6, s=26, label="Non-ETS SNVs (n = 343)")
ax2.scatter(df_360.loc[ets_mask, "mpra_GBM_SF7996"], df_360.loc[ets_mask, "v2_foll_sc_rna"],
            color="#e67e22", edgecolor="black", linewidth=0.5, s=42, label="Creates GGAA/TTCC (n = 17)", zorder=4)

for pos, alt, lbl, col, off in [
    (1295135, "A", "C250T", "#8e44ad", (-38, -5)),
    (1295113, "A", "C228T", "#c0392b", (-40, 6)),
    (1295113, "T", "c.-124C>A", "#d35400", (-55, 8)),
    (1295101, "A", "C216T", "#2980b9", (8, -14)),
    (1295091, "T", "c.-102C>A", "#34495e", (10, -4)),
]:
    r = df_360[(df_360["pos"] == pos) & (df_360["alt"] == alt)].iloc[0]
    ax2.scatter(r["mpra_GBM_SF7996"], r["v2_foll_sc_rna"], color=col, s=75, edgecolor="black", linewidth=0.8, zorder=6)
    ax2.annotate(lbl, (r["mpra_GBM_SF7996"], r["v2_foll_sc_rna"]), xytext=off, textcoords="offset points",
                 fontsize=8.2, fontweight="bold", color=col,
                 arrowprops=dict(arrowstyle="->", color=col, lw=0.7))

rho_m = metrics["v2_vs_mpra_GBM_SF7996"]["spearman_rho"]
r_m = metrics["v2_vs_mpra_GBM_SF7996"]["pearson_r"]
ax2.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
ax2.axvline(0, color="#555555", linestyle=":", linewidth=0.8)
ax2.text(0.04, 0.80, f"Spearman rho = {rho_m:.3f}\nPearson r = {r_m:.3f}\n(P < 10^-60, n = 360)",
         transform=ax2.transAxes, fontsize=8.8, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc"))
ax2.set_xlabel("Experimental MPRA Effect in SF7996 (log2FC, Kircher et al.)", fontweight="bold")
ax2.set_ylabel("AlphaGenome 2 Follicular TERT scRNA Score", fontweight="bold")
ax2.set_title("B. Concordance With Experimental MPRA", fontweight="bold", pad=10)
ax2.legend(frameon=True, facecolor="white", loc="upper left")

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Figure2_v4_ISM_ClinVar_MPRA.png", dpi=300)
fig.savefig(f"{FIG_DIR}/Figure2_v4_ISM_ClinVar_MPRA.pdf")
plt.close(fig)
print("Generated Figure 2.")

# =========================================================================
# FIGURE 3: Transcription Factor State Switch (AG1 Pan-Biosample vs AG2 Follicular vs AG2 BRAF-V600E)
# =========================================================================
variants_ord = ["C250T", "C228T", "CC242-243TT", "-57A>C", "C216T"]
tf_ord = ["ETV1", "ETV5", "ETV4", "GABPA", "GABPB1", "ELF1", "ELK4", "POLR2G", "POLR2AphosphoS5", "CTCF"]
tf_disp = {
    "POLR2AphosphoS5": "POLR2A (Ser5-P)",
    "POLR2G": "POLR2G",
    "ETV1": "ETV1 (PEA3)",
    "ETV4": "ETV4 (PEA3)",
    "ETV5": "ETV5 (PEA3)",
    "GABPA": "GABPA",
    "GABPB1": "GABPB1",
    "ELF1": "ELF1",
    "ELK4": "ELK4",
    "CTCF": "CTCF",
}

piv1 = df_hm.pivot(index="TF", columns="Variant", values="AG1_PanBiosample").reindex(index=tf_ord, columns=variants_ord)
piv2 = df_hm.pivot(index="TF", columns="Variant", values="AG2_ThyroidFollicular").reindex(index=tf_ord, columns=variants_ord)
piv3 = df_hm.pivot(index="TF", columns="Variant", values="AG2_A375_BRAFV600E").reindex(index=tf_ord, columns=variants_ord)
for p in (piv1, piv2, piv3):
    p.index = [tf_disp.get(i, i) for i in p.index]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6), dpi=300, sharey=True)
sns.heatmap(piv1, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0], cbar=False, linewidths=0.4, linecolor="#dddddd")
axes[0].set_title("A. AlphaGenome\n(Pan-Biosample ChIP-seq)", fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("Transcription Factor / Complex", fontweight="bold")

sns.heatmap(piv2, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1], cbar=False, linewidths=0.4, linecolor="#dddddd")
axes[1].set_title("B. AlphaGenome 2\n(Normal Thyroid Follicular Cell)", fontweight="bold")
axes[1].set_xlabel("Promoter Variant", fontweight="bold")
axes[1].set_ylabel("")

sns.heatmap(piv3, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[2], cbar_kws={"label": "Predicted ChIP Binding Gain"}, linewidths=0.4, linecolor="#dddddd")
axes[2].set_title("C. AlphaGenome 2\n(BRAF V600E-Mutant Cell State)", fontweight="bold")
axes[2].set_xlabel("")
axes[2].set_ylabel("")

for ax in axes:
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Figure3_v4_TF_CellState_Switch.png", dpi=300)
fig.savefig(f"{FIG_DIR}/Figure3_v4_TF_CellState_Switch.pdf")
plt.close(fig)
print("Generated Figure 3.")

# =========================================================================
# FIGURE 4: AlphaGenome 2 CpG Methylation Remodeling & Thyroid Lineage Specificity
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.1), dpi=300, gridspec_kw={"width_ratios": [1.3, 1.0]})

# Panel A: Smoothed / rolling CpG delta-beta across shared reference CpGs in Exon 1, Core Promoter, and THOR
df_cpg_ref = df_cpg[df_cpg["ref_beta"] >= 0.005].copy()
for v_lbl in ["C250T", "C228T", "-57A>C", "CC242-243TT", "C216T"]:
    sub = df_cpg_ref[df_cpg_ref["variant"] == v_lbl].sort_values("pos_grch38")
    roll = sub["delta_beta"].rolling(window=7, center=True, min_periods=2).mean()
    lw = 2.1 if v_lbl in ("C250T", "C228T", "C216T") else 1.5
    ls = "--" if v_lbl == "C216T" else "-"
    ax1.plot(sub["pos_grch38"], roll, label=v_lbl, color=PALETTE[v_lbl], linewidth=lw, linestyle=ls)

ax1.axvspan(1294530, 1294989, color="#ebf5fb", alpha=0.55, label="Exon 1 / 5' Gene Body")
ax1.axvspan(1294989, 1295138, color="#fef9e7", alpha=0.65, label="Unmethylated Core Promoter")
ax1.axvspan(1295138, 1295568, color="#fdedec", alpha=0.50, label="Upstream THOR")
ax1.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax1.set_xlim(1294540, 1295600)
ax1.set_ylim(-0.006, 0.052)
ax1.set_xlabel("GRCh38 Coordinate on Chromosome 5 (bp)", fontweight="bold")
ax1.set_ylabel("Predicted WGBS Methylation Shift (Alt - Ref Beta)", fontweight="bold")
ax1.set_title("A. Spatial CpG Methylation Profile in Thyroid Follicular Cells", fontweight="bold", pad=10)
ax1.legend(frameon=True, facecolor="white", loc="upper left", ncol=2, fontsize=7.8)

# Panel B: Lineage comparison across constituent human thyroid cell types (THOR WGBS score)
cell_order = [
    "Thyroid follicular cell",
    "Bulk thyroid gland",
    "Thyroid smooth muscle cell",
    "Thyroid fibroblast",
    "Thyroid endothelial cell",
    "Thyroid stromal cell",
]
sub_dec = df_deconv[df_deconv["condition_name"].isin(cell_order)].copy()
sub_dec["condition_name"] = pd.Categorical(sub_dec["condition_name"], categories=cell_order, ordered=True)
sns.barplot(
    data=sub_dec,
    y="condition_name",
    x="ag2_wgbs_thor",
    hue="variant",
    hue_order=["C250T", "-57A>C", "CC242-243TT", "C228T", "C216T"],
    palette=PALETTE,
    ax=ax2,
    edgecolor="black",
    linewidth=0.45,
)
ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax2.set_xlabel("Predicted THOR WGBS Methylation Score", fontweight="bold")
ax2.set_ylabel("")
ax2.set_title("B. Lineage Restriction of THOR Methylation Across Thyroid Cells", fontweight="bold", pad=10)
ax2.legend(title="Variant", frameon=True, facecolor="white", loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/Figure4_v4_Thyroid_Deconvolution_and_WGBS.png", dpi=300)
fig.savefig(f"{FIG_DIR}/Figure4_v4_Thyroid_Deconvolution_and_WGBS.pdf")
plt.close(fig)
print("Generated Figure 4.")
