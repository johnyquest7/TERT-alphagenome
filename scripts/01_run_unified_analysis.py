import os
import json
import requests
import dotenv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

from alphagenome.data import genome
from alphagenome.models.v1 import dna_client as v1_client, variant_scorers as v1_scorers
from alphagenome.models.v2 import dna_client as v2_client, variant_scorers as v2_scorers, dna_output
from alphagenome.atlas import atlas

dotenv.load_dotenv(os.path.expanduser("~/.env"))
API_KEY = os.environ["ALPHAGENOME_API_KEY"]

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

print("Initializing AlphaGenome v1, AlphaGenome v2, and AlphaGenome Atlas clients...")
c1 = v1_client.create(api_key=API_KEY)
c2 = v2_client.create(api_key=API_KEY)
ac = atlas.create(API_KEY)

# Unified 1-Mb interval (width = 2**20 = 1,048,576 bp) centered at chr5:1,295,130
INTERVAL_MAIN = genome.Interval("chr5", 770842, 1819418)
INTERVAL_SHIFT50K = genome.Interval("chr5", 770842 + 50000, 1819418 + 50000)
# 120-bp core promoter ISM interval: 0-based [1295070, 1295190) == 1-based chr5:1,295,071-1,295,190
# Relative to NM_198253.3 ATG (c.1 = chr5:1,294,989 on minus strand): c.-201 to c.-82
ISM_INTERVAL = genome.Interval("chr5", 1295070, 1295190)

PRIMARY_VARIANTS = [
    ("C216T", genome.Variant("chr5", 1295101, "G", "A"), "c.-112C>T", "-112 bp", "Rare variant in PTC; uncertain functional effect [7]"),
    ("C228T", genome.Variant("chr5", 1295113, "G", "A"), "c.-124C>T", "-124 bp", "Canonical activating hotspot; creates de novo ETS motif [1,2]"),
    ("CC242-243TT", genome.Variant("chr5", 1295127, "GG", "AA"), "c.-139_-138delinsTT", "-139 to -138 bp", "Activating tandem variant; creates de novo ETS motif [8]"),
    ("C250T", genome.Variant("chr5", 1295135, "G", "A"), "c.-146C>T", "-146 bp", "Canonical activating hotspot; creates de novo ETS motif [1,2]"),
    ("-57A>C", genome.Variant("chr5", 1295046, "T", "G"), "c.-57A>C", "-57 bp", "Activating hereditary/somatic promoter mutation [2,9]"),
]

# =========================================================================
# 1. ALPHAGENOME 1 FIVE PRIMARY VARIANTS (from unified fixed 1-Mb run)
# =========================================================================
print("\n[1/7] Verifying AlphaGenome 1 five-variant fixed-1Mb matrix...")
v1_5var_path = "data/tert_five_variants_scores_fixed1mb.zip"
if os.path.exists(v1_5var_path):
    df_v1_5 = pd.read_csv(v1_5var_path)
else:
    v1_sc = [v1_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in ["RNA_SEQ", "CAGE", "ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "PROCAP"]]
    parts = []
    for lbl, var, _, _, _ in PRIMARY_VARIANTS:
        res = c1.score_variant(interval=INTERVAL_MAIN, variant=var, variant_scorers=v1_sc)
        for ad in res:
            td = v1_scorers.tidy_scores([ad], match_gene_strand=True)
            if td is not None and not td.empty:
                td["variant_label"] = lbl
                parts.append(td)
    df_v1_5 = pd.concat(parts, ignore_index=True)

# =========================================================================
# 2. ALPHAGENOME 2 FIVE PRIMARY VARIANTS ACROSS THYROID CELL STATES + AVI
# =========================================================================
print("\n[2/7] Scoring five primary variants in AlphaGenome 2 across thyroid cell states + AVI...")
THYROID_CONDITIONS = [
    ("UBERON:0002046_CL:0002258", "Thyroid follicular cell"),
    ("UBERON:0002046", "Bulk thyroid gland"),
    ("UBERON:0002046_CL:0000057", "Thyroid fibroblast"),
    ("UBERON:0002046_CL:0000115", "Thyroid endothelial cell"),
    ("UBERON:0002046_CL:0000499", "Thyroid stromal cell"),
    ("UBERON:0002046_CL:0000192", "Thyroid smooth muscle cell"),
    ("UBERON:0002046_CL:0000359", "Thyroid vascular smooth muscle cell"),
]

v2_mod_scorers = [
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["SC_GENE_EXPRESSION"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["SC_ATAC"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["CAGE"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["ATAC"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["DNASE"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["CHIP_HISTONE"],
    v2_scorers.RECOMMENDED_VARIANT_SCORERS["METHYLATION"],
]

deconv_rows = []
for cond_id, cond_name in THYROID_CONDITIONS:
    for lbl, var, hgvs, rel_atg, _ in PRIMARY_VARIANTS:
        res = c2.score_variant(interval=INTERVAL_MAIN, variant=var, condition=cond_id, variant_scorers=v2_mod_scorers)
        dfs = [v2_scorers.tidy_scores([ad], match_gene_strand=True) for ad in res]
        dfa = pd.concat([d for d in dfs if d is not None and not d.empty], ignore_index=True)

        sc_rna_all = dfa[(dfa["output_type"] == "SC_GENE_EXPRESSION") & (dfa["gene_name"] == "TERT")]["raw_score"].mean()
        sc_rna_obs_sub = dfa[(dfa["output_type"] == "SC_GENE_EXPRESSION") & (dfa["gene_name"] == "TERT") & (dfa["imputation_mode"] == "observed")]
        sc_rna_obs = sc_rna_obs_sub["raw_score"].mean() if not sc_rna_obs_sub.empty else np.nan

        sc_atac_all = dfa[dfa["output_type"] == "SC_ATAC"]["raw_score"].mean()
        sc_atac_obs_sub = dfa[(dfa["output_type"] == "SC_ATAC") & (dfa["imputation_mode"] == "observed")]
        sc_atac_obs = sc_atac_obs_sub["raw_score"].mean() if not sc_atac_obs_sub.empty else np.nan

        rna_bulk = dfa[(dfa["output_type"] == "RNA_SEQ") & (dfa["gene_name"] == "TERT")]["raw_score"].mean()
        cage_val = dfa[dfa["output_type"] == "CAGE"]["raw_score"].mean()
        atac_val = dfa[dfa["output_type"] == "ATAC"]["raw_score"].mean()
        dnase_val = dfa[dfa["output_type"] == "DNASE"]["raw_score"].mean()
        h3k4_val = dfa[(dfa["output_type"] == "CHIP_HISTONE") & (dfa["histone_mark"] == "H3K4me3")]["raw_score"].mean()
        h3k27_val = dfa[(dfa["output_type"] == "CHIP_HISTONE") & (dfa["histone_mark"] == "H3K27ac")]["raw_score"].mean()

        wgbs = dfa[(dfa["output_type"] == "METHYLATION") & (dfa["Assay title"] == "WGBS")]
        wgbs_all = wgbs["raw_score"].mean()
        wgbs_body = wgbs[wgbs["cpg_position"] <= 1294989]["raw_score"].mean()
        wgbs_prox = wgbs[(wgbs["cpg_position"] > 1294989) & (wgbs["cpg_position"] <= 1295137)]["raw_score"].mean()
        wgbs_thor = wgbs[(wgbs["cpg_position"] >= 1295138) & (wgbs["cpg_position"] <= 1295568)]["raw_score"].mean()

        deconv_rows.append({
            "condition_id": cond_id,
            "condition_name": cond_name,
            "variant": lbl,
            "hgvs": hgvs,
            "ag2_sc_rna_mean": sc_rna_all,
            "ag2_sc_rna_observed": sc_rna_obs,
            "ag2_sc_atac_mean": sc_atac_all,
            "ag2_sc_atac_observed": sc_atac_obs,
            "ag2_rna_seq": rna_bulk,
            "ag2_cage": cage_val,
            "ag2_atac": atac_val,
            "ag2_dnase": dnase_val,
            "ag2_h3k4me3": h3k4_val,
            "ag2_h3k27ac": h3k27_val,
            "ag2_wgbs_overall": wgbs_all,
            "ag2_wgbs_exon1_body": wgbs_body,
            "ag2_wgbs_core_promoter": wgbs_prox,
            "ag2_wgbs_thor": wgbs_thor,
        })

df_deconv = pd.DataFrame(deconv_rows)
df_deconv.to_csv(f"{OUT_DIR}/ag2_thyroid_celltypes_deconvolution.csv", index=False)
print("Saved ag2_thyroid_celltypes_deconvolution.csv")

# AVI for 5 variants (note CC242-243TT is MNV, SNVs supported)
avi_5 = {}
for lbl, var, _, _, _ in PRIMARY_VARIANTS:
    if len(var.reference_bases) == 1 and len(var.alternate_bases) == 1:
        res_avi = ac.query_variant(var, requested_scorers=["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE"])
        raw = float(res_avi["AVI_SCORE"].X[0, 0])
        q = float(res_avi["AVI_SCORE"].layers["quantiles"][0, 0])
        phred = -10.0 * np.log10(max(1.0 - q, 1e-10))
        fi_ad = res_avi["AVI_SCORE_FEATURE_IMPORTANCE"]
        top_feat = str(pd.Series(fi_ad.X[0], index=fi_ad.var["name"]).idxmax())
        avi_5[lbl] = (raw, q, phred, (1.0 - q) * 100.0, top_feat)
    else:
        avi_5[lbl] = (np.nan, np.nan, np.nan, np.nan, "MNV (not indexed)")

# Build unified Table 1 & Table 2
t1_rows, t2_rows = [], []
for lbl, var, hgvs, rel_atg, status in PRIMARY_VARIANTS:
    raw_a, q_a, phred_a, top_pct_a, top_feat_a = avi_5[lbl]
    t1_rows.append({
        "Variant": lbl,
        "GRCh38 Coordinate": f"{var.chromosome}:{var.position:,}" if len(var.reference_bases) == 1 else f"{var.chromosome}:{var.position:,}-{var.position + len(var.reference_bases) - 1:,}",
        "Forward Allele": f"{var.reference_bases}>{var.alternate_bases}",
        "HGVS (NM_198253.3)": hgvs,
        "Position Relative to ATG": rel_atg,
        "AVI Phred (Top %)": f"{phred_a:.2f} ({top_pct_a:.2f}%)" if not np.isnan(phred_a) else "N/A (dinucleotide)",
        "Functional and Clinical Context": status,
    })

    sub1 = df_v1_5[df_v1_5["variant_label"] == lbl]
    v1_rna = sub1[(sub1["output_type"] == "RNA_SEQ") & (sub1["biosample_name"] == "thyroid gland") & (sub1["gene_name"] == "TERT")]["raw_score"].mean()
    v1_cage = sub1[(sub1["output_type"] == "CAGE") & (sub1["biosample_name"] == "thyroid gland")]["raw_score"].mean()
    v1_atac = sub1[(sub1["output_type"] == "ATAC") & (sub1["biosample_name"] == "thyroid gland")]["raw_score"].mean()
    v1_dnase = sub1[(sub1["output_type"] == "DNASE") & (sub1["biosample_name"] == "thyroid gland")]["raw_score"].mean()
    v1_h3k4 = sub1[(sub1["output_type"] == "CHIP_HISTONE") & (sub1["biosample_name"] == "thyroid gland") & (sub1["histone_mark"] == "H3K4me3")]["raw_score"].mean()
    v1_h3k27 = sub1[(sub1["output_type"] == "CHIP_HISTONE") & (sub1["biosample_name"] == "thyroid gland") & (sub1["histone_mark"] == "H3K27ac")]["raw_score"].mean()
    v1_procap = sub1[sub1["output_type"] == "PROCAP"]["raw_score"].mean()

    sub2 = df_deconv[(df_deconv["variant"] == lbl) & (df_deconv["condition_id"] == "UBERON:0002046_CL:0002258")].iloc[0]
    t2_rows.append({
        "Variant": lbl,
        "AG1 Bulk Thyroid RNA": round(v1_rna, 3),
        "AG1 CAGE": round(v1_cage, 3),
        "AG1 ATAC": round(v1_atac, 3),
        "AG1 DNase": round(v1_dnase, 3),
        "AG1 H3K4me3": round(v1_h3k4, 3),
        "AG1 H3K27ac": round(v1_h3k27, 3),
        "AG1 PRO-cap*": round(v1_procap, 3),
        "AG2 Follicular scRNA": round(sub2["ag2_sc_rna_mean"], 3),
        "AG2 Follicular scATAC": round(sub2["ag2_sc_atac_mean"], 3),
        "AG2 Follicular H3K4me3": round(sub2["ag2_h3k4me3"], 3),
        "AG2 Follicular WGBS (THOR)": round(sub2["ag2_wgbs_thor"], 3),
        "AVI Phred": round(phred_a, 2) if not np.isnan(phred_a) else np.nan,
    })

pd.DataFrame(t1_rows).to_csv(f"{OUT_DIR}/table1_variants_verified.csv", index=False)
pd.DataFrame(t2_rows).to_csv(f"{OUT_DIR}/table2_ag1_vs_ag2_multimodal.csv", index=False)
print("Table 2 summary:")
print(pd.DataFrame(t2_rows).to_string(index=False))

# Export base-by-base WGBS CpG methylation profiles in primary thyroid follicular cells for all 5 variants
print("\n[3/7] Exporting base-resolution WGBS CpG profiles in thyroid follicular cells...")
cpg_profile_rows = []
sub_iv = genome.Interval("chr5", 1294530, 1295730)
for lbl, var, _, _, _ in PRIMARY_VARIANTS:
    pred = c2.predict_variant(
        interval=INTERVAL_MAIN,
        variant=var,
        condition="UBERON:0002046_CL:0002258",
        requested_outputs=[dna_output.OutputType.METHYLATION],
    )
    ref_vals = pred.reference.methylation.slice_by_interval(sub_iv).values[:, 0]
    alt_vals = pred.alternate.methylation.slice_by_interval(sub_iv).values[:, 0]
    nz = np.where((ref_vals > 0.002) | (alt_vals > 0.002))[0]
    for idx in nz:
        pos1 = 1294530 + idx + 1
        cpg_profile_rows.append({
            "variant": lbl,
            "pos_grch38": pos1,
            "rel_to_atg": -(pos1 - 1294989),
            "ref_beta": float(ref_vals[idx]),
            "alt_beta": float(alt_vals[idx]),
            "delta_beta": float(alt_vals[idx] - ref_vals[idx]),
        })
pd.DataFrame(cpg_profile_rows).to_csv(f"{OUT_DIR}/ag2_follicular_wgbs_profiles_5variants.csv", index=False)

# =========================================================================
# 4. TRANSCRIPTION FACTOR HIERARCHY: AG1 PAN-BIOSAMPLE vs AG2 CELL STATES
# =========================================================================
print("\n[4/7] Computing TF hierarchies across AG1 (pan-biosample + thyroid) and AG2 (follicular vs BRAF-V600E A375)...")
ets_genes = [
    "GABPA", "ETS1", "ETS2", "ELK1", "ELK3", "ELK4", "ELF1", "ELF2", "ELF3", "ELF4", "ELF5",
    "ETV1", "ETV2", "ETV3", "ETV4", "ETV5", "ETV6", "ETV7", "SPI1", "SPIB", "SPIC", "SPDEF",
    "ERG", "FLI1", "FEV", "EHF", "ERF"
]
tf_v1 = df_v1_5[df_v1_5["output_type"] == "CHIP_TF"]
tf_counts_v1 = tf_v1.groupby("transcription_factor")["track_name"].nunique()
factors_ge3_v1 = set(tf_counts_v1[tf_counts_v1 >= 3].index)

tf_summary_rows = []
ag2_tf_conditions = [
    ("UBERON:0002046_CL:0002258", "AG2_Follicular"),
    ("EFO:0002103", "AG2_A375_BRAFV600E"),
    ("EFO:0005720", "AG2_SKMEL5_BRAFV600E"),
]
ag2_tf_store = {}
for cond_id, cond_tag in ag2_tf_conditions:
    for lbl, var, _, _, _ in PRIMARY_VARIANTS:
        res_tf = c2.score_variant(
            interval=INTERVAL_MAIN,
            variant=var,
            condition=cond_id,
            variant_scorers=[v2_scorers.RECOMMENDED_VARIANT_SCORERS["CHIP_TF"]],
        )
        df_tf2 = v2_scorers.tidy_scores(res_tf, match_gene_strand=True)
        ag2_tf_store[(cond_tag, lbl)] = df_tf2

for lbl, _, _, _, _ in PRIMARY_VARIANTS:
    sub = tf_v1[tf_v1["variant_label"] == lbl]
    f_mean = sub.groupby("transcription_factor")["raw_score"].mean().sort_values(ascending=False)
    f_med = sub.groupby("transcription_factor")["raw_score"].median().sort_values(ascending=False)
    f_max = sub.groupby("transcription_factor")["raw_score"].max().sort_values(ascending=False)

    r_mean = f_mean.rank(ascending=False, method="min")
    r_med = f_med.rank(ascending=False, method="min")
    r_max = f_max.rank(ascending=False, method="min")

    f_ge3 = f_mean[f_mean.index.isin(factors_ge3_v1)]
    r_ge3 = f_ge3.rank(ascending=False, method="min")

    ets_present = [g for g in f_mean.index if g in ets_genes]
    top20 = list(f_mean.head(20).index)

    # Thyroid-only CTCF and POLR2A in AG1
    thy_tf = sub[sub["biosample_name"] == "thyroid gland"]
    thy_polr2a = thy_tf[thy_tf["transcription_factor"] == "POLR2A"]["raw_score"].mean()
    thy_ctcf = thy_tf[thy_tf["transcription_factor"] == "CTCF"]["raw_score"].mean()

    # AG2 Follicular & BRAF-V600E A375
    df_foll = ag2_tf_store[("AG2_Follicular", lbl)]
    fm_foll = df_foll.groupby("transcription_factor")["raw_score"].mean().sort_values(ascending=False)
    rm_foll = fm_foll.rank(ascending=False, method="min")

    df_a375 = ag2_tf_store[("AG2_A375_BRAFV600E", lbl)]
    fm_a375 = df_a375.groupby("transcription_factor")["raw_score"].mean().sort_values(ascending=False)
    rm_a375 = fm_a375.rank(ascending=False, method="min")

    df_skm = ag2_tf_store[("AG2_SKMEL5_BRAFV600E", lbl)]
    fm_skm = df_skm.groupby("transcription_factor")["raw_score"].mean().sort_values(ascending=False)
    rm_skm = fm_skm.rank(ascending=False, method="min")

    tf_summary_rows.append({
        "Variant": lbl,
        "AG1_ETV1_Score": round(float(f_mean["ETV1"]), 3),
        "AG1_ETV1_Rank_of_751": int(r_mean["ETV1"]),
        "AG1_ETV1_MedianAgg_Rank": int(r_med["ETV1"]),
        "AG1_ETV4_Rank_of_751": int(r_mean["ETV4"]),
        "AG1_ETV5_Rank_of_751": int(r_mean["ETV5"]),
        "AG1_GABPA_Rank_of_751": int(r_mean["GABPA"]),
        "AG1_GABPB1_Rank_of_751": int(r_mean["GABPB1"]),
        "AG1_Median_ETS_Rank_of_751": int(round(r_mean[ets_present].median())),
        "AG1_ETS_in_Top20": sum(1 for x in top20 if x in ets_genes),
        "AG1_GABPA_Rank_of_126_ge3": int(r_ge3["GABPA"]),
        "AG1_GABPB1_Rank_of_126_ge3": int(r_ge3["GABPB1"]),
        "AG1_Thyroid_POLR2A": round(float(thy_polr2a), 3),
        "AG1_Thyroid_CTCF": round(float(thy_ctcf), 3),
        "AG2_Foll_GABPA_Score": round(float(fm_foll["GABPA"]), 3),
        "AG2_Foll_GABPA_Rank_of_778": int(rm_foll["GABPA"]),
        "AG2_Foll_GABPB1_Rank_of_778": int(rm_foll["GABPB1"]),
        "AG2_Foll_ELF1_Rank_of_778": int(rm_foll["ELF1"]),
        "AG2_Foll_POLR2G_Rank_of_778": int(rm_foll["POLR2G"]),
        "AG2_Foll_POLR2A_S5P_Rank_of_778": int(rm_foll["POLR2AphosphoS5"]),
        "AG2_Foll_ETV1_Score": round(float(fm_foll["ETV1"]), 3),
        "AG2_Foll_ETV1_Rank_of_778": int(rm_foll["ETV1"]),
        "AG2_A375_ETV1_Score": round(float(fm_a375["ETV1"]), 3),
        "AG2_A375_ETV1_Rank_of_778": int(rm_a375["ETV1"]),
        "AG2_A375_GABPA_Rank_of_778": int(rm_a375["GABPA"]),
        "AG2_SKMEL5_ETV1_Score": round(float(fm_skm["ETV1"]), 3),
        "AG2_SKMEL5_ETV1_Rank_of_778": int(rm_skm["ETV1"]),
    })

df_tf_sum = pd.DataFrame(tf_summary_rows)
df_tf_sum.to_csv(f"{OUT_DIR}/table3_tf_rankings_ag1_vs_ag2_states.csv", index=False)
print("TF Summary (AG1 vs AG2 Follicular vs AG2 BRAF-V600E):")
print(df_tf_sum.to_string(index=False))

# Also save factor-wise matrix for Figure 3 heatmap (AG1 pan-biosample vs AG2 Follicular vs AG2 BRAF-V600E)
key_tfs = ["ETV1", "ETV4", "ETV5", "GABPA", "GABPB1", "ELF1", "ELK4", "POLR2AphosphoS5", "POLR2G", "KDM4A", "CTCF"]
hm_rows = []
for f in key_tfs:
    for lbl, _, _, _, _ in PRIMARY_VARIANTS:
        s1 = tf_v1[(tf_v1["variant_label"] == lbl) & (tf_v1["transcription_factor"] == f)]["raw_score"].mean()
        sf = ag2_tf_store[("AG2_Follicular", lbl)]
        s2_foll = sf[sf["transcription_factor"] == f]["raw_score"].mean()
        sa = ag2_tf_store[("AG2_A375_BRAFV600E", lbl)]
        s2_a375 = sa[sa["transcription_factor"] == f]["raw_score"].mean()
        hm_rows.append({"TF": f, "Variant": lbl, "AG1_PanBiosample": s1, "AG2_ThyroidFollicular": s2_foll, "AG2_A375_BRAFV600E": s2_a375})
pd.DataFrame(hm_rows).to_csv(f"{OUT_DIR}/tf_heatmap_comparison_data.csv", index=False)

# =========================================================================
# 5-7. UNIFIED 360-SNV ISM, AVI, CLINVAR, AND MPRA
# =========================================================================

def fast_parse_v1(ism_list, full=True):
    # Precompute row/col indices from first element!
    ad0 = ism_list[0][0]
    tert_rows = np.flatnonzero(ad0.obs["gene_name"].values == "TERT")
    thy_rna_cols = np.flatnonzero((ad0.var["biosample_name"].values == "thyroid gland") & np.isin(ad0.var["strand"].values, ["-", "."]))
    if full:
        thy_cage_cols = np.flatnonzero(ism_list[0][1].var["biosample_name"].values == "thyroid gland")
        thy_atac_cols = np.flatnonzero(ism_list[0][2].var["biosample_name"].values == "thyroid gland")
    rows = []
    for ad_l in ism_list:
        v = ad_l[0].uns["variant"]
        rna_v = float(ad_l[0].X[tert_rows][:, thy_rna_cols].mean())
        rec = {"pos": int(v.position), "ref": str(v.reference_bases), "alt": str(v.alternate_bases), "v1_rna": rna_v}
        if full:
            rec["v1_cage"] = float(ad_l[1].X[0, thy_cage_cols].mean())
            rec["v1_atac"] = float(ad_l[2].X[0, thy_atac_cols].mean())
        rows.append(rec)
    return pd.DataFrame(rows)

def fast_parse_v2(ism_list, full=True):
    ad0 = ism_list[0][0]
    tert_rows = np.flatnonzero(ad0.obs["gene_name"].values == "TERT")
    str_cols = np.flatnonzero(np.isin(ad0.var["strand"].values, ["-", "."]))
    if full:
        wgbs_cols = np.flatnonzero(ism_list[0][4].var["Assay title"].values == "WGBS")
        cpg_pos = ism_list[0][4].obs["cpg_position"].values.astype(int)
        thor_rows = np.flatnonzero((cpg_pos >= 1295138) & (cpg_pos <= 1295568))
    rows = []
    for ad_l in ism_list:
        v = ad_l[0].uns["variant"]
        sc_rna = float(ad_l[0].X[tert_rows][:, str_cols].mean())
        rec = {"pos": int(v.position), "ref": str(v.reference_bases), "alt": str(v.alternate_bases), "v2_foll_sc_rna": sc_rna}
        if full:
            rec["v2_foll_sc_atac"] = float(ad_l[1].X.mean())
            rec["v2_foll_cage"] = float(ad_l[2].X.mean())
            rec["v2_foll_atac"] = float(ad_l[3].X.mean())
            rec["v2_foll_wgbs_thor"] = float(ad_l[4].X[thor_rows][:, wgbs_cols].mean())
        rows.append(rec)
    return pd.DataFrame(rows)

print("1. Scoring v1 ISM main (12 chunks)...")
v1_ism_main = fast_parse_v1(c1.score_ism_variants(
    INTERVAL_MAIN, ISM_INTERVAL,
    variant_scorers=[v1_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in ["RNA_SEQ", "CAGE", "ATAC"]],
    progress_bar=False, max_workers=6
), full=True)
print(f"   done in {time.time()-t_start:.1f}s")

print("2. Scoring v1 ISM +50kb shift...")
v1_ism_shift = fast_parse_v1(c1.score_ism_variants(
    INTERVAL_SHIFT50K, ISM_INTERVAL,
    variant_scorers=[v1_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]],
    progress_bar=False, max_workers=6
), full=False).rename(columns={"v1_rna": "v1_rna_shift50k"})[["pos", "ref", "alt", "v1_rna_shift50k"]]
print(f"   done in {time.time()-t_start:.1f}s")

print("3. Scoring v2 Follicular ISM main...")
v2_ism_main = fast_parse_v2(c2.score_ism_variants(
    INTERVAL_MAIN, ISM_INTERVAL, condition="UBERON:0002046_CL:0002258",
    variant_scorers=[v2_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in ["SC_GENE_EXPRESSION", "SC_ATAC", "CAGE", "ATAC", "METHYLATION"]],
    progress_bar=False, max_workers=6
), full=True)
print(f"   done in {time.time()-t_start:.1f}s")

print("4. Scoring v2 Follicular ISM +50kb shift...")
v2_ism_shift = fast_parse_v2(c2.score_ism_variants(
    INTERVAL_SHIFT50K, ISM_INTERVAL, condition="UBERON:0002046_CL:0002258",
    variant_scorers=[v2_scorers.RECOMMENDED_VARIANT_SCORERS["SC_GENE_EXPRESSION"]],
    progress_bar=False, max_workers=6
), full=False).rename(columns={"v2_foll_sc_rna": "v2_foll_sc_rna_shift50k"})[["pos", "ref", "alt", "v2_foll_sc_rna_shift50k"]]
print(f"   done in {time.time()-t_start:.1f}s")

print("5. Querying Atlas AVI across [1295070, 1295190)...")
avi_iv_res = ac.query_interval(ISM_INTERVAL, requested_scorers=["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE"])
avi_ad = avi_iv_res["AVI_SCORE"]
fi_ad = avi_iv_res["AVI_SCORE_FEATURE_IMPORTANCE"]
avi_rows = []
for i in range(avi_ad.shape[0]):
    v = avi_ad.obs["variant"].iloc[i]
    raw_v = float(avi_ad.X[i, 0])
    q_v = float(avi_ad.layers["quantiles"][i, 0])
    phred_v = -10.0 * np.log10(max(1.0 - q_v, 1e-10))
    top_mod = str(pd.Series(fi_ad.X[i], index=fi_ad.var["name"]).idxmax())
    avi_rows.append({
        "pos": int(v.position), "ref": str(v.reference_bases), "alt": str(v.alternate_bases),
        "avi_raw": raw_v, "avi_quantile": q_v, "avi_phred": phred_v, "avi_top_modality": top_mod
    })
df_avi_360 = pd.DataFrame(avi_rows)

df_360 = (v1_ism_main
          .merge(v1_ism_shift, on=["pos", "ref", "alt"])
          .merge(v2_ism_main, on=["pos", "ref", "alt"])
          .merge(v2_ism_shift, on=["pos", "ref", "alt"])
          .merge(df_avi_360, on=["pos", "ref", "alt"]))

with open("data/reference_snapshots/ucsc_seq_1295065_1295196.json") as f:
    u_seq = json.load(f)["dna"].upper()
def get_base(p1):
    return u_seq[p1 - 1295066]

comp = str.maketrans("ACGT", "TGCA")
ets_flags, hgvs_list = [], []
for _, r in df_360.iterrows():
    p = int(r["pos"])
    ref_b, alt_b = str(r["ref"]), str(r["alt"])
    hgvs_list.append(f"c.{-(p - 1294989)}{ref_b.translate(comp)}>{alt_b.translate(comp)}")
    ref_7 = "".join(get_base(k) for k in range(p - 3, p + 4))
    alt_7 = ref_7[:3] + alt_b + ref_7[4:]
    alt_creates = any((alt_7[i:i+4] in ("GGAA", "TTCC")) and (ref_7[i:i+4] not in ("GGAA", "TTCC")) for i in range(4))
    ets_flags.append(bool(alt_creates))

df_360["hgvs_nm198253"] = hgvs_list
df_360["creates_ets_4mer"] = ets_flags
df_360["v1_rank"] = df_360["v1_rna"].rank(ascending=False, method="min").astype(int)
df_360["v1_shift50k_rank"] = df_360["v1_rna_shift50k"].rank(ascending=False, method="min").astype(int)
df_360["v2_rank"] = df_360["v2_foll_sc_rna"].rank(ascending=False, method="min").astype(int)
df_360["v2_shift50k_rank"] = df_360["v2_foll_sc_rna_shift50k"].rank(ascending=False, method="min").astype(int)
df_360["avi_local_rank"] = df_360["avi_phred"].rank(ascending=False, method="min").astype(int)

# Merge cached Kircher MPRA
with open("data/reference_snapshots/ucsc_kircher_mpra.json") as f:
    r_mpra = json.load(f)
m_items = [x for x in r_mpra["mpraVarDb"] if "Kircher" in str(x.get("mpraStudy", "")) and x.get("alt") in ["A", "C", "G", "T"] and len(str(x.get("ref", ""))) == 1]
df_m = pd.DataFrame(m_items)
df_m["pos"] = df_m["chromEnd"].astype(int)
df_m["exp_idx"] = df_m.groupby(["pos", "ref", "alt"]).cumcount()
piv_m = df_m.pivot(index=["pos", "ref", "alt"], columns="exp_idx", values="log2FC").reset_index()
piv_m.columns = ["pos", "ref", "alt", "mpra_HEK293T", "mpra_GBM_SF7996", "mpra_GBM_siScramble", "mpra_GBM_siGABPA"]
piv_m["mpra_mean"] = piv_m[["mpra_HEK293T", "mpra_GBM_SF7996", "mpra_GBM_siScramble"]].mean(axis=1)

df_360 = df_360.merge(piv_m, on=["pos", "ref", "alt"], how="left").sort_values("v1_rank").reset_index(drop=True)
df_360.to_csv(f"{OUT_DIR}/tert_ism_360_unified_v1_v2_avi_mpra.csv", index=False)

# Summary metrics
ets_sub = df_360[df_360["creates_ets_4mer"]]
metrics = {
    "v1_median": float(df_360["v1_rna"].median()),
    "v1_p95": float(np.percentile(df_360["v1_rna"], 95)),
    "v2_median": float(df_360["v2_foll_sc_rna"].median()),
    "v2_p95": float(np.percentile(df_360["v2_foll_sc_rna"], 95)),
    "ets_4mer_count": int(df_360["creates_ets_4mer"].sum()),
    "ets_v1_median": float(ets_sub["v1_rna"].median()),
    "ets_v1_rank_range": [int(ets_sub["v1_rank"].min()), int(ets_sub["v1_rank"].max())],
    "ets_v2_median": float(ets_sub["v2_foll_sc_rna"].median()),
    "ets_v2_rank_range": [int(ets_sub["v2_rank"].min()), int(ets_sub["v2_rank"].max())],
    "v1_shift50k_spearman": float(spearmanr(df_360["v1_rna"], df_360["v1_rna_shift50k"])[0]),
    "v2_shift50k_spearman": float(spearmanr(df_360["v2_foll_sc_rna"], df_360["v2_foll_sc_rna_shift50k"])[0]),
    "v1_vs_v2_spearman": float(spearmanr(df_360["v1_rna"], df_360["v2_foll_sc_rna"])[0]),
    "v1_vs_v2_pearson": float(pearsonr(df_360["v1_rna"], df_360["v2_foll_sc_rna"])[0]),
}
for tag, p_coord in [("C250T", 1295135), ("C228T", 1295113)]:
    r_row = df_360[(df_360["pos"] == p_coord) & (df_360["alt"] == "A")].iloc[0]
    metrics[f"{tag}_v1_shift_pct"] = float((r_row["v1_rna_shift50k"] - r_row["v1_rna"]) / r_row["v1_rna"] * 100.0)
    metrics[f"{tag}_v2_shift_pct"] = float((r_row["v2_foll_sc_rna_shift50k"] - r_row["v2_foll_sc_rna"]) / r_row["v2_foll_sc_rna"] * 100.0)

for m_col in ["mpra_HEK293T", "mpra_GBM_SF7996", "mpra_mean"]:
    s1, p1 = spearmanr(df_360["v1_rna"], df_360[m_col])
    r1, pr1 = pearsonr(df_360["v1_rna"], df_360[m_col])
    s2, p2 = spearmanr(df_360["v2_foll_sc_rna"], df_360[m_col])
    r2, pr2 = pearsonr(df_360["v2_foll_sc_rna"], df_360[m_col])
    metrics[f"v1_vs_{m_col}"] = {"spearman_rho": round(float(s1), 3), "spearman_p": float(p1), "pearson_r": round(float(r1), 3), "pearson_p": float(pr1)}
    metrics[f"v2_vs_{m_col}"] = {"spearman_rho": round(float(s2), 3), "spearman_p": float(p2), "pearson_r": round(float(r2), 3), "pearson_p": float(pr2)}

with open(f"{OUT_DIR}/summary_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Summary metrics:", json.dumps(metrics, indent=2))

# Table 4
sel_coords = [
    (1295135, "A", "C250T (c.-146C>T)"),
    (1295113, "A", "C228T (c.-124C>T)"),
    (1295141, "G", "c.-152A>C"),
    (1295113, "T", "c.-124C>A (C228 alternative)"),
    (1295128, "A", "c.-139C>T (CC242-243TT component)"),
    (1295113, "C", "c.-124C>G (C228 alternative)"),
    (1295135, "T", "c.-146C>A (C250 alternative)"),
    (1295135, "C", "c.-146C>G (C250 alternative)"),
    (1295101, "A", "C216T (c.-112C>T)"),
    (1295091, "T", "c.-102C>A (GC-box disruption)"),
]
t4_rows = []
for p_c, a_c, name_c in sel_coords:
    r = df_360[(df_360["pos"] == p_c) & (df_360["alt"] == a_c)].iloc[0]
    t4_rows.append({
        "Variant / Substitution": name_c,
        "Coordinate (Fwd)": f"chr5:{p_c:,} {r['ref']}>{a_c}",
        "AG1 Thyroid RNA (Rank)": f"{r['v1_rna']:+.3f} ({int(r['v1_rank'])})",
        "AG1 +50kb Rank": int(r["v1_shift50k_rank"]),
        "AG2 Follicular scRNA (Rank)": f"{r['v2_foll_sc_rna']:+.3f} ({int(r['v2_rank'])})",
        "AG2 +50kb Rank": int(r["v2_shift50k_rank"]),
        "AVI Phred": round(float(r["avi_phred"]), 2),
        "MPRA GBM log2FC": round(float(r["mpra_GBM_SF7996"]), 2),
    })
df_t4 = pd.DataFrame(t4_rows)
df_t4.to_csv(f"{OUT_DIR}/table4_selected_ism.csv", index=False)
print("\nTable 4:")
print(df_t4.to_string(index=False))

# Table 5 (cached ClinVar JSON)
with open("data/reference_snapshots/clinvar_1295071_1295190_live.json") as f:
    cv_raw = json.load(f)
cids, esum = cv_raw["ids"], cv_raw["esummary"]
comp_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
cv_benchmark = []
for cid in cids:
    item = esum.get(cid, {})
    if item.get("obj_type") != "single nucleotide variant":
        continue
    title = item.get("title", "")
    locs = [v for v in item.get("variation_set", [{}])[0].get("variation_loc", []) if v.get("assembly_name") == "GRCh38"]
    if not locs:
        continue
    pos_cv = int(locs[0]["start"])
    m_chg = re.search(r"([ACGT])>([ACGT])$", title)
    if not m_chg:
        continue
    tx_r, tx_a = m_chg.group(1), m_chg.group(2)
    fwd_r, fwd_a = comp_map[tx_r], comp_map[tx_a]
    germline = item.get("germline_classification", {}).get("description", "")
    onc = item.get("oncogenicity_classification", {}).get("description", "")
    impact = item.get("clinical_impact_classification", {}).get("description", "")
    if germline == "Pathogenic" or onc == "Oncogenic":
        status_str = "Pathogenic / Oncogenic (Tier I)" if germline == "Pathogenic" else "Oncogenic / Tier I (Conflicting germline)"
    elif impact:
        status_str = impact
    else:
        status_str = germline or "Uncertain significance"
    match = df_360[(df_360["pos"] == pos_cv) & (df_360["alt"] == fwd_a)]
    if not match.empty:
        r = match.iloc[0]
        cv_benchmark.append({
            "Source / Accession": f"VCV{int(cid):09d}",
            "HGVS (NM_198253.3)": r["hgvs_nm198253"],
            "Legacy Name": "C250T" if (pos_cv==1295135 and fwd_a=="A") else ("C228T" if (pos_cv==1295113 and fwd_a=="A") else "-"),
            "GRCh38 (Fwd)": f"chr5:{pos_cv:,} {fwd_r}>{fwd_a}",
            "Clinical Classification": status_str,
            "AG1 RNA (Rank)": f"{r['v1_rna']:+.3f} ({int(r['v1_rank'])})",
            "AG2 Follicular scRNA (Rank)": f"{r['v2_foll_sc_rna']:+.3f} ({int(r['v2_rank'])})",
            "AG1 CAGE": round(float(r["v1_cage"]), 3),
            "AVI Phred": round(float(r["avi_phred"]), 2),
            "MPRA GBM": round(float(r["mpra_GBM_SF7996"]), 2),
            "v2_rank_num": int(r["v2_rank"]),
        })

r_c216 = df_360[(df_360["pos"] == 1295101) & (df_360["alt"] == "A")].iloc[0]
cv_benchmark.append({
    "Source / Accession": "Literature PTC [7]",
    "HGVS (NM_198253.3)": "c.-112C>T",
    "Legacy Name": "C216T",
    "GRCh38 (Fwd)": "chr5:1,295,101 G>A",
    "Clinical Classification": "Uncertain significance (PTC)",
    "AG1 RNA (Rank)": f"{r_c216['v1_rna']:+.3f} ({int(r_c216['v1_rank'])})",
    "AG2 Follicular scRNA (Rank)": f"{r_c216['v2_foll_sc_rna']:+.3f} ({int(r_c216['v2_rank'])})",
    "AG1 CAGE": round(float(r_c216["v1_cage"]), 3),
    "AVI Phred": round(float(r_c216["avi_phred"]), 2),
    "MPRA GBM": round(float(r_c216["mpra_GBM_SF7996"]), 2),
    "v2_rank_num": int(r_c216["v2_rank"]),
})

df_cv = pd.DataFrame(cv_benchmark).sort_values("v2_rank_num").drop(columns=["v2_rank_num"]).reset_index(drop=True)
df_cv.to_csv(f"{OUT_DIR}/table5_clinvar_16snv_plus_c216t_benchmark.csv", index=False)
print("\nTable 5:")
print(df_cv.to_string(index=False))
print(f"\nALL DONE in {time.time()-t_start:.1f}s!")
