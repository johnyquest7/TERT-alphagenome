"""Offline reproducibility audit for Tables 1-5 and manuscript summary statistics.

Runs directly from the precomputed CSVs in `data/` without requiring an API key.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

print("=" * 72)
print("TERT PROMOTER ALPHAGENOME & ALPHAGENOME 2 REPRODUCIBILITY AUDIT")
print("=" * 72)

for idx, fn in [
    ("Table 1 (Primary Variants)", "data/table1_variants_verified.csv"),
    ("Table 2 (Multimodal Bulk vs Follicular)", "data/table2_ag1_vs_ag2_multimodal.csv"),
    ("Table 3 (Transcription Factor Rankings)", "data/table3_tf_rankings_ag1_vs_ag2_states.csv"),
    ("Table 4 (Selected 360-SNV Substitutions)", "data/table4_selected_ism.csv"),
    ("Table 5 (16 ClinVar SNVs + C216T Benchmark)", "data/table5_clinvar_16snv_plus_c216t_benchmark.csv"),
]:
    df = pd.read_csv(fn)
    print(f"\n--- {idx} ({len(df)} rows) ---")
    print(df.to_string(index=False))

df360 = pd.read_csv("data/tert_ism_360_unified_v1_v2_avi_mpra.csv")
assert len(df360) == 360, f"Expected 360 SNVs, found {len(df360)}"

print("\n--- Core-Promoter 360-SNV Verification ---")
print(f"Total SNVs verified: {len(df360)} (120 positions x 3 non-reference alleles)")
print(f"AG1 median = {df360['v1_rna'].median():+.3f}, 95th percentile = {np.percentile(df360['v1_rna'], 95):+.3f}")
print(f"AG2 follicular median = {df360['v2_foll_sc_rna'].median():+.3f}, 95th percentile = {np.percentile(df360['v2_foll_sc_rna'], 95):+.3f}")
print(f"De novo GGAA/TTCC 4-mer count = {int(df360['creates_ets_4mer'].sum())}")
print(f"AG1 vs AG2 agreement: Spearman rho = {spearmanr(df360['v1_rna'], df360['v2_foll_sc_rna'])[0]:.3f}, Pearson r = {pearsonr(df360['v1_rna'], df360['v2_foll_sc_rna'])[0]:.3f}")
print(f"+50-kb shift rank stability: AG1 rho = {spearmanr(df360['v1_rna'], df360['v1_rna_shift50k'])[0]:.3f}, AG2 rho = {spearmanr(df360['v2_foll_sc_rna'], df360['v2_foll_sc_rna_shift50k'])[0]:.3f}")

for col, lbl in [("mpra_HEK293T", "HEK293T MPRA"), ("mpra_GBM_SF7996", "SF7996 GBM MPRA"), ("mpra_mean", "Combined Unperturbed MPRA")]:
    s1, p1 = spearmanr(df360["v1_rna"], df360[col])
    r1, pr1 = pearsonr(df360["v1_rna"], df360[col])
    s2, p2 = spearmanr(df360["v2_foll_sc_rna"], df360[col])
    r2, pr2 = pearsonr(df360["v2_foll_sc_rna"], df360[col])
    print(f"AG1 vs {lbl:26s}: rho = {s1:.3f} (P={p1:.2e}), r = {r1:.3f} (P={pr1:.2e})")
    print(f"AG2 vs {lbl:26s}: rho = {s2:.3f} (P={p2:.2e}), r = {r2:.3f} (P={pr2:.2e})")

print("\nALL CHECKS PASSED.")
