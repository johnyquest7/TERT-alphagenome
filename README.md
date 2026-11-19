# In Silico Analysis of *TERT* Promoter Variants in Thyroid Follicular Cells and Bulk Thyroid Tissue Using AlphaGenome and AlphaGenome 2

**Author:** Johnson Thomas, MD, FACE  
**Affiliation:** Department of Endocrinology, Mercy Hospital, Springfield, Missouri 65807  

> **Research-Use Disclaimer:** AlphaGenome and AlphaGenome 2 are non-clinical research models developed by Google DeepMind. None of the scores, rankings, or scripts in this repository are validated for clinical diagnosis, variant classification in patient reports, or surgical/therapeutic decision-making.

---

## Overview

This repository contains all reproducible Python scripts, precomputed variant-effect matrices, reference snapshots, and publication figures supporting the study:

> *In Silico Analysis of TERT Promoter Variants in Thyroid Follicular Cells and Bulk Thyroid Tissue Using AlphaGenome and AlphaGenome 2* (Version 4).

Key components evaluated in a unified 1-Mb GRCh38 interval (`chr5:770,842-1,819,418`, width = $2^{20}$ = 1,048,576 bp):
1. **Multi-modal regulatory comparison** of five primary *TERT* (`NM_198253.3`) promoter variants—canonical hotspots `C228T` (`c.-124C>T`) and `C250T` (`c.-146C>T`), activating dinucleotide `CC242-243TT` (`c.-139_-138delinsTT`), activating variant `-57A>C` (`c.-57A>C`), and rare papillary thyroid carcinoma (PTC) variant `C216T` (`c.-112C>T`)—across bulk thyroid gland tracks (**AlphaGenome**) and primary human thyroid follicular epithelial cells (`UBERON:0002046_CL:0002258`, **AlphaGenome 2**), alongside genome-wide **AlphaGenome Variant Impact (AVI)** Phred scores.
2. **Thyroid gland cell-type deconvolution and WGBS CpG DNA methylation** across constituent primary thyroid cell populations (`CL:0002258` follicular cells, fibroblasts, endothelial cells, smooth muscle cells, and stromal cells) and promoter subregions (exon 1 / 5' gene body, unmethylated core promoter, and upstream *TERT* hypermethylated oncological region [THOR]).
3. **Cell-state-dependent transcription factor hierarchies** comparing AlphaGenome pan-biosample ChIP-seq (751 factors across 1,617 tracks) with AlphaGenome 2 conditioned on quiescent normal primary thyroid follicular cells versus *BRAF* V600E-mutant MAPK-active cancer cell states (778 factors).
4. **360-variant core-promoter saturation mutagenesis** (`chr5:1,295,071-1,295,190`; `c.-201` to `c.-82` relative to ATG), `+50-kb` input-framing sensitivity (`chr5:820,842-1,869,418`), benchmarking against all **16 curated NCBI ClinVar single-nucleotide variants** in the region, and quantitative comparison against published *TERT* massively parallel reporter assay (MPRA) data (Kircher et al., *Nat Commun.* 2019;10:3583).

---

## Repository Structure

```text
TERT-alphagenome/
├── data/
│   ├── table1_variants_verified.csv                    # Table 1: Verified GRCh38 & NM_198253.3 coordinates + AVI
│   ├── table2_ag1_vs_ag2_multimodal.csv                # Table 2: Bulk thyroid (AG1) vs follicular cell (AG2) scores
│   ├── table3_tf_rankings_ag1_vs_ag2_states.csv        # Table 3: TF hierarchies across AG1 & AG2 cell states
│   ├── table4_selected_ism.csv                         # Table 4: Selected substitutions from 360-SNV scan
│   ├── table5_clinvar_16snv_plus_c216t_benchmark.csv   # Table 5: All 16 ClinVar SNVs + PTC C216T benchmark
│   ├── tert_ism_360_unified_v1_v2_avi_mpra.csv         # Complete 360-SNV AG1, AG2, +50kb shift, AVI, & MPRA matrix
│   ├── ag2_thyroid_celltypes_deconvolution.csv         # AG2 scores across 7 thyroid tissue/cell-type conditions
│   ├── ag2_follicular_wgbs_profiles_5variants.csv      # Base-resolution follicular WGBS beta & delta-beta profiles
│   ├── tf_heatmap_comparison_data.csv                  # Factor-wise binding matrix underlying Figure 3
│   ├── tert_five_variants_scores_fixed1mb.zip          # Compressed 96,725-row track-level AG1 fixed-1Mb matrix
│   ├── summary_metrics.json                            # Summary correlations, percentiles, and framing metrics
│   └── reference_snapshots/                            # Frozen ClinVar, UCSC hg38 sequence, and MPRAVarDB snapshots
├── figures/
│   ├── Figure1_v4_Multimodal_AG1_AG2.{png,pdf}
│   ├── Figure2_v4_ISM_ClinVar_MPRA.{png,pdf}
│   ├── Figure3_v4_TF_CellState_Switch.{png,pdf}
│   └── Figure4_v4_Thyroid_Deconvolution_and_WGBS.{png,pdf}
├── scripts/
│   ├── 01_run_unified_analysis.py                      # Live API pipeline (requires ALPHAGENOME_API_KEY)
│   ├── 02_generate_figures.py                          # Offline figure generator (rebuilds Figures 1-4 from data/)
│   └── 03_verify_tables_and_stats.py                   # Offline audit of Tables 1-5 and all correlations
├── requirements.txt
└── LICENSE
```

---

## Quick Start & Reproducing Outputs

### 1. Offline Verification & Figure Regeneration (No API Key Required)

All precomputed outputs and frozen public reference snapshots are provided in `data/`. You can verify every table, statistical correlation, and figure in seconds:

```bash
pip install -r requirements.txt

# Verify Tables 1-5, medians/percentiles, +50-kb shift stability, and MPRA correlations
python scripts/03_verify_tables_and_stats.py

# Regenerate Figures 1-4 (300 DPI PNG and vector PDF) in figures/
python scripts/02_generate_figures.py
```

*(Or with [`uv`](https://docs.astral.sh/uv/): `uv run --with-requirements requirements.txt scripts/03_verify_tables_and_stats.py`)*

### 2. Re-running Live AlphaGenome, AlphaGenome 2, and Atlas AVI Queries

To regenerate all predictions directly from the Google DeepMind AlphaGenome servers, obtain a non-commercial research API key from [deepmind.google.com/science/alphagenome](https://deepmind.google.com/science/alphagenome/) and store `ALPHAGENOME_API_KEY=<your_key>` in `~/.env`:

```bash
python scripts/01_run_unified_analysis.py
python scripts/02_generate_figures.py
```
