# Repository Completion TODO

## ✅ COMPLETED

### Repository Structure
- [x] Main directory structure created
- [x] README.md (comprehensive)
- [x] LICENSE (MIT)
- [x] CITATION.cff
- [x] requirements.txt
- [x] environment.yml
- [x] .gitignore
- [x] data/README.md (data dictionary)
- [x] docs/installation_guide.md

## 📋 TO COMPLETE

### 1. Copy Existing Files from Project

#### Data (`data/`)
- [ ] Copy `biomassa_por_UA_corrigido.csv` from `/mnt/project/`
- [ ] Copy `goias_df_features_buffer50m_v3_2018.csv` from `/mnt/project/`

#### Scripts (`scripts/`)

**02_biomass_calculation/**
- [ ] Copy `calcular_biomassa_IFN_corrigido.py` from `/mnt/project/`
- [ ] Create README.md explaining allometric equations

**03_qaoa_selection/**
- [ ] Copy `script_1_qaoa_calibracao_features.py` from `/mnt/project/`
- [ ] Rename to `qaoa_feature_selection.py`
- [ ] Create README.md with QUBO explanation

**04_spatial_validation/**
- [ ] Copy `script_2_mapa_goias_corrigido.py` from `/mnt/project/`
- [ ] Copy `script_3_validacao_espacial.py` from `/mnt/project/`
- [ ] Rename appropriately

**05_ibm_quantum/**
- [ ] Extract IBM hardware validation code (if separate script exists)
- [ ] Copy `qaoa_aer_k5_validation.py` (created today)
- [ ] Create README with IBM Quantum setup instructions

**06_sensitivity_analysis/**
- [ ] Copy `script_3_sensitivity_ndvi_re.py` (created today)
- [ ] Create README explaining sensitivity methodology

#### Results (`results/`)
- [ ] Copy all JSON files from `/mnt/project/`:
  - `qaoa_mega_varredura_justa_resultados.json`
  - `qaoa_ibm_k6_results.json`
  - `best_global_subset.json`
  - `top_methods_subsets.json`
- [ ] Copy all CSV files:
  - `lomro_*.csv` (5 files)
  - `nestedcv_*.csv` (3 files)
  - `sensitivity_*.csv` (2 files)
- [ ] Create `qaoa_aer_k5_validation.json` (run script to generate)

#### Figures (`figures/`)

**main/**
- [ ] Copy key figures from `/mnt/project/`:
  - `fig_map_macroregions_phytophysiognomy.png`
  - `fig_lomro_scatter_baseline.png`
  - `fig_delta_rmse_summary.png`
- [ ] Create/copy additional main figures as needed

**supplementary/**
- [ ] Copy remaining figures from `/mnt/project/`:
  - `fig_r2_boxplot.png`
  - `fig_calibration_residuals_*.png`
  - `fig_oof_scatter_*.png` (all methods)
  - `fig_heatmap_delta_rmse.png`
  - `fig_sensitivity_*.png`

#### Logs (`results/` or new `logs/` directory)
- [ ] Copy log files:
  - `log_script_1.txt`
  - `log_script_2.txt`
  - `log_script_3.txt`

### 2. Create Missing Files

#### Scripts

**01_gee_extraction/**
- [ ] Create/copy `extract_features_gee.js` (GEE JavaScript code)
- [ ] Create README explaining GEE workflow
- [ ] Document feature extraction parameters

**03_qaoa_selection/**
- [ ] Create `test_installation.py` (quick test script)

#### Supplementary Materials

**supplementary/tables/**
- [ ] Create `tableS1_feature_pool_detailed.csv`
  - All 20 features with full specifications
- [ ] Create `tableS2_qubo_penalty_calibration.csv`
  - λ values for k=4..12
- [ ] Create `tableS3_algorithm_hyperparameters.csv`
  - All 6 methods with parameters
- [ ] Create `tableS4_allometric_coefficients.csv`
  - Ribeiro & Chave equation details

**supplementary/text/**
- [ ] `S1_allometric_derivation.md`
  - Detailed allometric equation derivation
  - Correction factors explanation
  - Multi-stem treatment
- [ ] `S2_qubo_mathematical_details.md`
  - QUBO → Ising conversion
  - Hamiltonian formulation
  - Penalty calibration methodology
- [ ] `S3_statistical_methods_extended.md`
  - Bootstrap procedure details
  - Wilcoxon test specifics
  - Holm-Bonferroni correction
  - Block-based resampling
- [ ] `S4_sensitivity_analysis_full.md`
  - Complete k=5 vs k=6 analysis
  - Per-holdout breakdown
  - Wilcoxon results table

#### Documentation

**docs/**
- [ ] `reproduction_workflow.md`
  - Step-by-step execution guide
  - Expected outputs
  - Troubleshooting common issues
- [ ] `faq.md`
  - Common questions
  - Interpretation guides
- [ ] `troubleshooting.md`
  - Error messages and solutions

#### Notebooks (Optional but recommended)

**notebooks/**
- [ ] `01_data_exploration.ipynb`
  - Biomass distribution
  - Feature correlations
  - Spatial visualization
- [ ] `02_qubo_landscape_visualization.ipynb`
  - Energy landscape plots
  - Convergence visualization
- [ ] `03_results_visualization.ipynb`
  - Main figures generation
  - Supplementary plots
- [ ] `04_sensitivity_analysis.ipynb`
  - Interactive k=5 vs k=6 comparison

### 3. Final Setup Tasks

#### GitHub
- [ ] Create GitHub repository (public or private)
- [ ] Push initial commit
- [ ] Add comprehensive .gitattributes for large files (if needed)
- [ ] Enable GitHub Actions for CI/CD (optional)

#### Zenodo
- [ ] Link GitHub repo to Zenodo
- [ ] Create first release (v1.0.0)
- [ ] Get DOI
- [ ] Update README and CITATION.cff with DOI

#### Testing
- [ ] Test complete workflow on fresh clone
- [ ] Verify all scripts run without errors
- [ ] Check all file paths are relative (not absolute)
- [ ] Confirm results reproducibility

#### Documentation
- [ ] Update README with actual DOI
- [ ] Add author information (replace placeholders)
- [ ] Add ORCID IDs
- [ ] Update contact emails
- [ ] Add acknowledgments section

### 4. Pre-Submission Checklist

Before manuscript submission:
- [ ] All data files in place and documented
- [ ] All scripts tested and working
- [ ] README complete with accurate information
- [ ] Supplementary materials comprehensive
- [ ] DOI obtained from Zenodo
- [ ] License finalized
- [ ] Contact information current
- [ ] Links in manuscript match repository

## Priority Order

**PHASE 1 (ESSENTIAL)** - Do before manuscript submission:
1. Copy all existing scripts/data
2. Create supplementary tables (S1-S4)
3. Create basic supplementary text (S1-S4)
4. Test workflow reproducibility
5. Get Zenodo DOI

**PHASE 2 (IMPORTANT)** - Can be done during review:
1. Create Jupyter notebooks
2. Expand documentation (FAQ, troubleshooting)
3. Add GEE extraction code
4. Polish README and guides

**PHASE 3 (NICE TO HAVE)** - Post-publication:
1. Add CI/CD workflows
2. Create video tutorials
3. Blog post
4. Extended examples

## Notes

- **Private vs Public:** Can keep repo private during review, add reviewers as collaborators
- **Large files:** If data >100MB, consider Git LFS or Zenodo data deposition
- **Code polish:** Scripts work as-is, but add more comments for public consumption
- **Paths:** Make all paths relative, use `os.path.join()` for cross-platform compatibility

---

**Created:** February 2026  
**Status:** Repository structure complete, content migration in progress
