# Quantum-Inspired Feature Selection for Cerrado Biomass Estimation

[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 📄 Citation

If you use this code or data, please cite:

```bibtex
@article{ToBeDetermined2026,
  title={Quantum-inspired feature selection for Cerrado aboveground biomass estimation: QUBO formulation with rigorous spatial validation},
  author={To Be Determined},
  journal={Environmental Modelling \& Software},
  year={2026},
  note={Manuscript in preparation}
}
```

## 🌳 Overview

This repository contains code, data, and supplementary materials for the manuscript:

**"Quantum-inspired feature selection for Cerrado aboveground biomass estimation: QUBO formulation with rigorous spatial validation"**

We demonstrate the **first application of Quantum Approximate Optimization Algorithm (QAOA)** to feature selection in tropical savanna biomass modeling, with validation on IBM quantum hardware.

### Key Contributions

- 🔬 **QUBO-based feature selection** for ecological remote sensing
- 🌍 **Rigorous spatial validation** via leave-one-macro-region-out (LO-MRO)
- ⚛️ **Hardware quantum validation** on IBM Torino (perfect reproducibility across 10 runs)
- 📊 **Sensitivity analysis** comparing 5 vs 6 features (NDVI duplicate impact)
- 🎯 **Final performance:** R² = 0.608, RMSE = 21.28 Mg/ha (spatially blind test)

## 📁 Repository Structure

```
cerrado-biomass-qaoa/
├── data/                    # Processed datasets (n=290 plots)
├── scripts/                 # Reproducible Python/JavaScript code
│   ├── 01_gee_extraction/
│   ├── 02_biomass_calculation/
│   ├── 03_qaoa_selection/
│   ├── 04_spatial_validation/
│   ├── 05_ibm_quantum/
│   └── 06_sensitivity_analysis/
├── results/                 # JSON/CSV outputs from all analyses
├── figures/                 # Main and supplementary figures
├── supplementary/           # Extended tables, text, documentation
├── notebooks/               # Jupyter notebooks (exploratory)
└── docs/                    # Installation and reproduction guides
```

## 🚀 Quick Start

### 1. Clone repository
```bash
git clone https://github.com/yourusername/cerrado-biomass-qaoa.git
cd cerrado-biomass-qaoa
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate biomass_qaoa
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Run complete workflow
```bash
# Step 1: Calculate biomass from field inventory
python scripts/02_biomass_calculation/calcular_biomassa_IFN_corrigido.py

# Step 2: QAOA feature selection (classical simulation)
python scripts/03_qaoa_selection/script_1_qaoa_calibration.py

# Step 3: Spatial validation (nested CV + LO-MRO)
python scripts/04_spatial_validation/script_2_nested_cv.py
python scripts/04_spatial_validation/script_3_lomro_validation.py

# Step 4: Sensitivity analysis (k=5 vs k=6)
python scripts/06_sensitivity_analysis/sensitivity_ndvi_duplicate.py

# Step 5: QAOA validation on AerSimulator (k=5)
python scripts/05_ibm_quantum/qaoa_aer_k5_validation.py
```

### 4. View results
Results are saved to `results/` directory as JSON/CSV files. Key outputs:
- `qaoa_mega_varredura_resultados.json` - QAOA convergence across k=4..12
- `qaoa_ibm_k6_results.json` - IBM hardware validation (10 runs)
- `lomro_benchmark_summary.csv` - Spatial validation metrics
- `sensitivity_comparison.csv` - k=5 vs k=6 sensitivity analysis

## 📊 Data Availability

### Field Inventory
- **Source:** Brazilian National Forest Inventory (IFN-BR), SFB/IBGE
- **Region:** Goiás state, Central-West Brazil
- **Sample size:** 290 permanent plots (1 ha each)
- **Biomass range:** 1.2 - 189.4 Mg/ha
- **Vegetation:** Cerrado native formations (savanna to woodland)

**Note:** Raw IFN data subject to SFB/IBGE terms of use. Processed data included in `data/` directory.

### Remote Sensing Features
- **Optical:** Sentinel-2 / Landsat 8 (dry season 2018)
- **SAR:** Sentinel-1 GRD (VV, VH polarization)
- **LiDAR:** Meta/WRI Global Canopy Height 2023
- **Ancillary:** OpenLandMap soil, SRTM topography, MapBiomas land cover
- **Feature pool:** 20 candidates → 5 selected by QAOA

## 🔬 Methods Summary

### QUBO Formulation
Feature selection formulated as Quadratic Unconstrained Binary Optimization:

```
min E(x) = -Σᵢ wᵢxᵢ + λ(Σᵢ xᵢ - k)²
```

Where:
- `xᵢ ∈ {0,1}`: binary selector for feature i
- `wᵢ`: normalized quality score (R² marginal)
- `k`: target cardinality
- `λ`: penalty parameter (calibrated per k)

### Optimization Algorithms Compared
1. **QAOA (simulated)** - Quantum Approximate Optimization Algorithm (p=2 layers)
2. **QAOA (hardware)** - IBM Torino quantum processor validation
3. Exact enumeration (combinatorial brute force)
4. Genetic Algorithm
5. Simulated Annealing
6. RF importance ranking

### Validation Strategy
- **Nested CV:** 5-fold spatial GroupKFold (12 contiguous blocks)
- **LO-MRO:** Leave-one-macro-region-out (5 geographic holdouts)
- **Statistics:** Wilcoxon signed-rank + Holm-Bonferroni correction
- **Spatial blocks:** Longitude-based to maximize geographic separation

## 🖥️ Hardware Requirements

### Classical Computing
- **Minimum:** 16GB RAM, 4-core CPU
- **Recommended:** 32GB RAM, 8-core CPU, NVIDIA GPU (for XGBoost)
- **Execution time:** ~2 hours (complete workflow)

### Quantum Computing
- **Platform:** IBM Quantum (free account sufficient)
- **QPU time:** ~120 seconds (10 runs × 12s per job)
- **Simulator:** Qiskit AerSimulator (unlimited, runs locally)

## 📦 Software Dependencies

### Core
- Python 3.10+
- Qiskit 1.0+ (quantum algorithms)
- scikit-learn 1.3+ (machine learning)
- pandas 2.0+, numpy 1.24+
- scipy 1.11+ (statistics)

### Optional
- xgboost 2.0+ (GPU acceleration)
- matplotlib 3.7+, seaborn 0.12+ (visualization)
- jupyter-lab (notebooks)
- geopandas (spatial analysis)

### External APIs
- Google Earth Engine API (feature extraction - requires account)
- IBM Quantum services (hardware validation - requires account)

See `requirements.txt` for complete list with pinned versions.

## 📈 Key Results

### Feature Selection
**Selected features (k=5):**
1. `canopy_height` - Meta/WRI 2023 global canopy height
2. `clay_pct` - OpenLandMap soil clay content (0-5 cm)
3. `NDVI_seca` - Normalized Difference Vegetation Index (dry season)
4. `NDWI_seca` - Normalized Difference Water Index (dry season)
5. `HV_dB` - Sentinel-1 cross-pol backscatter difference

### Performance Metrics
| Validation | R² | RMSE (Mg/ha) | MAE (Mg/ha) | Bias (Mg/ha) |
|------------|-----|--------------|-------------|--------------|
| Nested CV (k=6) | 0.6577 | - | - | - |
| LO-MRO (k=6) | 0.6016 | 21.48 | 16.17 | -0.63 |
| LO-MRO (k=5) | 0.6077 | 21.28 | 15.95 | -0.48 |

**ΔRMSE (k=5 - k=6):** -0.20 Mg/ha (not significant, p=0.125)

### Quantum Hardware Validation
- **Platform:** IBM ibm_torino (127-qubit processor)
- **Runs:** 10 independent executions
- **Reproducibility:** Perfect (R² std = 0.0000 across 10 runs)
- **ZNE improvement:** ΔR² < 0.0001 (negligible hardware noise)

## 📚 Supplementary Materials

Extended documentation in `supplementary/`:

### Tables
- **S1:** Feature pool detailed specifications (20 features)
- **S2:** QUBO penalty calibration (λ by k=4..12)
- **S3:** Algorithm hyperparameters (all 6 methods)
- **S4:** Allometric equation coefficients

### Text
- **S1:** Allometric equation derivation and correction factors
- **S2:** QUBO mathematical formulation (Ising Hamiltonian)
- **S3:** Statistical methods (bootstrap, Wilcoxon, Holm-Bonferroni)
- **S4:** Sensitivity analysis (k=5 vs k=6 detailed results)

### Figures
- **S1:** Spatial blocks and macro-regions visualization
- **S2:** Feature importance by selection method
- **S3:** QUBO landscape characterization (energy gaps)
- **S4:** IBM hardware reproducibility traces

## 🔄 Reproducing Results

See `docs/reproduction_workflow.md` for detailed step-by-step instructions.

**Expected outputs:**
- ✅ QAOA convergence identical to `results/qaoa_mega_varredura_resultados.json`
- ✅ LO-MRO metrics within ±0.01 R² of reported values
- ✅ Sensitivity analysis ΔRMSE = -0.20 ± 0.05 Mg/ha

**Known sources of variability:**
- Random Forest: seed-controlled (SEED=42)
- QAOA optimizer: stochastic (seeds documented in scripts)
- Bootstrap CI: resampling (BOOTSTRAP_SEED=123)

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

Report bugs or request features via [GitHub Issues](https://github.com/yourusername/cerrado-biomass-qaoa/issues).

## 📧 Contact

- **Lead author:** [Your Name] (email@institution.edu)
- **Institution:** [Your University/Institute]
- **Lab/Group:** [Research Group]

For questions about:
- **Code:** Open a GitHub Issue
- **Data access:** Contact lead author
- **Collaboration:** Send email with proposal

## 📜 License

This project is licensed under the MIT License - see `LICENSE` file for details.

## 🙏 Acknowledgments

- Brazilian Forest Service (SFB) and IBGE for IFN data
- IBM Quantum for hardware access (ibm_torino)
- Google Earth Engine platform for remote sensing processing
- Meta AI and World Resources Institute for global canopy height dataset
- OpenLandMap consortium for soil data

## 🔗 Related Resources

- **Manuscript:** [Link when published]
- **Zenodo archive:** [DOI when created]
- **Presentation slides:** [Link if available]
- **Blog post:** [Link if available]

---

**Last updated:** February 2026  
**Status:** Manuscript in preparation for *Environmental Modelling & Software*
