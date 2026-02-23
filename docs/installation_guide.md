# Installation Guide

## Prerequisites

- **Operating System:** Linux, macOS, or Windows with WSL2
- **Python:** 3.10 or higher
- **Memory:** Minimum 16GB RAM (32GB recommended)
- **Disk space:** ~5GB (includes data, results, environments)

## Option 1: Conda (Recommended)

### 1. Install Miniconda/Anaconda

If not already installed:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Create environment

```bash
cd cerrado-biomass-qaoa
conda env create -f environment.yml
conda activate biomass_qaoa
```

### 3. Verify installation

```bash
python -c "import qiskit, sklearn, pandas; print('All packages imported successfully!')"
```

## Option 2: pip + venv

### 1. Create virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import qiskit, sklearn, pandas; print('Success!')"
```

## Optional: GPU Support (XGBoost)

### NVIDIA GPU (CUDA)

1. Install CUDA Toolkit 11.8+
2. Install XGBoost with GPU support:

```bash
pip uninstall xgboost
pip install xgboost[gpu]
```

3. Verify GPU detection:

```bash
python -c "import xgboost as xgb; print(xgb.XGBRegressor(tree_method='gpu_hist').get_params())"
```

## External API Setup

### Google Earth Engine

1. Create GEE account: https://earthengine.google.com/signup/
2. Initialize credentials:

```bash
earthengine authenticate
```

3. Test access:

```python
import ee
ee.Initialize()
print("GEE initialized successfully!")
```

### IBM Quantum (Optional - for hardware validation)

1. Create IBM Quantum account: https://quantum.ibm.com/
2. Get API token from account settings
3. Save credentials:

```bash
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN_HERE")
```

4. Test connection:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print("Available backends:", service.backends())
```

**Note:** Hardware validation requires QPU credits (~120s for 10 runs).

## Troubleshooting

### Import errors

**Problem:** `ModuleNotFoundError: No module named 'qiskit'`

**Solution:**
```bash
pip install qiskit qiskit-aer qiskit-algorithms
```

### GDAL/Rasterio issues (Linux)

**Problem:** `ERROR: Failed building wheel for GDAL`

**Solution:**
```bash
conda install -c conda-forge gdal rasterio geopandas
```

### Memory errors

**Problem:** `MemoryError` during nested CV

**Solution:**
- Reduce `n_jobs=-1` to `n_jobs=4` in scripts
- Close other applications
- Use machine with more RAM

### Qiskit version conflicts

**Problem:** `ImportError: cannot import name 'QAOA'`

**Solution:**
```bash
pip install --upgrade qiskit qiskit-algorithms
```

### XGBoost GPU not detected

**Problem:** `XGBoost GPU not found, falling back to CPU`

**Solution:**
- Verify CUDA installation: `nvcc --version`
- Reinstall XGBoost GPU: `pip install xgboost --no-cache-dir`
- Check GPU visibility: `nvidia-smi`

## Testing Installation

Run quick test:

```bash
python -c "
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('scikit-learn: OK')
print('Qiskit: OK')
print('All dependencies working!')
"
```

## Next Steps

After successful installation:

1. Read `docs/reproduction_workflow.md` for execution instructions
2. Run test script: `python scripts/03_qaoa_selection/test_installation.py`
3. Execute complete workflow (see Quick Start in main README)

## Support

- **Installation issues:** Open [GitHub Issue](https://github.com/yourusername/cerrado-biomass-qaoa/issues)
- **Conda environments:** https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- **Qiskit documentation:** https://qiskit.org/documentation/
- **scikit-learn:** https://scikit-learn.org/stable/install.html

## System-Specific Notes

### Windows (WSL2 recommended)

Use Windows Subsystem for Linux 2 for best compatibility:
```powershell
wsl --install
```

### macOS (M1/M2 Apple Silicon)

Use Rosetta for some packages:
```bash
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
```

### HPC/Cluster

Load modules before installation:
```bash
module load python/3.10
module load cuda/11.8  # if GPU available
```

---

**Last updated:** February 2026
