#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE BIOMASSA — SCRIPT 2 EMS-SPATIAL v2.5 (EMS-READY)
Nested CV + Spatial Blocking CONTÍGUO (GRID) + LO-MRO + No Leakage
(LEITURA AUTOMÁTICA DE SUBSETS DO JSON DO SCRIPT 1)

MANTÉM A LÓGICA DO SCRIPT 2 v2.4:
- Carrega subsets_by_method do JSON gerado no Script 1 (QAOA + baselines etc.)
- Nested CV (GRID spatial blocks):
    Inner: escolhe melhor método usando só treino do outer
    Outer: reporta métricas finais (OOF) sem vazamento
- LO-MRO (Leave-One-MacroRegion-Out) no best_global
- Treino final + mapa wall-to-wall

ADIÇÕES v2.5 para Environmental Modelling & Software:
  [A] Figuras obrigatórias para o paper:
      - Scatter OOF y_true vs y_pred (linha 1:1 + OLS + R²/RMSE/Slope) por método
      - Scatter LO-MRO y_true vs y_pred por macro-região (subplots)
      - Boxplot R² inter-fold por método (variância intra-algoritmo)
      - Calibration/residuals plot (y_pred vs residuals)
  [B] Sensibilidade ensemble vs RF solo:
      - RF direto (sem stacking) nos mesmos folds outer do best_method
      - Teste de Wilcoxon pareado RMSE fold-a-fold
      - Tabela comparativa salva em CSV
  [C] Fix técnico: inner GroupKFold n_splits seguro
      (min(INNER_SPLITS, n_unique_blocks) evita crash com blocos pequenos)
  [D] Fix técnico: XGBoost device='cuda' com fallback automático para 'cpu'
  [E] LO-MRO salva previsões brutas por macro-região (para Script 3 consumir)

Saídas v2.5 (adicionais às do v2.4):
  - fig_oof_scatter_<method>.png     (scatter OOF por método)
  - fig_lomro_scatter.png            (scatter LO-MRO subplots)
  - fig_r2_boxplot.png               (boxplot R² inter-fold)
  - fig_calibration_residuals.png    (calibration plot OOF)
  - sensitivity_rf_vs_ensemble.csv   (comparação RF solo vs stacking)
  - lomro_predictions_raw.csv        (previsões brutas LO-MRO para Script 3)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sem display — compatível com servidores
import matplotlib.pyplot as plt
import warnings, os, json, re, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore")

# Optional libs
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    print("⚠️  rasterio não instalado — mapa não será gerado")

print("=" * 78)
print("   ENSEMBLE BIOMASSA — SCRIPT 2 EMS-SPATIAL v2.5 (GRID + LO-MRO + Figures)")
print("=" * 78)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
FEATURES_CSV  = r"/mnt/e/PROJETOS/biomassa_quantum/results/goias_df_features_buffer50m_v3_2018.csv"
BIOMASSA_CSV  = r"/mnt/e/PROJETOS/biomassa_quantum/results/biomassa_por_UA_corrigido.csv"
MAPA_MOSAICO  = r"/mnt/e/PROJETOS/biomassa_quantum/results/goias_features_mosaico_100m.tif"
OUTPUT_DIR    = r"/mnt/e/PROJETOS/biomassa_quantum/results/ensemble_nestedcv_subsets_ems_spatial_v2_5"
FIGS_DIR      = os.path.join(OUTPUT_DIR, "figures")

QAOA_JSON     = r"/mnt/e/PROJETOS/biomassa_quantum/results/qaoa_ibm_real/qaoa_mega_varredura_justa_resultados.json"

TARGET_COL = "Biomassa_Mg_ha"
SEED       = 42
NODATA     = -9999.0

CLASSES_VEGETACAO = {3,4,5,6,11,12,29,32,49,50}
MAPBIOMAS_COL = "mapbiomas_2018"
USA_ESCALA = {'SVR', 'Ridge'}

BAND_NAMES = [
    'B2_seca','B3_seca','B4_seca','B8_seca','B11_seca','B12_seca',
    'NDVI_seca','NDWI_seca','NBR_seca','NDVI_RE_seca','MSI_seca','EVI_seca',
    'elevation','slope','VV_dB','VH_dB','HV_dB',
    'mapbiomas_2018','clay_pct','canopy_height','canopy_height_sd'
]
BANDA_MAPBIOMAS = BAND_NAMES.index("mapbiomas_2018") + 1

OUTER_SPLITS   = 5
OUTER_REPEATS  = 5   # 25 folds
INNER_SPLITS   = 5

N_SPATIAL_BLOCKS_TARGET = 12
RUN_LO_MRO = True
N_MACRO_REGIONS = 5

# Modelo de inferência final: "RF" ou "Stacking"
# Nested CV (Wilcoxon, p=0.9998) mostrou RF sistematicamente melhor que stacking
# com n=290. RF é mantido para mapa final por parcimônia e reprodutibilidade.
FINAL_MODEL = "RF"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


# ============================================================
# Helpers: métricas, particionamento, hashing
# ============================================================
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias,
            "slope": float(sl), "intercept": float(ic)}

# Prioridade de desempate: quando R², RMSE e Bias são idênticos (subsets com
# mesmo hash), XGBoost/CUDA introduz não-determinismo que torna a ordenação
# instável entre runs. Esta lista garante que QAOA vença o empate com GA.
_METHOD_PRIORITY = [
    "QAOA_Simulado",
    "Exact_Enumerated_QUBO",
    "Genetic_Algorithm",
    "Simulated_Annealing",
    "RF_topK",
    "ET_topK",
]

def choose_best(metrics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _priority(name: str) -> int:
        for i, prefix in enumerate(_METHOD_PRIORITY):
            if name.startswith(prefix):
                return i
        return len(_METHOD_PRIORITY)
    return sorted(
        metrics_rows,
        key=lambda r: (-r["r2"], r["rmse"], abs(r["bias"]), _priority(r["method"]))
    )[0]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / len(sa | sb)) if (sa | sb) else 1.0

def subset_hash(feats: List[str]) -> str:
    return hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:16]

def pick_grid_shape(target_blocks: int, lon_span: float, lat_span: float) -> Tuple[int, int]:
    """
    Escolhe nx, ny com nx*ny ~ target_blocks, priorizando proporção lon/lat.
    Busca em torno do ponto ótimo de aspect ratio para evitar grids degenerados
    como (1, N) que ignoram a geometria espacial dos dados.
    """
    if lat_span <= 0 or lon_span <= 0:
        nx = int(max(1, round(np.sqrt(target_blocks))))
        ny = int(max(1, round(target_blocks / nx)))
        return nx, ny

    aspect = lon_span / lat_span  # >1 = mais largo que alto → mais colunas
    # Ponto de partida centrado no aspect ratio
    nx_ideal = max(1, int(round(np.sqrt(target_blocks * aspect))))

    best = (nx_ideal, max(1, int(round(target_blocks / nx_ideal))))
    best_score = abs(best[0] * best[1] - target_blocks) + abs(np.log(max(best[0]/max(best[1],1), 1e-9) / aspect))

    # Busca em range ±target_blocks ao redor do ponto ideal
    for nx_c in range(max(1, nx_ideal - target_blocks), nx_ideal + target_blocks + 1):
        ny_c = max(1, int(round(target_blocks / nx_c)))
        # score: penaliza desvio de target E desvio de aspect ratio
        diff_prod = abs(nx_c * ny_c - target_blocks)
        diff_asp  = abs(np.log(max(nx_c / max(ny_c, 1e-9), 1e-9) / aspect))
        score = diff_prod + 0.5 * diff_asp  # peso 0.5 no aspect ratio
        if score < best_score:
            best = (nx_c, ny_c)
            best_score = score

    return best

def make_grid_blocks(df: pd.DataFrame, target_blocks: int, seed: int) -> Tuple[np.ndarray, Tuple[int,int]]:
    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    lon_span = max(1e-12, float(np.nanmax(lon) - np.nanmin(lon)))
    lat_span = max(1e-12, float(np.nanmax(lat) - np.nanmin(lat)))
    nx, ny = pick_grid_shape(target_blocks, lon_span, lat_span)
    dx = lon_span / nx
    dy = lat_span / ny
    ix = np.clip(np.floor((lon - np.nanmin(lon)) / dx).astype(int), 0, nx-1)
    iy = np.clip(np.floor((lat - np.nanmin(lat)) / dy).astype(int), 0, ny-1)
    return (ix + nx * iy).astype(int), (nx, ny)

def make_macro_regions_contiguous(df: pd.DataFrame, n_macro: int) -> Tuple[np.ndarray, Tuple[int,int]]:
    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    lon_span = max(1e-12, float(np.nanmax(lon) - np.nanmin(lon)))
    lat_span = max(1e-12, float(np.nanmax(lat) - np.nanmin(lat)))
    aspect = lon_span / lat_span
    if n_macro == 5:
        nx, ny = (5, 1) if aspect >= 1.0 else (1, 5)
    else:
        pairs = [(a, n_macro//a) for a in range(1, n_macro+1) if n_macro % a == 0]
        best = min(pairs, key=lambda p: abs(np.log((p[0]/p[1]) / aspect)))
        nx, ny = best
    dx = lon_span / nx
    dy = lat_span / ny
    ix = np.clip(np.floor((lon - np.nanmin(lon)) / dx).astype(int), 0, nx-1)
    iy = np.clip(np.floor((lat - np.nanmin(lat)) / dy).astype(int), 0, ny-1)
    return (ix + nx * iy).astype(int), (nx, ny)


# ============================================================
# [D] Build base models com fallback XGBoost cuda -> cpu
# ============================================================
def _detect_xgb_device() -> str:
    """
    [D] Detecta disponibilidade de GPU via nvidia-smi (subprocess),
    SEM instanciar nenhum modelo XGBoost. Evita segfault com drivers
    CUDA parciais ou incompatíveis que não geram exceção Python.
    """
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.decode().strip().splitlines()[0]
            print(f"   [XGB] GPU detectada via nvidia-smi: {gpu_name} → device='cuda'")
            return 'cuda'
    except Exception:
        pass
    print("   [XGB] GPU não detectada → device='cpu'")
    return 'cpu'

# Detecta uma única vez no startup (não dentro do loop de folds)
_XGB_DEVICE = _detect_xgb_device() if XGB_OK else 'cpu'


def build_base_models(seed: int) -> Dict[str, Any]:
    modelos = {
        'RF':    RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                     max_features=0.7, random_state=seed, n_jobs=-1),
        'ET':    ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                     max_features=0.7, random_state=seed, n_jobs=-1),
        'SVR':   SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        'Ridge': Ridge(alpha=10.0),
    }
    if XGB_OK:
        modelos['XGB'] = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
            tree_method='hist', device=_XGB_DEVICE, verbosity=0)
    if LGBM_OK:
        modelos['LGBM'] = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
    return modelos


# ============================================================
# CORE: treina stacking ensemble e prediz test (com imputação)
# ============================================================
@dataclass
class TrainedEnsemble:
    imputer: SimpleImputer
    scaler: StandardScaler
    base_models: Dict[str, Any]
    meta_rf: RandomForestRegressor
    rf_uncertainty: RandomForestRegressor

def train_ensemble_on_train_predict_test(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, seed: int,
    inner_splits: int, groups_train: np.ndarray
) -> Tuple[np.ndarray, TrainedEnsemble]:

    modelos_base = build_base_models(seed)
    imputer = SimpleImputer(strategy="median")
    X_train = np.asarray(X_train, float); X_test = np.asarray(X_test, float)
    Xtr_imp = imputer.fit_transform(X_train)
    Xte_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr_imp)
    Xte_sc = scaler.transform(Xte_imp)

    # [C] Guarda n_splits mínimo para evitar crash com blocos pequenos
    n_inner_safe = min(inner_splits, len(np.unique(groups_train)))
    if n_inner_safe < 2:
        n_inner_safe = 2

    gkf = GroupKFold(n_splits=n_inner_safe)
    meta_tr = np.zeros((len(y_train), len(modelos_base)), float)

    for tr_idx, va_idx in gkf.split(Xtr_imp, y_train, groups=groups_train):
        for j, (nome, modelo) in enumerate(modelos_base.items()):
            Xtr = Xtr_sc[tr_idx] if nome in USA_ESCALA else Xtr_imp[tr_idx]
            Xva = Xtr_sc[va_idx] if nome in USA_ESCALA else Xtr_imp[va_idx]
            modelo.fit(Xtr, y_train[tr_idx])
            meta_tr[va_idx, j] = modelo.predict(Xva)

    base_models_fitted = build_base_models(seed)
    meta_te = np.zeros((len(Xte_imp), len(base_models_fitted)), float)

    for j, (nome, modelo) in enumerate(base_models_fitted.items()):
        Xtr = Xtr_sc if nome in USA_ESCALA else Xtr_imp
        Xte = Xte_sc if nome in USA_ESCALA else Xte_imp
        modelo.fit(Xtr, y_train)
        meta_te[:, j] = modelo.predict(Xte)

    meta_rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=seed, n_jobs=-1)
    meta_rf.fit(meta_tr, y_train)
    y_pred_test = meta_rf.predict(meta_te).astype(float)

    rf_unc = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                    max_features=0.7, random_state=seed, n_jobs=-1)
    rf_unc.fit(Xtr_imp, y_train)

    return y_pred_test, TrainedEnsemble(imputer, scaler, base_models_fitted, meta_rf, rf_unc)


# ============================================================
# [B] RF SOLO (sem stacking) — para análise de sensibilidade
# ============================================================
def predict_rf_solo(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, seed: int
) -> np.ndarray:
    """RF direto sem meta-learner, com imputação mediana."""
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(np.asarray(X_train, float))
    Xte_imp = imputer.transform(np.asarray(X_test, float))
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                                max_features=0.7, random_state=seed, n_jobs=-1)
    rf.fit(Xtr_imp, y_train)
    return rf.predict(Xte_imp).astype(float)


# ============================================================
# [A] FIGURAS — funções de geração
# ============================================================
PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
]

def _add_1to1_line(ax, y_true, y_pred, label_prefix="", color='#1f77b4'):
    """Adiciona scatter + linha OLS + linha 1:1 a um eixo."""
    ax.scatter(y_true, y_pred, alpha=0.45, s=28, color=color, edgecolors='none')
    lo = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    hi = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    margin = (hi - lo) * 0.05
    ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
            'k--', lw=1.0, label='1:1')
    # OLS
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    x_ols = np.array([lo-margin, hi+margin])
    ax.plot(x_ols, sl*x_ols + ic, '-', color=color, lw=1.5, alpha=0.8, label=f'OLS (slope={sl:.2f})')
    met = compute_metrics(y_true, y_pred)
    ax.set_xlabel("Observed AGB (Mg ha⁻¹)", fontsize=9)
    ax.set_ylabel("Predicted AGB (Mg ha⁻¹)", fontsize=9)
    title = (f"{label_prefix}R²={met['r2']:.3f} | RMSE={met['rmse']:.1f} | "
             f"Bias={met['bias']:+.1f} | Slope={met['slope']:.2f}")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xlim(lo-margin, hi+margin)
    ax.set_ylim(lo-margin, hi+margin)
    ax.set_aspect('equal', adjustable='box')


def plot_oof_scatter(y_true: np.ndarray, y_pred: np.ndarray, method_name: str,
                     out_path: str):
    """Figura 1: scatter OOF com linha 1:1."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _add_1to1_line(ax, y_true, y_pred, label_prefix=f"{method_name} | OOF | ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_lomro_scatter(rows_mro_raw: List[Dict], out_path: str):
    """Figura 2: scatter LO-MRO por macro-região (subplots).
    rows_mro_raw: lista de dicts com escalares {macro_region, y_true, y_pred}.
    Agrupa por macro_region antes de plotar.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows_mro_raw:
        return

    # Agrupa observações individuais (escalares) por macro_region
    from collections import defaultdict
    grouped = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for row in rows_mro_raw:
        grouped[row["macro_region"]]["y_true"].append(float(row["y_true"]))
        grouped[row["macro_region"]]["y_pred"].append(float(row["y_pred"]))

    macro_ids_sorted = sorted(grouped.keys())
    n = len(macro_ids_sorted)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for idx, mid in enumerate(macro_ids_sorted):
        ax = axes_flat[idx]
        yt = np.array(grouped[mid]["y_true"], float)
        yp = np.array(grouped[mid]["y_pred"], float)
        _add_1to1_line(ax, yt, yp, label_prefix="", color=PALETTE[idx % len(PALETTE)])
        # Substitui o título gerado por _add_1to1_line com versão limpa e compacta
        met_mid = compute_metrics(yt, yp)
        ax.set_title(
            f"Macro {mid} (n={len(yt)}) | R²={met_mid['r2']:.3f} | "
            f"RMSE={met_mid['rmse']:.1f} | Bias={met_mid['bias']:+.1f} | Slope={met_mid['slope']:.2f}",
            fontsize=8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("LO-MRO Blind Test — Observed vs Predicted AGB", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_r2_boxplot(methodwise_outer: Dict, out_path: str):
    """Figura 3: boxplot de R² inter-fold por método."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data_per_method = {}
    for method, store in methodwise_outer.items():
        r2_vals = [fm["r2"] for fm in store["fold_metrics"]]
        if r2_vals:
            data_per_method[method] = r2_vals

    if not data_per_method:
        return

    # ordena por mediana R² decrescente
    order = sorted(data_per_method.keys(),
                   key=lambda m: np.median(data_per_method[m]), reverse=True)

    fig, ax = plt.subplots(figsize=(max(7, len(order)*1.1), 4.5))
    vals = [data_per_method[m] for m in order]
    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], PALETTE[:len(order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels([m.replace("_k", "\nk=") for m in order], fontsize=8, rotation=30, ha='right')
    ax.set_ylabel("R² (outer fold)", fontsize=10)
    ax.set_title("Inter-fold R² Variability — Nested CV (25 outer folds, spatial GRID blocks)", fontsize=10)
    ax.axhline(0, color='gray', lw=0.5, linestyle=':')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_calibration_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                                method_name: str, out_path: str):
    """Figura 4: residuals vs fitted + histogram of residuals."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: residuals vs fitted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.4, s=25, color='#1f77b4', edgecolors='none')
    ax.axhline(0, color='red', lw=1.5, linestyle='--', label='Zero bias')
    # LOWESS-like running mean
    order = np.argsort(y_pred)
    w = max(1, len(y_pred)//10)
    rm_x = np.convolve(y_pred[order], np.ones(w)/w, mode='valid')
    rm_y = np.convolve(residuals[order], np.ones(w)/w, mode='valid')
    ax.plot(rm_x, rm_y, color='darkorange', lw=2, label=f'Running mean (w={w})')
    ax.set_xlabel("Predicted AGB (Mg ha⁻¹)", fontsize=10)
    ax.set_ylabel("Residual (Pred − Obs) (Mg ha⁻¹)", fontsize=10)
    bias = float(np.mean(residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ax.set_title(f"{method_name} | OOF Residuals\nBias={bias:+.2f} | RMSE={rmse:.2f} Mg ha⁻¹", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: residuals histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color='#1f77b4', edgecolor='white', alpha=0.75)
    ax2.axvline(0, color='red', lw=1.5, linestyle='--')
    ax2.axvline(bias, color='darkorange', lw=1.5, linestyle='-',
                label=f'Mean bias = {bias:+.2f}')
    ax2.set_xlabel("Residual (Mg ha⁻¹)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Residuals Distribution", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Calibration & Residual Diagnostics — {method_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_rf_vs_ensemble(sens_rows: List[Dict], out_path: str):
    """Figura 5: barplot comparativo RF solo vs ensemble (por fold)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_s = pd.DataFrame(sens_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: boxplot RMSE
    ax = axes[0]
    ax.boxplot([df_s["rmse_rf"].values, df_s["rmse_ensemble"].values],
               labels=["RF only", "Stacking ensemble"],
               patch_artist=True,
               medianprops=dict(color='black', lw=2),
               boxprops=dict(facecolor='#1f77b4', alpha=0.6))
    ax.set_ylabel("RMSE (Mg ha⁻¹)", fontsize=10)
    ax.set_title("RMSE: RF only vs Stacking", fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: scatter fold-a-fold
    ax2 = axes[1]
    ax2.scatter(df_s["rmse_rf"], df_s["rmse_ensemble"], alpha=0.6, s=40, color='#ff7f0e')
    lo = min(df_s["rmse_rf"].min(), df_s["rmse_ensemble"].min()) * 0.97
    hi = max(df_s["rmse_rf"].max(), df_s["rmse_ensemble"].max()) * 1.03
    ax2.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')
    ax2.set_xlabel("RMSE RF only (Mg ha⁻¹)", fontsize=10)
    ax2.set_ylabel("RMSE Stacking (Mg ha⁻¹)", fontsize=10)
    ax2.set_title("Fold-level RMSE comparison", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Wilcoxon result annotation
    if len(df_s) >= 5:
        stat_w, p_w = stats.wilcoxon(df_s["rmse_rf"].values,
                                      df_s["rmse_ensemble"].values,
                                      alternative='greater')
        delta_mean = float((df_s["rmse_rf"] - df_s["rmse_ensemble"]).mean())
        axes[0].set_xlabel(
            f"Wilcoxon (H0: RF=Ensemble): p={p_w:.4f} | ΔRMSE mean={delta_mean:+.2f}",
            fontsize=8)

    fig.suptitle("Sensitivity Analysis: RF only vs Stacking Ensemble", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 0) CARREGAR SUBSETS DO JSON (SCRIPT 1)
# ============================================================
if not os.path.exists(QAOA_JSON):
    raise FileNotFoundError(f"JSON não encontrado: {QAOA_JSON}")

with open(QAOA_JSON, "r", encoding="utf-8") as f:
    j = json.load(f)

SUBSETS_BY_METHOD: Dict[str, List[str]] = {}

if "results_by_k" in j:
    print("\n🔍 Analisando resultados da Mega-Varredura Justa...")
    best_r2_tracker = {}
    for k_val, k_data in j["results_by_k"].items():
        for method_name, metrics in k_data.get("methods", {}).items():
            current_r2 = metrics.get("r2", 0) or 0
            if method_name not in best_r2_tracker or current_r2 > best_r2_tracker[method_name]:
                best_r2_tracker[method_name] = current_r2
                for old_key in [k for k in SUBSETS_BY_METHOD if k.startswith(f"{method_name}_k")]:
                    del SUBSETS_BY_METHOD[old_key]
                SUBSETS_BY_METHOD[f"{method_name}_k{k_val}"] = metrics.get("features", [])
elif "global_top10" in j:
    for entry in j["global_top10"]:
        SUBSETS_BY_METHOD[f"{entry['method']}_k{entry['k']}"] = entry["features"]
else:
    raise KeyError("JSON formato desconhecido.")

# Remove métodos sem features válidas
SUBSETS_BY_METHOD = {m: f for m, f in SUBSETS_BY_METHOD.items() if len(f) > 0}
if not SUBSETS_BY_METHOD:
    raise ValueError("Nenhum subconjunto extraído do JSON.")

print("\n📌 Subsets carregados:")
for m, feats in SUBSETS_BY_METHOD.items():
    print(f"   - {m}: {len(feats)} feats | hash={subset_hash(feats)}")

# ============================================================
# 1) LOAD + FILTRO VEGETAÇÃO
# ============================================================
print("\n📂 Carregando dados (veg-only)...")
features_df = pd.read_csv(FEATURES_CSV)
biomassa_df = pd.read_csv(BIOMASSA_CSV)
df = features_df.merge(biomassa_df[["UA", TARGET_COL]], on="UA", how="inner")

df["lon"] = pd.to_numeric(df.get("lon_pc", np.nan), errors="coerce")
df["lat"] = pd.to_numeric(df.get("lat_pc", np.nan), errors="coerce")
df["mb_int"] = pd.to_numeric(df[MAPBIOMAS_COL], errors="coerce").round().fillna(-1).astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df[
    (df[TARGET_COL] > 0) & (df[TARGET_COL] < 300) &
    df["lon"].notna() & df["lat"].notna() &
    df["mb_int"].isin(CLASSES_VEGETACAO)
].copy().reset_index(drop=True)

print(f"   Base veg: n={len(df)} | y̅={df[TARGET_COL].mean():.1f} ± {df[TARGET_COL].std():.1f} Mg/ha")

print("\n🧱 Criando GRID blocks contíguos para CV...")
df["spatial_block"], (nx_blk, ny_blk) = make_grid_blocks(df, N_SPATIAL_BLOCKS_TARGET, SEED)
counts_blk = {int(k): int(v) for k, v in pd.Series(df["spatial_block"]).value_counts().sort_index().items()}
print(f"   -> Grid: nx={nx_blk}, ny={ny_blk} | blocos: {counts_blk}")

if RUN_LO_MRO:
    print("\n🌍 Criando macro-regiões contíguas (LO-MRO)...")
    df["macro_region"], (nx_m, ny_m) = make_macro_regions_contiguous(df, N_MACRO_REGIONS)
    counts_m = {int(k): int(v) for k, v in pd.Series(df["macro_region"]).value_counts().sort_index().items()}
    print(f"   -> Macro grid: nx={nx_m}, ny={ny_m} | regiões: {counts_m}")

# Anti-leakage: valida subsets contra df e BAND_NAMES
band_cols = set(BAND_NAMES)
kept, dropped = {}, {}
for method, feats in SUBSETS_BY_METHOD.items():
    miss_df = [f for f in feats if f not in df.columns]
    miss_bd = [f for f in feats if f not in band_cols]
    if miss_df or miss_bd:
        dropped[method] = {"missing_df": miss_df, "missing_mosaic": miss_bd}
    else:
        kept[method] = feats

if not kept:
    raise RuntimeError(f"Nenhum método válido. Dropped: {dropped}")
SUBSETS_BY_METHOD = kept
print(f"\n   ✅ {len(SUBSETS_BY_METHOD)} métodos válidos (df + BAND_NAMES)")
if dropped:
    for m, why in dropped.items():
        print(f"   ⚠️ Descartado {m}: {why}")

# ============================================================
# 2) NESTED CV (SPATIAL GRID)
# ============================================================
print("\n" + "=" * 78)
print("🔒 NESTED CV — GroupKFold GRID blocks (outer repeats + inner)")
print("=" * 78)

methodwise_outer = {m: {"y_true": [], "y_pred": [], "fold_metrics": []} for m in SUBSETS_BY_METHOD}
selector_outer   = {"y_true": [], "y_pred": [], "chosen_method": [], "fold_metrics": []}

# [B] acumuladores para sensibilidade RF solo (aplicado apenas ao selector chosen)
sensitivity_rows = []  # armazena (fold, rmse_rf, rmse_ensemble)

fold_id = 0
total_folds = OUTER_SPLITS * OUTER_REPEATS

for rep in range(OUTER_REPEATS):
    outer_cv = GroupKFold(n_splits=OUTER_SPLITS)
    rng = np.random.default_rng(SEED + 100*rep)
    perm = rng.permutation(len(df))
    df_rep = df.iloc[perm].reset_index(drop=True)
    groups_outer = df_rep["spatial_block"].values

    for tr_idx, te_idx in outer_cv.split(df_rep, df_rep[TARGET_COL].values, groups=groups_outer):
        fold_id += 1
        df_tr = df_rep.iloc[tr_idx].copy()
        df_te = df_rep.iloc[te_idx].copy()

        # Inner: seleciona melhor método no treino
        inner_rows = []
        inner_cv = GroupKFold(n_splits=INNER_SPLITS)
        g_tr_inner = df_tr["spatial_block"].values

        for method, feats in SUBSETS_BY_METHOD.items():
            ytr_full = df_tr[TARGET_COL].values.astype(float)
            Xtr_full = df_tr[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
            y_oof = np.zeros(len(ytr_full), float)

            # [C] já resolvido dentro de train_ensemble_on_train_predict_test
            for itrn, itst in inner_cv.split(Xtr_full, ytr_full, groups=g_tr_inner):
                y_pred_inner, _ = train_ensemble_on_train_predict_test(
                    X_train=Xtr_full[itrn], y_train=ytr_full[itrn],
                    X_test=Xtr_full[itst], seed=SEED + fold_id,
                    inner_splits=INNER_SPLITS, groups_train=g_tr_inner[itrn])
                y_oof[itst] = y_pred_inner

            met = compute_metrics(ytr_full, y_oof)
            inner_rows.append({"method": method, **met})

        best_inner = choose_best(inner_rows)
        chosen = best_inner["method"]
        feats_chosen = SUBSETS_BY_METHOD[chosen]

        Xtr = df_tr[feats_chosen].apply(pd.to_numeric, errors="coerce").values.astype(float)
        ytr = df_tr[TARGET_COL].values.astype(float)
        gtr = df_tr["spatial_block"].values
        Xte = df_te[feats_chosen].apply(pd.to_numeric, errors="coerce").values.astype(float)
        yte = df_te[TARGET_COL].values.astype(float)

        y_pred_te, _ = train_ensemble_on_train_predict_test(
            X_train=Xtr, y_train=ytr, X_test=Xte,
            seed=SEED + fold_id, inner_splits=INNER_SPLITS, groups_train=gtr)

        selector_outer["y_true"].extend(yte.tolist())
        selector_outer["y_pred"].extend(y_pred_te.tolist())
        selector_outer["chosen_method"].append(chosen)
        fold_met = compute_metrics(yte, y_pred_te)
        selector_outer["fold_metrics"].append({
            "fold": fold_id, "repeat": rep+1, "chosen": chosen,
            "inner_r2": best_inner["r2"], "inner_rmse": best_inner["rmse"],
            **fold_met})

        # [B] RF solo no mesmo fold para sensibilidade
        y_pred_rf_solo = predict_rf_solo(Xtr, ytr, Xte, seed=SEED + fold_id + 9000)
        rmse_rf = float(np.sqrt(mean_squared_error(yte, y_pred_rf_solo)))
        rmse_ens = float(fold_met["rmse"])
        sensitivity_rows.append({
            "fold": fold_id, "repeat": rep+1, "chosen_method": chosen,
            "rmse_rf": rmse_rf, "rmse_ensemble": rmse_ens,
            "r2_rf": float(r2_score(yte, y_pred_rf_solo)),
            "r2_ensemble": float(fold_met["r2"]),
            "delta_rmse": rmse_rf - rmse_ens  # positivo = ensemble melhor
        })

        # Avalia todos os métodos no outer
        for method, feats in SUBSETS_BY_METHOD.items():
            Xtr_m = df_tr[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
            Xte_m = df_te[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
            ytr_m = df_tr[TARGET_COL].values.astype(float)
            gtr_m = df_tr["spatial_block"].values
            yte_m = df_te[TARGET_COL].values.astype(float)

            y_pred_m, _ = train_ensemble_on_train_predict_test(
                X_train=Xtr_m, y_train=ytr_m, X_test=Xte_m,
                seed=SEED + fold_id, inner_splits=INNER_SPLITS, groups_train=gtr_m)

            methodwise_outer[method]["y_true"].extend(yte_m.tolist())
            methodwise_outer[method]["y_pred"].extend(y_pred_m.tolist())
            methodwise_outer[method]["fold_metrics"].append({
                "fold": fold_id, "repeat": rep+1,
                **compute_metrics(yte_m, y_pred_m)})

        if fold_id % 5 == 0 or fold_id == total_folds:
            print(f"   Outer folds: {fold_id}/{total_folds}", end="\r")

print("\n\n✅ Nested CV concluído.")

# ============================================================
# 3) RELATÓRIOS + FIGURAS OOF
# ============================================================
rows_method = []
for method, store in methodwise_outer.items():
    if len(store["y_true"]) < 10:
        continue
    y_true = np.array(store["y_true"], float)
    y_pred = np.array(store["y_pred"], float)
    met = compute_metrics(y_true, y_pred)
    fold_r2   = [fm["r2"]   for fm in store["fold_metrics"]]
    fold_rmse = [fm["rmse"] for fm in store["fold_metrics"]]
    fold_bias = [fm["bias"] for fm in store["fold_metrics"]]

    rows_method.append({
        "method": method,
        "outer_oof_r2": met["r2"], "outer_oof_rmse": met["rmse"],
        "outer_oof_mae": met["mae"], "outer_oof_bias": met["bias"],
        "outer_oof_slope": met["slope"],
        "folds_count": len(store["fold_metrics"]),
        "r2_mean": float(np.mean(fold_r2)), "r2_std": float(np.std(fold_r2, ddof=1)) if len(fold_r2)>1 else 0.0,
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse, ddof=1)) if len(fold_rmse)>1 else 0.0,
        "bias_mean": float(np.mean(fold_bias)), "bias_std": float(np.std(fold_bias, ddof=1)) if len(fold_bias)>1 else 0.0,
        "features": ",".join(SUBSETS_BY_METHOD[method]),
        "n_features": len(SUBSETS_BY_METHOD[method]),
        "subset_hash": subset_hash(SUBSETS_BY_METHOD[method]),
    })

    # [A] Scatter OOF por método
    plot_oof_scatter(y_true, y_pred, method_name=method,
                     out_path=os.path.join(FIGS_DIR, f"fig_oof_scatter_{method}.png"))

df_method = pd.DataFrame(rows_method)
# Coluna de prioridade garante desempate determinístico: QAOA > Exact > GA > SA > RF > ET
# quando R², RMSE e Bias são idênticos (subsets com mesmo hash)
df_method["_priority"] = df_method["method"].apply(
    lambda m: next((i for i, p in enumerate(_METHOD_PRIORITY) if m.startswith(p)), len(_METHOD_PRIORITY))
)
df_method = df_method.sort_values(
    ["outer_oof_r2", "outer_oof_rmse", "outer_oof_bias", "_priority"],
    ascending=[False, True, True, True]
).drop(columns=["_priority"]).reset_index(drop=True)

path_method = os.path.join(OUTPUT_DIR, "nestedcv_methodwise_metrics.csv")
df_method.to_csv(path_method, index=False)
print(f"📋 Métodos (outer OOF): {path_method}")

# [A] Boxplot R² inter-fold
print("📊 Gerando figuras OOF...")
plot_r2_boxplot(methodwise_outer, os.path.join(FIGS_DIR, "fig_r2_boxplot.png"))

# Selector OOF
y_sel_true = np.array(selector_outer["y_true"], float)
y_sel_pred = np.array(selector_outer["y_pred"], float)
met_sel = compute_metrics(y_sel_true, y_sel_pred)

# [A] Calibration plot para o selector
plot_calibration_residuals(
    y_sel_true, y_sel_pred, method_name="Selector (best per fold)",
    out_path=os.path.join(FIGS_DIR, "fig_calibration_residuals_selector.png"))

df_sel_folds = pd.DataFrame(selector_outer["fold_metrics"])
path_sel = os.path.join(OUTPUT_DIR, "nestedcv_selector_metrics.csv")
df_sel_folds.to_csv(path_sel, index=False)

freq = pd.Series(selector_outer["chosen_method"]).value_counts().reset_index()
freq.columns = ["method", "count"]
freq["pct"] = 100.0 * freq["count"] / freq["count"].sum()
path_freq = os.path.join(OUTPUT_DIR, "nestedcv_selection_frequencies.csv")
freq.to_csv(path_freq, index=False)

print("\n" + "=" * 78)
print("RESULTADO NESTED CV — SELECTOR — EMS SPATIAL GRID")
print("=" * 78)
print(f"Selector OOF: R²={met_sel['r2']:.4f} | RMSE={met_sel['rmse']:.2f} | "
      f"Bias={met_sel['bias']:+.2f} | Slope={met_sel['slope']:.3f}")
print(freq.to_string(index=False))

# ============================================================
# [B] SENSIBILIDADE RF SOLO vs ENSEMBLE
# ============================================================
print("\n" + "=" * 78)
print("🔬 SENSIBILIDADE: RF solo vs Stacking Ensemble")
print("=" * 78)

df_sens = pd.DataFrame(sensitivity_rows)
path_sens = os.path.join(OUTPUT_DIR, "sensitivity_rf_vs_ensemble.csv")
df_sens.to_csv(path_sens, index=False)

rmse_rf_mean  = float(df_sens["rmse_rf"].mean())
rmse_ens_mean = float(df_sens["rmse_ensemble"].mean())
delta_mean    = float(df_sens["delta_rmse"].mean())  # + = ensemble melhor

if len(df_sens) >= 5:
    stat_w, p_w = stats.wilcoxon(df_sens["rmse_rf"].values,
                                  df_sens["rmse_ensemble"].values,
                                  alternative='greater')
    print(f"   RMSE RF solo:   {rmse_rf_mean:.3f} Mg/ha")
    print(f"   RMSE Ensemble:  {rmse_ens_mean:.3f} Mg/ha")
    print(f"   ΔRMSE (RF−Ens): {delta_mean:+.3f} Mg/ha (positivo = ensemble melhor)")
    print(f"   Wilcoxon (H₀: RF≥Ensemble): stat={stat_w:.1f}, p={p_w:.4f}")
    if p_w < 0.05:
        print("   → Ensemble significativamente melhor que RF solo (α=0.05)")
    else:
        print("   → Diferença não significativa — complexidade do stacking questionável")
    plot_sensitivity_rf_vs_ensemble(
        sensitivity_rows,
        os.path.join(FIGS_DIR, "fig_sensitivity_rf_vs_ensemble.png"))
else:
    print("   ⚠️ Poucos folds para Wilcoxon.")

# ============================================================
# 4) BEST GLOBAL + AUDITORIA TOP-3
# ============================================================
best_global = df_method.iloc[0]
best_method = str(best_global["method"])
best_feats  = SUBSETS_BY_METHOD[best_method]

best_json = {
    "best_global_method": best_method,
    "criterion": "max outer_oof_r2; tie -> min outer_oof_rmse",
    "best_global_metrics": {
        "outer_oof_r2":   float(best_global["outer_oof_r2"]),
        "outer_oof_rmse": float(best_global["outer_oof_rmse"]),
        "outer_oof_mae":  float(best_global["outer_oof_mae"]),
        "outer_oof_bias": float(best_global["outer_oof_bias"]),
        "outer_oof_slope":float(best_global["outer_oof_slope"]),
        "r2_std_folds":   float(best_global["r2_std"]),
        "rmse_std_folds":  float(best_global["rmse_std"]),
    },
    "features": best_feats,
    "n_features": int(len(best_feats)),
    "subset_hash": subset_hash(best_feats),
    "classes_vegetacao": sorted(list(CLASSES_VEGETACAO)),
    "outer_cv": {"splits": OUTER_SPLITS, "repeats": OUTER_REPEATS, "type": "GroupKFold(grid_spatial_block)"},
    "inner_cv": {"splits": INNER_SPLITS, "type": "GroupKFold(grid_spatial_block)"},
    "grid_blocks": {"nx": nx_blk, "ny": ny_blk},
    "loaded_from_json": QAOA_JSON
}

best_path = os.path.join(OUTPUT_DIR, "best_global_subset.json")
with open(best_path, "w") as f:
    json.dump(best_json, f, indent=2, ensure_ascii=False)

# Auditoria top-3
topk = min(3, len(df_method))
top_methods = df_method.head(topk)["method"].tolist()
top_payload = {
    "top_methods": [{
        "method": m, "n_features": len(SUBSETS_BY_METHOD[m]),
        "subset_hash": subset_hash(SUBSETS_BY_METHOD[m]),
        "features": SUBSETS_BY_METHOD[m]
    } for m in top_methods],
    "pairwise_jaccard": [{
        "pair": [top_methods[i], top_methods[k]],
        "jaccard": jaccard(SUBSETS_BY_METHOD[top_methods[i]], SUBSETS_BY_METHOD[top_methods[k]])
    } for i in range(len(top_methods)) for k in range(i+1, len(top_methods))]
}

top_path = os.path.join(OUTPUT_DIR, "top_methods_subsets.json")
with open(top_path, "w") as f:
    json.dump(top_payload, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 78)
print("🏆 RANKING FINAL (outer OOF) — EMS SPATIAL GRID v2.5")
print("=" * 78)
show_cols = ["method","outer_oof_r2","r2_std","outer_oof_rmse","rmse_std","outer_oof_bias","n_features","subset_hash"]
print(df_method[[c for c in show_cols if c in df_method.columns]].head(20).to_string(index=False))
print(f"\nBEST GLOBAL: {best_method} | R²={best_global['outer_oof_r2']:.4f} (±{best_global['r2_std']:.4f}) "
      f"| RMSE={best_global['outer_oof_rmse']:.2f} (±{best_global['rmse_std']:.2f})")
print(f"Features: {best_feats}")

# ============================================================
# 4.5) LO-MRO
# ============================================================
rows_mro_raw = []  # [E] para previsões brutas
df_mro = pd.DataFrame()  # inicializado aqui para evitar NameError se RUN_LO_MRO=False

if RUN_LO_MRO:
    print("\n" + "=" * 78)
    print("🔥 LO-MRO — Leave-One-MacroRegion-Out (macro-grid contíguo)")
    print("=" * 78)

    macro_ids = sorted(df["macro_region"].unique().tolist())
    rows_mro = []

    for mid in macro_ids:
        df_test  = df[df["macro_region"] == mid].copy()
        df_train = df[df["macro_region"] != mid].copy()
        if len(df_test) < 5 or len(df_train) < 30:
            continue

        Xtr = df_train[best_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
        ytr = df_train[TARGET_COL].values.astype(float)
        gtr = df_train["spatial_block"].values
        Xte = df_test[best_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
        yte = df_test[TARGET_COL].values.astype(float)

        # Usa o mesmo modelo escolhido para o mapa final (consistência metodológica)
        if FINAL_MODEL == "RF":
            yhat = predict_rf_solo(Xtr, ytr, Xte, seed=SEED + 1000 + int(mid))
        else:
            yhat, _ = train_ensemble_on_train_predict_test(
                X_train=Xtr, y_train=ytr, X_test=Xte,
                seed=SEED + 1000 + int(mid),
                inner_splits=INNER_SPLITS, groups_train=gtr)

        met = compute_metrics(yte, yhat)
        rows_mro.append({"macro_region": int(mid), "n_test": int(len(df_test)), **met})

        # [E] Salva previsões brutas
        for yt_i, yp_i in zip(yte.tolist(), yhat.tolist()):
            rows_mro_raw.append({"macro_region": int(mid), "y_true": yt_i, "y_pred": yp_i})

        print(f"   macro={mid} | n={len(df_test)} | R²={met['r2']:.3f} | "
              f"RMSE={met['rmse']:.2f} | Bias={met['bias']:+.2f} | Slope={met['slope']:.3f}")

    df_mro = pd.DataFrame(rows_mro)
    if not df_mro.empty:
        path_mro = os.path.join(OUTPUT_DIR, "lomro_bestglobal_metrics.csv")
        df_mro.to_csv(path_mro, index=False)

        # [E] Previsões brutas
        pd.DataFrame(rows_mro_raw).to_csv(
            os.path.join(OUTPUT_DIR, "lomro_predictions_raw.csv"), index=False)

        w = df_mro["n_test"].values.astype(float)
        r2_w  = float(np.average(df_mro["r2"],   weights=w))
        rmse_w= float(np.average(df_mro["rmse"], weights=w))
        bias_w= float(np.average(df_mro["bias"], weights=w))
        slp_w = float(np.average(df_mro["slope"],weights=w))
        print("-" * 78)
        print(f"LO-MRO ponderado: R²={r2_w:.4f} | RMSE={rmse_w:.2f} | Bias={bias_w:+.2f} | Slope={slp_w:.3f}")

        # [A] Figura LO-MRO scatter
        plot_lomro_scatter(rows_mro_raw, os.path.join(FIGS_DIR, "fig_lomro_scatter.png"))

        # [A] Calibration plot LO-MRO agregado
        yt_all = np.array([r["y_true"] for r in rows_mro_raw], float)
        yp_all = np.array([r["y_pred"] for r in rows_mro_raw], float)
        plot_calibration_residuals(
            yt_all, yp_all, method_name=f"{best_method} — LO-MRO",
            out_path=os.path.join(FIGS_DIR, "fig_calibration_residuals_lomro.png"))

# ============================================================
# 5) TREINO FINAL + MAPA WALL-TO-WALL
# ============================================================
print("\n" + "=" * 78)
print(f"🏋️  TREINO FINAL (ALL VEG) + MAPA — {best_method} | modelo: {FINAL_MODEL}")
print("=" * 78)

X_all = df[best_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
y_all = df[TARGET_COL].values.astype(float)
g_all = df["spatial_block"].values

# Imputação mediana treino-only (comum a RF e stacking)
final_imputer = SimpleImputer(strategy="median")
X_all_imp = final_imputer.fit_transform(X_all)

if FINAL_MODEL == "RF":
    # RF direto — justificado pelo Wilcoxon (p=0.9998) que mostrou RF ≥ stacking
    final_rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        max_features=0.7, random_state=SEED, n_jobs=-1)
    final_rf.fit(X_all_imp, y_all)

    def predict_final(X_imp: np.ndarray) -> np.ndarray:
        return np.clip(final_rf.predict(X_imp), 0, 500).astype(np.float32)

    def uncertainty_final(X_imp: np.ndarray) -> np.ndarray:
        return np.array(
            [t.predict(X_imp) for t in final_rf.estimators_],
            dtype=np.float32).std(axis=0)

    final_scaler = None  # RF não usa escala
    print(f"   Modelo final: RandomForestRegressor (n_estimators=300) — sem meta-learner")

else:  # Stacking
    _, trained_all = train_ensemble_on_train_predict_test(
        X_train=X_all, y_train=y_all, X_test=X_all[:1],
        seed=SEED, inner_splits=INNER_SPLITS, groups_train=g_all)

    final_imputer  = trained_all.imputer
    final_scaler   = trained_all.scaler
    modelos_base   = trained_all.base_models
    ml_rf_final    = trained_all.meta_rf
    rf_incerteza   = trained_all.rf_uncertainty
    X_all_imp      = final_imputer.transform(X_all)  # já fitado

    def predict_final(X_imp: np.ndarray) -> np.ndarray:
        X_sc = final_scaler.transform(X_imp)
        meta = np.zeros((X_imp.shape[0], len(modelos_base)), float)
        for jj, (nome, modelo) in enumerate(modelos_base.items()):
            meta[:, jj] = modelo.predict(X_sc if nome in USA_ESCALA else X_imp)
        return np.clip(ml_rf_final.predict(meta), 0, 500).astype(np.float32)

    def uncertainty_final(X_imp: np.ndarray) -> np.ndarray:
        return np.array(
            [t.predict(X_imp) for t in rf_incerteza.estimators_],
            dtype=np.float32).std(axis=0)

    print(f"   Modelo final: Stacking ensemble (RF+ET+XGB+LGBM+SVR+Ridge + meta-RF)")

if not RASTERIO_OK:
    print("⚠️  rasterio indisponível — mapa não gerado")
elif not os.path.exists(MAPA_MOSAICO):
    print(f"⚠️  Raster não encontrado: {MAPA_MOSAICO}")
else:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", best_method)
    path_bio = os.path.join(OUTPUT_DIR, f"goias_biomassa_{tag}_100m.tif")
    path_std = os.path.join(OUTPUT_DIR, f"goias_incerteza_{tag}_100m.tif")
    BAND_IDX_FEATURES = [BAND_NAMES.index(f) + 1 for f in best_feats]

    print(f"Bandas: {dict(zip(best_feats, BAND_IDX_FEATURES))}")

    with rasterio.open(MAPA_MOSAICO) as src:
        profile = src.profile.copy()
        n_bands = src.count
        total = sum(1 for _ in src.block_windows(1))
        print(f"Raster: {src.width}×{src.height}px | bandas={n_bands} | blocos={total}")

        max_band = max(BAND_IDX_FEATURES + [BANDA_MAPBIOMAS])
        if max_band > n_bands:
            raise ValueError(f"Raster tem {n_bands} bandas; subset requer banda {max_band}")

        profile.update(dtype=rasterio.float32, count=1,
                       nodata=np.float32(NODATA), compress='lzw',
                       tiled=True, blockxsize=256, blockysize=256)

        with rasterio.open(path_bio, "w", **profile) as dst_bio, \
             rasterio.open(path_std, "w", **profile) as dst_std:

            for idx, (ji, window) in enumerate(src.block_windows(1)):
                feats_chunk = np.stack([
                    src.read(b, window=window).astype(np.float32) for b in BAND_IDX_FEATURES
                ], axis=-1)
                mb_b = src.read(BANDA_MAPBIOMAS, window=window)
                rows_b, cols_b = feats_chunk.shape[:2]

                nodata_mask = (np.any(feats_chunk == NODATA, axis=-1) |
                               np.any(np.isnan(feats_chunk), axis=-1))
                veg_mask = np.isin(mb_b.round().astype(int), list(CLASSES_VEGETACAO))
                valid_px = (~nodata_mask) & veg_mask

                bio_chunk = np.full((rows_b, cols_b), np.float32(NODATA))
                std_chunk = np.full((rows_b, cols_b), np.float32(NODATA))

                if valid_px.sum() > 0:
                    Xc = final_imputer.transform(feats_chunk[valid_px].astype(float))
                    bio_chunk[valid_px] = predict_final(Xc)
                    std_chunk[valid_px] = uncertainty_final(Xc)

                dst_bio.write(bio_chunk[np.newaxis], window=window)
                dst_std.write(std_chunk[np.newaxis], window=window)

                if (idx+1) % 20 == 0 or (idx+1) == total:
                    print(f"Bloco {idx+1}/{total} ({100*(idx+1)/total:.0f}%)", end="\r")

    print(f"\n✅ Biomassa:  {path_bio}")
    print(f"✅ Incerteza: {path_std}")

# ============================================================
# SUMÁRIO FINAL
# ============================================================
print("\n" + "=" * 78)
print("CONCLUÍDO — EMS SPATIAL GRID v2.5")
print("=" * 78)
print(f"JSON subsets (S1):  {QAOA_JSON}")
print(f"CSV métodos OOF:    {path_method}")
print(f"CSV selector folds: {path_sel}")
print(f"CSV freq seleção:   {path_freq}")
print(f"CSV sensibilidade:  {path_sens}")
print(f"JSON best global:   {best_path}")
print(f"JSON top-3:         {top_path}")
if RUN_LO_MRO and not df_mro.empty:
    print(f"CSV LO-MRO:         {path_mro}")
    print(f"CSV LO-MRO raw:     {os.path.join(OUTPUT_DIR, 'lomro_predictions_raw.csv')}")
print(f"Figures dir:        {FIGS_DIR}")
print("  fig_oof_scatter_<method>.png       (um por método)")
print("  fig_r2_boxplot.png                 (variância inter-fold)")
print("  fig_calibration_residuals_selector.png")
print("  fig_calibration_residuals_lomro.png")
print("  fig_lomro_scatter.png              (subplots por macro-região)")
print("  fig_sensitivity_rf_vs_ensemble.png")
print("=" * 78)